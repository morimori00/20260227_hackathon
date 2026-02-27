from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame


@dataclass
class Metrics:
    mae: float
    rmse: float


def load_data(path: str, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError("Unsupported file type. Use .csv, .xlsx, or .xls")


def clean_data(
    df: pd.DataFrame,
    timestamp_col: str,
    *,
    fill_method: str = "ffill",
    date_format: str | None = None,
    dayfirst: bool = False,
) -> pd.DataFrame:
    """
    Parse/sort by timestamp and handle missing values.

    fill_method:
    - "ffill": forward fill then backward fill
    - "drop": drop rows containing any missing values
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")

    out = df.copy()
    out[timestamp_col] = pd.to_datetime(
        out[timestamp_col],
        errors="coerce",
        format=date_format,
        dayfirst=dayfirst,
    )
    out = out.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    out = out.set_index(timestamp_col)

    if fill_method == "ffill":
        out = out.ffill().bfill()
    elif fill_method == "drop":
        out = out.dropna()
    else:
        raise ValueError("fill_method must be 'ffill' or 'drop'")

    return out


def make_features(
    df: pd.DataFrame,
    target_col: str,
    *,
    lags: tuple[int, ...] = (1, 2, 3, 24),
    rolling_window: int = 24,
) -> pd.DataFrame:
    """Add lag/rolling/time features for time-series modeling."""
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    out = df.copy()
    # Bank-style files often store numbers with commas/currency symbols as text.
    if not pd.api.types.is_numeric_dtype(out[target_col]):
        cleaned = (
            out[target_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(r"[^\d\.\-]", "", regex=True)
        )
        out[target_col] = pd.to_numeric(cleaned, errors="coerce")
        out = out.dropna(subset=[target_col])

    for lag in lags:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)

    out[f"{target_col}_roll_mean_{rolling_window}"] = (
        out[target_col].rolling(rolling_window).mean()
    )
    out[f"{target_col}_roll_std_{rolling_window}"] = (
        out[target_col].rolling(rolling_window).std()
    )

    out["hour"] = out.index.hour
    out["day_of_week"] = out.index.dayofweek
    out["month"] = out.index.month

    # Drop rows where lag/rolling features are not yet available.
    out = out.dropna()
    return out


def train_val_split(df: pd.DataFrame, split_date: str) -> SplitResult:
    """Split by date to avoid leakage from future to past."""
    split_ts = pd.to_datetime(split_date)
    train = df[df.index < split_ts].copy()
    val = df[df.index >= split_ts].copy()

    if train.empty or val.empty:
        min_dt = df.index.min()
        max_dt = df.index.max()
        raise ValueError(
            "Split produced an empty train or validation set. "
            f"Data date range is {min_dt.date()} to {max_dt.date()}. "
            f"Choose a split date strictly inside that range."
        )

    return SplitResult(train=train, val=val)


def evaluate_naive_baseline(
    train_df: pd.DataFrame, val_df: pd.DataFrame, target_col: str
) -> Metrics:
    """
    Naive baseline:
    predict each validation point using previous timestep value (lag-1).
    """
    lag_col = f"{target_col}_lag_1"
    if lag_col not in val_df.columns:
        raise ValueError(f"Missing required lag feature: {lag_col}")

    # Use precomputed lag feature to avoid expensive index lookups on duplicate dates.
    preds = val_df[lag_col]
    actual = val_df[target_col]

    valid = preds.notna() & actual.notna()
    preds = preds[valid]
    actual = actual[valid]

    if preds.empty:
        raise ValueError("No valid points to evaluate baseline.")

    err = actual - preds
    mae = float(err.abs().mean())
    rmse = float(math.sqrt((err**2).mean()))
    return Metrics(mae=mae, rmse=rmse)


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-series preprocessing starter")
    parser.add_argument("--data", required=True, help="Path to CSV")
    parser.add_argument(
        "--sheet-name",
        default=0,
        help="Excel sheet name or index (ignored for CSV). Default: 0",
    )
    parser.add_argument("--timestamp-col", required=True, help="Timestamp column name")
    parser.add_argument("--target-col", required=True, help="Target column name")
    parser.add_argument(
        "--split-date",
        required=True,
        help="Validation split date, e.g. 2025-01-01",
    )
    parser.add_argument(
        "--date-format",
        default=None,
        help="Optional datetime format, e.g. %%d-%%b-%%y for 29-Jun-17",
    )
    parser.add_argument(
        "--dayfirst",
        action="store_true",
        help="Parse day first in dates (e.g. 29-06-2017).",
    )
    args = parser.parse_args()

    print("Loading data...", flush=True)
    sheet_name: str | int
    sheet_name = int(args.sheet_name) if str(args.sheet_name).isdigit() else args.sheet_name
    df = load_data(args.data, sheet_name=sheet_name)
    print("Cleaning data...", flush=True)
    df = clean_data(
        df,
        args.timestamp_col,
        date_format=args.date_format,
        dayfirst=args.dayfirst,
    )
    print("Building features...", flush=True)
    df = make_features(df, args.target_col)
    print("Splitting train/validation...", flush=True)
    split = train_val_split(df, args.split_date)
    print("Evaluating baseline...", flush=True)
    metrics = evaluate_naive_baseline(split.train, split.val, args.target_col)

    print("Train shape:", split.train.shape)
    print("Validation shape:", split.val.shape)
    print("Columns:", list(df.columns))
    print(f"Naive baseline MAE: {metrics.mae:,.4f}")
    print(f"Naive baseline RMSE: {metrics.rmse:,.4f}")


if __name__ == "__main__":
    main()
