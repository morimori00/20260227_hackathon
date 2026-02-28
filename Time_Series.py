from __future__ import annotations

import argparse
import json
import math
import os
import urllib.parse
import urllib.request
from urllib.error import HTTPError
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


@dataclass
class WatsonConfig:
    apikey: str
    wml_url: str
    deployment_id: str
    version: str = "2021-05-01"


@dataclass
class NgrokConfig:
    base_url: str
    anomaly_threshold: float = 0.4


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


def _post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
    body = json.dumps(payload).encode("utf-8")
    merged_headers = {"Content-Type": "application/json", **headers}
    req = urllib.request.Request(url, data=body, headers=merged_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Watson scoring request failed ({e.code} {e.reason}). Details: {details}"
        ) from e


def _get_json(url: str, headers: dict[str, str] | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET request failed ({e.code} {e.reason}). Details: {details}") from e


def get_ibm_iam_token(apikey: str) -> str:
    if not apikey or apikey == "your_ibm_cloud_api_key":
        raise ValueError("Replace IBM_CLOUD_API_KEY placeholder with your real IBM API key.")

    token_url = "https://iam.cloud.ibm.com/identity/token"
    payload = urllib.parse.urlencode(
        {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": apikey,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    req = urllib.request.Request(token_url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"IAM token request failed ({e.code} {e.reason}). Details: {details}"
        ) from e
    return data["access_token"]


def score_with_watson(
    df: pd.DataFrame,
    *,
    target_col: str,
    watson: WatsonConfig,
    max_rows: int = 200,
) -> pd.DataFrame:
    """
    Score rows using an IBM Watson ML deployment.

    Sends feature columns (all columns except target) in Watson's expected format:
    {"input_data": [{"fields": [...], "values": [[...], ...]}]}
    """
    feature_cols = [c for c in df.columns if c != target_col]
    score_df = df[feature_cols].copy()
    score_df = score_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    score_df = score_df.head(max_rows)

    token = get_ibm_iam_token(watson.apikey)
    url = (
        f"{watson.wml_url.rstrip('/')}/ml/v4/deployments/"
        f"{watson.deployment_id}/predictions?version={watson.version}"
    )
    payload = {
        "input_data": [
            {
                "fields": feature_cols,
                "values": score_df.values.tolist(),
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = _post_json(url, payload, headers)

    # Most Watson responses return predictions[0].values.
    pred_values: list = []
    predictions = resp.get("predictions", [])
    if predictions and isinstance(predictions[0], dict):
        pred_values = predictions[0].get("values", [])

    # Fall back to generic key if deployment payload differs.
    if not pred_values and "values" in resp:
        pred_values = resp["values"]

    if not pred_values:
        raise ValueError(f"Watson response did not contain prediction values: {resp}")

    out = score_df.copy()
    # Common Watson output shape is [[prediction], ...] or [prediction, ...]
    first = pred_values[0]
    if isinstance(first, list):
        out["watson_prediction"] = [row[0] if row else None for row in pred_values]
    else:
        out["watson_prediction"] = pred_values
    return out


def check_ngrok_health(ngrok: NgrokConfig) -> dict:
    return _get_json(f"{ngrok.base_url.rstrip('/')}/health")


def score_with_ngrok(
    df: pd.DataFrame,
    *,
    target_col: str,
    ngrok: NgrokConfig,
    window_size: int = 60,
    max_windows: int = 5,
) -> pd.DataFrame:
    """
    Score sliding windows against ngrok /predict endpoint.
    Expected payload:
    {"series": [...], "anomaly_threshold": 0.4}
    """
    series = pd.to_numeric(df[target_col], errors="coerce").dropna().astype(float).tolist()
    if len(series) < window_size:
        raise ValueError(
            f"Need at least {window_size} numeric target points for ngrok scoring."
        )

    rows = []
    start = len(series) - window_size
    windows_sent = 0
    for i in range(start, len(series) - window_size + 1):
        if windows_sent >= max_windows:
            break
        window = series[i : i + window_size]
        payload = {"series": window, "anomaly_threshold": ngrok.anomaly_threshold}
        resp = _post_json(f"{ngrok.base_url.rstrip('/')}/predict", payload, headers={})
        rows.append(
            {
                "window_start_idx": i,
                "window_end_idx": i + window_size - 1,
                "response_json": json.dumps(resp),
            }
        )
        windows_sent += 1

    return pd.DataFrame(rows)


def parse_ngrok_predictions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten ngrok response_json into analytics-friendly columns."""
    rows: list[dict] = []
    for _, row in raw_df.iterrows():
        payload = {}
        try:
            payload = json.loads(str(row.get("response_json", "{}")))
        except json.JSONDecodeError:
            payload = {}
        rows.append(
            {
                "window_start_idx": int(row.get("window_start_idx", -1)),
                "window_end_idx": int(row.get("window_end_idx", -1)),
                "anomaly_score": payload.get("anomaly_score"),
                "cluster": payload.get("cluster"),
                "confidence": payload.get("confidence"),
                "is_anomaly": payload.get("is_anomaly"),
                "pattern_name": payload.get("pattern_name"),
                "series_length": payload.get("series_length"),
                "model_seq_len": payload.get("model_seq_len"),
            }
        )
    return pd.DataFrame(rows)


def plot_ngrok_anomalies(
    val_df: pd.DataFrame,
    *,
    target_col: str,
    parsed_df: pd.DataFrame,
    out_path: str,
) -> None:
    """Plot validation target series and mark predicted anomaly endpoints."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: python3 -m pip install matplotlib"
        ) from e

    series_df = pd.DataFrame({"value": pd.to_numeric(val_df[target_col], errors="coerce")})
    series_df = series_df.dropna().reset_index()
    # First column is the former index (could be named "index", "DATE", etc.).
    ts_col = series_df.columns[0]
    series_df = series_df.rename(columns={ts_col: "timestamp"})
    series_df["row_idx"] = range(len(series_df))

    is_anomaly = (
        parsed_df["is_anomaly"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )
    anom = parsed_df[is_anomaly].copy()
    if anom.empty:
        anom_points = pd.DataFrame(columns=["timestamp", "value"])
    else:
        anom_points = anom.merge(
            series_df[["row_idx", "timestamp", "value"]],
            left_on="window_end_idx",
            right_on="row_idx",
            how="left",
        )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series_df["timestamp"], series_df["value"], linewidth=1.2, label=target_col)
    if not anom_points.empty:
        ax.scatter(
            anom_points["timestamp"],
            anom_points["value"],
            color="red",
            s=35,
            label="Predicted anomaly",
            zorder=3,
        )
    ax.set_title("Validation Series with Predicted Anomalies")
    ax.set_xlabel("Time")
    ax.set_ylabel(target_col)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


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
    parser.add_argument(
        "--use-watson",
        action="store_true",
        help="Score validation rows using an IBM Watson ML deployment.",
    )
    parser.add_argument(
        "--watson-url",
        default=None,
        help="IBM Watson ML base URL, e.g. https://us-south.ml.cloud.ibm.com",
    )
    parser.add_argument(
        "--watson-deployment-id",
        default=None,
        help="Watson deployment ID for scoring.",
    )
    parser.add_argument(
        "--watson-version",
        default="2021-05-01",
        help="Watson API version query parameter.",
    )
    parser.add_argument(
        "--watson-api-key-env",
        default="IBM_CLOUD_API_KEY",
        help="Environment variable containing IBM Cloud API key.",
    )
    parser.add_argument(
        "--watson-max-rows",
        type=int,
        default=200,
        help="Max validation rows to send to Watson scoring endpoint.",
    )
    parser.add_argument(
        "--watson-output",
        default="results/watson_predictions.csv",
        help="Where to save Watson scoring output CSV.",
    )
    parser.add_argument(
        "--use-ngrok",
        action="store_true",
        help="Score using teammate ngrok API (/health and /predict).",
    )
    parser.add_argument(
        "--ngrok-url",
        default=None,
        help="Ngrok base URL, e.g. https://example.ngrok-free.dev",
    )
    parser.add_argument(
        "--ngrok-threshold",
        type=float,
        default=0.4,
        help="anomaly_threshold sent to /predict.",
    )
    parser.add_argument(
        "--ngrok-window-size",
        type=int,
        default=60,
        help="Series window length sent to /predict.",
    )
    parser.add_argument(
        "--ngrok-max-windows",
        type=int,
        default=5,
        help="Max windows to score (kept small for quick testing).",
    )
    parser.add_argument(
        "--ngrok-output",
        default="results/ngrok_predictions.csv",
        help="Where to save ngrok prediction responses.",
    )
    parser.add_argument(
        "--ngrok-parsed-output",
        default="results/ngrok_predictions_parsed.csv",
        help="Where to save flattened ngrok prediction columns.",
    )
    parser.add_argument(
        "--ngrok-plot-output",
        default="results/ngrok_anomalies.png",
        help="Where to save anomaly plot image.",
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

    if args.use_watson:
        if not args.watson_url or not args.watson_deployment_id:
            raise ValueError(
                "--use-watson requires --watson-url and --watson-deployment-id"
            )
        if args.watson_deployment_id == "YOUR_DEPLOYMENT_ID":
            raise ValueError("Replace YOUR_DEPLOYMENT_ID with your real Watson deployment ID.")
        apikey = os.getenv(args.watson_api_key_env)
        if not apikey:
            raise ValueError(
                f"Set environment variable {args.watson_api_key_env} with your IBM API key."
            )

        print("Scoring with Watson deployment...", flush=True)
        watson_cfg = WatsonConfig(
            apikey=apikey,
            wml_url=args.watson_url,
            deployment_id=args.watson_deployment_id,
            version=args.watson_version,
        )
        pred_df = score_with_watson(
            split.val,
            target_col=args.target_col,
            watson=watson_cfg,
            max_rows=args.watson_max_rows,
        )
        out_path = Path(args.watson_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(out_path, index=False)
        print(f"Watson predictions saved: {out_path}")

    if args.use_ngrok:
        if not args.ngrok_url:
            raise ValueError("--use-ngrok requires --ngrok-url")
        ngrok_cfg = NgrokConfig(
            base_url=args.ngrok_url,
            anomaly_threshold=args.ngrok_threshold,
        )
        print("Checking ngrok health...", flush=True)
        health = check_ngrok_health(ngrok_cfg)
        print(f"Ngrok health: {health}")

        print("Scoring with ngrok /predict...", flush=True)
        ngrok_pred_df = score_with_ngrok(
            split.val,
            target_col=args.target_col,
            ngrok=ngrok_cfg,
            window_size=args.ngrok_window_size,
            max_windows=args.ngrok_max_windows,
        )
        ngrok_out = Path(args.ngrok_output)
        ngrok_out.parent.mkdir(parents=True, exist_ok=True)
        ngrok_pred_df.to_csv(ngrok_out, index=False)
        print(f"Ngrok predictions saved: {ngrok_out}")

        parsed_df = parse_ngrok_predictions(ngrok_pred_df)
        parsed_out = Path(args.ngrok_parsed_output)
        parsed_out.parent.mkdir(parents=True, exist_ok=True)
        parsed_df.to_csv(parsed_out, index=False)
        print(f"Parsed ngrok predictions saved: {parsed_out}")

        plot_ngrok_anomalies(
            split.val,
            target_col=args.target_col,
            parsed_df=parsed_df,
            out_path=args.ngrok_plot_output,
        )
        print(f"Ngrok anomaly plot saved: {args.ngrok_plot_output}")


if __name__ == "__main__":
    main()
