from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from Time_Series import (
    NgrokConfig,
    check_ngrok_health,
    clean_data,
    evaluate_naive_baseline,
    make_features,
    parse_ngrok_predictions,
    plot_ngrok_anomalies,
    score_with_ngrok,
    train_val_split,
)


def read_uploaded_file(uploaded_file, sheet_name: str | int = 0) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file, sheet_name=sheet_name)
    raise ValueError("Unsupported file type. Please upload .csv, .xlsx, or .xls")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.set_page_config(page_title="Time-Series Anomaly Demo", layout="wide")
st.title("Time-Series Anomaly Demo")
st.caption("Upload data, run preprocessing, evaluate baseline, and score via ngrok API.")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if not uploaded_file:
    st.info("Upload a file to begin.")
    st.stop()

sheet_name_input = st.text_input("Excel sheet name or index", value="0")
sheet_name: str | int
sheet_name = int(sheet_name_input) if sheet_name_input.isdigit() else sheet_name_input

try:
    raw_df = read_uploaded_file(uploaded_file, sheet_name=sheet_name)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

if raw_df.empty:
    st.error("Uploaded file has no rows.")
    st.stop()

st.subheader("Preview")
st.dataframe(raw_df.head(20), use_container_width=True)

cols = list(raw_df.columns)
col_a, col_b, col_c = st.columns(3)
with col_a:
    timestamp_col = st.selectbox("Timestamp column", cols, index=0)
with col_b:
    target_col = st.selectbox(
        "Target column",
        cols,
        index=1 if len(cols) > 1 else 0,
    )
with col_c:
    split_date = st.text_input("Split date (YYYY-MM-DD)", value="2018-01-01")

col_d, col_e = st.columns(2)
with col_d:
    date_format = st.text_input(
        "Optional date format (e.g. %d-%b-%y)",
        value="",
    ).strip()
with col_e:
    dayfirst = st.checkbox("Day-first date parsing", value=False)
debug_date_parsing = st.checkbox("Date parsing debug (prints diagnostics to terminal)", value=False)

use_ngrok = st.checkbox("Use ngrok scoring", value=True)
ngrok_url = st.text_input(
    "Ngrok URL",
    value="https://dreamier-evelia-milkier.ngrok-free.dev",
    disabled=not use_ngrok,
)
col_f, col_g = st.columns(2)
with col_f:
    ngrok_threshold = st.number_input(
        "Anomaly threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        disabled=not use_ngrok,
    )
with col_g:
    ngrok_window_size = st.number_input(
        "Window size",
        min_value=5,
        max_value=500,
        value=60,
        step=1,
        disabled=not use_ngrok,
    )

ngrok_max_windows = st.number_input(
    "Max windows to score",
    min_value=1,
    max_value=5000,
    value=5,
    step=1,
    disabled=not use_ngrok,
)

if st.button("Run Analysis", type="primary"):
    try:
        work_df = clean_data(
            raw_df,
            timestamp_col=timestamp_col,
            date_format=(date_format or None),
            dayfirst=dayfirst,
            debug_date_parsing=debug_date_parsing,
        )
        feat_df = make_features(work_df, target_col=target_col)
        split = train_val_split(feat_df, split_date=split_date)
        metrics = evaluate_naive_baseline(split.train, split.val, target_col=target_col)
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

    st.subheader("Baseline Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train rows", f"{len(split.train):,}")
    c2.metric("Validation rows", f"{len(split.val):,}")
    c3.metric("MAE", f"{metrics.mae:,.4f}")
    c4.metric("RMSE", f"{metrics.rmse:,.4f}")

    if use_ngrok:
        try:
            ngrok_cfg = NgrokConfig(
                base_url=ngrok_url,
                anomaly_threshold=float(ngrok_threshold),
            )
            health = check_ngrok_health(ngrok_cfg)
            raw_pred_df = score_with_ngrok(
                split.val,
                target_col=target_col,
                ngrok=ngrok_cfg,
                window_size=int(ngrok_window_size),
                max_windows=int(ngrok_max_windows),
            )
            parsed_pred_df = parse_ngrok_predictions(raw_pred_df)
        except Exception as e:
            st.error(f"Ngrok scoring failed: {e}")
            st.stop()

        st.subheader("Ngrok Health")
        st.json(health)

        st.subheader("Parsed Predictions")
        st.dataframe(parsed_pred_df, use_container_width=True)
        anomaly_count = (
            parsed_pred_df["is_anomaly"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
            .sum()
        )
        st.metric("Predicted anomalies", int(anomaly_count))

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plot_path = tmp.name
        plot_ngrok_anomalies(
            split.val,
            target_col=target_col,
            parsed_df=parsed_pred_df,
            out_path=plot_path,
        )
        plot_bytes = Path(plot_path).read_bytes()
        st.subheader("Anomaly Plot")
        st.image(plot_bytes, use_container_width=True)

        st.download_button(
            "Download raw ngrok predictions CSV",
            data=to_csv_bytes(raw_pred_df),
            file_name="ngrok_predictions.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download parsed ngrok predictions CSV",
            data=to_csv_bytes(parsed_pred_df),
            file_name="ngrok_predictions_parsed.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download anomaly plot PNG",
            data=plot_bytes,
            file_name="ngrok_anomalies.png",
            mime="image/png",
        )
