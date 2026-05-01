from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "xgboost_trade_win_model_v2.pkl"
NUMERIC_FEATURES = [
    "Execution Price",
    "Size Tokens",
    "Size USD",
    "Start Position",
    "Fee",
    "value",
]


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def feature_groups(feature_names: list[str]) -> dict[str, list[str]]:
    return {
        "coins": sorted([f for f in feature_names if f.startswith("Coin_")]),
        "directions": sorted([f for f in feature_names if f.startswith("Direction_")]),
        "side": sorted([f for f in feature_names if f.startswith("Side_")]),
        "sentiment": sorted(
            [f for f in feature_names if f.startswith("classification_")]
        ),
    }


def build_input_row(
    feature_names: list[str],
    numeric_inputs: dict[str, float],
    coin: str,
    direction: str,
    side: str,
    sentiment: str,
) -> pd.DataFrame:
    row = {feature: 0.0 for feature in feature_names}

    for key, value in numeric_inputs.items():
        if key in row:
            row[key] = float(value)

    for selected in [coin, direction, side, sentiment]:
        if selected in row:
            row[selected] = 1.0

    return pd.DataFrame([row], columns=feature_names)


def main():
    st.set_page_config(page_title="Trade Win Predictor", page_icon="📈", layout="wide")
    st.title("📈 XGBoost Trade Win Predictor")
    st.caption(
        "Interactive frontend for `xgboost_trade_win_model_v2.pkl` with auto feature encoding."
    )

    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()

    try:
        model = load_model()
    except Exception as exc:
        st.exception(exc)
        st.stop()

    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        booster = model.get_booster()
        feature_names = list(booster.feature_names or [])

    if not feature_names:
        st.error("Could not read feature names from model. Please re-export model.")
        st.stop()

    groups = feature_groups(feature_names)

    with st.expander("Model details", expanded=False):
        st.write(
            {
                "model_type": type(model).__name__,
                "total_features": len(feature_names),
                "numeric_features": len(NUMERIC_FEATURES),
                "coin_options": len(groups["coins"]),
                "direction_options": len(groups["directions"]),
                "side_options": len(groups["side"]),
                "sentiment_options": len(groups["sentiment"]),
            }
        )

    left, right = st.columns([1.5, 1], gap="large")

    with left:
        st.subheader("Trade Inputs")

        c1, c2 = st.columns(2)
        with c1:
            execution_price = st.number_input(
                "Execution Price",
                min_value=0.0,
                value=100.0,
                step=0.01,
                format="%.6f",
            )
            size_tokens = st.number_input(
                "Size Tokens",
                min_value=0.0,
                value=1.0,
                step=0.01,
                format="%.6f",
            )
            size_usd = st.number_input(
                "Size USD",
                min_value=0.0,
                value=100.0,
                step=1.0,
                format="%.6f",
            )
        with c2:
            start_position = st.number_input(
                "Start Position",
                value=0.0,
                step=1.0,
                format="%.6f",
            )
            fee = st.number_input(
                "Fee",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.6f",
            )
            value = st.number_input(
                "value",
                value=0.0,
                step=1.0,
                format="%.6f",
            )

        st.subheader("Categorical Inputs")

        coin_choice = st.selectbox("Coin", options=groups["coins"], index=0)
        direction_choice = st.selectbox(
            "Direction", options=groups["directions"], index=0
        )
        side_choice = st.selectbox("Side", options=groups["side"], index=0)
        sentiment_choice = st.selectbox(
            "Market Sentiment (classification)",
            options=groups["sentiment"],
            index=0,
        )

    numeric_inputs = {
        "Execution Price": execution_price,
        "Size Tokens": size_tokens,
        "Size USD": size_usd,
        "Start Position": start_position,
        "Fee": fee,
        "value": value,
    }

    input_df = build_input_row(
        feature_names=feature_names,
        numeric_inputs=numeric_inputs,
        coin=coin_choice,
        direction=direction_choice,
        side=side_choice,
        sentiment=sentiment_choice,
    )

    with right:
        st.subheader("Prediction")
        if st.button("Predict Win Probability", type="primary", use_container_width=True):
            proba = float(model.predict_proba(input_df)[0][1])
            label = "Win likely ✅" if proba >= 0.5 else "Loss risk ⚠️"

            st.metric("Win Probability", f"{proba * 100:.2f}%")
            st.progress(proba)
            st.markdown(f"**Model signal:** {label}")

        st.caption(
            "Tip: this app automatically one-hot encodes coin, direction, side, and sentiment."
        )

    with st.expander("Encoded model input (debug)", expanded=False):
        st.dataframe(input_df.T.rename(columns={0: "value"}), use_container_width=True)


if __name__ == "__main__":
    main()
