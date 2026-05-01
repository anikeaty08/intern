# 📈 XGBoost Trade Win Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit)
![Model](https://img.shields.io/badge/Model-XGBoost-success)
![Status](https://img.shields.io/badge/Status-Working-brightgreen)

An interactive Streamlit app to predict whether a trade is likely to be a **Win (1)** or **Loss (0)** using a trained XGBoost classifier.

---

## ✨ Project Structure

```text
xxx/
├── app.py                          # Streamlit frontend
├── requirements.txt                # Python dependencies
├── xgboost_trade_win_model_v2.pkl  # Trained model
├── model_feature_columns.pkl       # Saved feature column schema
├── trader_sentiment_.ipynb         # Training + evaluation notebook
├── input/
│   ├── historical_data.csv         # Raw historical trade data
│   └── fear_greed_index.csv        # Sentiment index data
└── cleaner_data/                   # Reserved for cleaned/processed outputs
```

---

## 🧠 Model Details

- **Model Type:** `XGBClassifier` (XGBoost)
- **Target:** Trade outcome (`Win=1`, `Loss=0`)
- **Input Features:** 247 engineered features
  - Numeric trade features (`Execution Price`, `Size Tokens`, `Size USD`, `Start Position`, `Fee`, `value`)
  - One-hot encoded categorical features:
    - `Coin_*`
    - `Direction_*`
    - `Side_*`
    - `classification_*` (sentiment)

---

## 📊 Performance (from notebook)

Metrics extracted from `trader_sentiment_.ipynb`:

### Validation Set
- **Accuracy:** `0.9105`
- **Precision:** `0.8216`
- **Recall:** `0.9999`
- **F1 Score:** `0.9020`
- **ROC AUC:** `0.9460`

### Final Test Set (Threshold = 0.70)
- **Accuracy:** `0.8625`
- **Precision:** `0.7333`
- **Recall:** `0.9910`
- **F1 Score:** `0.8429`
- **ROC AUC:** `0.8822`

---

## 🚀 How to Run the App

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

---

## 🎯 What the Frontend Does

- Loads `xgboost_trade_win_model_v2.pkl`
- Accepts user-friendly numeric and categorical inputs
- Auto-encodes inputs into the exact model feature format
- Shows predicted **Win Probability**
- Displays final prediction signal in a clear UI

---

## 🛠️ Tech Stack

- Python
- Pandas / NumPy
- XGBoost
- Streamlit

