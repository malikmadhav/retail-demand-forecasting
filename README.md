# Retail Demand Forecasting (CSE3141 Predictive Analytics – Mini Project)

Forecasts total daily retail sales with ARIMA, Holt-Winters and SARIMA, compared against a seasonal-naive baseline.
Deployed as a Streamlit web app.

## Run locally
```
pip install -r requirements.txt
python make_data.py     # (optional) regenerate the dataset
python train.py         # cleans data, EDA, trains + validates models, saves model
streamlit run app.py
```

## Deploy (free) on Streamlit Community Cloud
1. Push this folder to a public GitHub repo (keep `data/`, `models/`, `outputs/` folders).
2. Go to share.streamlit.io -> New app -> choose repo, branch `main`, main file `app.py` -> Deploy.

## Using the real Kaggle data
Download "Store Sales - Time Series Forecasting" (train.csv), rename columns `store_nbr, family, sales, onpromotion, date`
and save as `data/store_sales.csv` (add a `holiday` column of zeros if missing), then run `python train.py`.

## Files
make_data.py · train.py · app.py · data/ · models/ · outputs/ · VIVA_GUIDE.md
