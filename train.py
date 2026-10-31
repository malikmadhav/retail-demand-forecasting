"""Retail Demand Forecasting - full pipeline (follows CSE3141 lecture plan).
Run:  python train.py
Outputs: outputs/*.png, outputs/metrics.csv, models/info.json, data/daily_clean.csv"""
import warnings, itertools, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ---------- 1. DATA COLLECTION ----------
df = pd.read_csv("data/store_sales.csv", parse_dates=["date"])
print("Raw shape:", df.shape); print(df.isna().sum())

# ---------- 2. PREPROCESSING ----------
# 2a. missing values: fill within each store+family series by time interpolation
df = df.sort_values(["store_nbr", "family", "date"])
df["sales"] = df.groupby(["store_nbr", "family"])["sales"].transform(lambda s: s.interpolate(limit_direction="both"))
# 2b. outliers: IQR rule per series, cap (winsorise) instead of deleting rows
def cap_iqr(s):
    q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
    return s.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
n_out = 0
def cap_and_count(s):
    global n_out
    c = cap_iqr(s); n_out += int((c != s).sum()); return c
df["sales"] = df.groupby(["store_nbr", "family"])["sales"].transform(cap_and_count)
print("Outliers capped by IQR:", n_out)

# ---------- 3. EDA ----------
print(df["sales"].describe())
print("Variance:", df["sales"].var(), " Std:", df["sales"].std())
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["sales"], bins=40, ax=ax[0], kde=True); ax[0].set_title("Distribution of sales")
sns.boxplot(data=df, x="family", y="sales", ax=ax[1]); ax[1].set_title("Sales by product family")
plt.tight_layout(); plt.savefig("outputs/eda_distribution.png", dpi=120); plt.close()

daily = df.groupby("date").agg(sales=("sales", "sum"), onpromotion=("onpromotion", "sum"),
                               holiday=("holiday", "max")).asfreq("D")
daily["dow"] = daily.index.dayofweek
plt.figure(figsize=(12, 4)); daily["sales"].plot(); plt.title("Total daily sales")
plt.tight_layout(); plt.savefig("outputs/eda_timeseries.png", dpi=120); plt.close()

plt.figure(figsize=(6, 4))
daily.groupby("dow")["sales"].mean().plot(kind="bar"); plt.title("Average sales by day of week (0=Mon)")
plt.tight_layout(); plt.savefig("outputs/eda_dayofweek.png", dpi=120); plt.close()

plt.figure(figsize=(6, 5)); sns.heatmap(daily[["sales", "onpromotion", "holiday", "dow"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation heatmap"); plt.tight_layout(); plt.savefig("outputs/eda_heatmap.png", dpi=120); plt.close()

# ---------- 4. FEATURE ENGINEERING ----------
daily["month"] = daily.index.month
daily["is_weekend"] = (daily["dow"] >= 5).astype(int)
daily["lag_7"] = daily["sales"].shift(7)
daily["roll_mean_7"] = daily["sales"].rolling(7).mean()
daily.to_csv("data/daily_clean.csv")

# ---------- 5. TIME SERIES ANALYSIS ----------
y = daily["sales"]
dec = seasonal_decompose(y, model="additive", period=7)
fig = dec.plot(); fig.set_size_inches(12, 8); plt.tight_layout(); plt.savefig("outputs/decomposition.png", dpi=120); plt.close()
def adf(series, name):
    stat, p, *_ = adfuller(series.dropna())
    print(f"ADF test [{name}]: statistic={stat:.3f}, p-value={p:.4f} ->", "stationary" if p < 0.05 else "NON-stationary")
    return p
p_raw = adf(y, "raw"); p_diff = adf(y.diff(), "1st difference")

# ---------- 6. TRAIN / TEST SPLIT (time-based, no shuffle) ----------
H = 60                                   # forecast horizon = last 60 days
train, test = y.iloc[:-H], y.iloc[-H:]
def metrics(actual, pred):
    return dict(RMSE=np.sqrt(mean_squared_error(actual, pred)), MAE=mean_absolute_error(actual, pred),
                MAPE=np.mean(np.abs((actual - pred) / actual)) * 100)
results, preds = {}, {}

# baseline: seasonal naive (same weekday last week)
naive = pd.Series([train.iloc[-7 + (i % 7)] for i in range(H)], index=test.index)
results["Seasonal Naive (baseline)"] = metrics(test, naive); preds["Seasonal Naive (baseline)"] = naive

# ARIMA
arima = ARIMA(train, order=(2, 1, 2)).fit()
preds["ARIMA(2,1,2)"] = arima.forecast(H); results["ARIMA(2,1,2)"] = metrics(test, preds["ARIMA(2,1,2)"])

# Holt-Winters (exponential smoothing, additive trend + seasonality, period 7)
hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7).fit()
preds["Holt-Winters"] = hw.forecast(H); results["Holt-Winters"] = metrics(test, preds["Holt-Winters"])

# SARIMA with GRID SEARCH on (p,d,q)(P,D,Q,7), chosen by AIC on training data only
best_aic, best_cfg = np.inf, None
for p_, q_, P_, Q_ in itertools.product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1]):
    try:
        m = SARIMAX(train, order=(p_, 1, q_), seasonal_order=(P_, 1, Q_, 7),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        if m.aic < best_aic: best_aic, best_cfg = m.aic, (p_, 1, q_, P_, 1, Q_)
    except Exception: pass
p_, d_, q_, P_, D_, Q_ = best_cfg
print("Best SARIMA via grid search:", (p_, d_, q_), (P_, D_, Q_, 7), "AIC=", round(best_aic, 1))
sarima = SARIMAX(train, order=(p_, d_, q_), seasonal_order=(P_, D_, Q_, 7),
                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
name = f"SARIMA{(p_, d_, q_)}x{(P_, D_, Q_, 7)}"
preds[name] = sarima.forecast(H); results[name] = metrics(test, preds[name])

# ---------- 7. EVALUATION ----------
res = pd.DataFrame(results).T.round(2).sort_values("RMSE")
print(res); res.to_csv("outputs/metrics.csv")
plt.figure(figsize=(12, 5)); plt.plot(train.iloc[-120:], label="Train (last 120d)", color="gray")
plt.plot(test, label="Actual", color="black", lw=2)
for k, v in preds.items(): plt.plot(v, label=k, alpha=.8)
plt.legend(); plt.title("Forecast vs actual (test period)"); plt.tight_layout()
plt.savefig("outputs/forecast_comparison.png", dpi=120); plt.close()

# Overfitting check: in-sample vs out-of-sample error for SARIMA
in_rmse = np.sqrt(mean_squared_error(train.iloc[14:], sarima.fittedvalues.iloc[14:]))
print(f"SARIMA train RMSE={in_rmse:.2f} vs test RMSE={results[name]['RMSE']:.2f}")

# Rolling-origin cross-validation (time-series CV) for the winning config: 3 folds of 30 days
cv = []
for k in range(3):
    cut = len(y) - H - 30 * (2 - k) - 30
    tr, te = y.iloc[:cut], y.iloc[cut:cut + 30]
    m = SARIMAX(tr, order=(p_, d_, q_), seasonal_order=(P_, D_, Q_, 7),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    cv.append(np.sqrt(mean_squared_error(te, m.forecast(30))))
print("Time-series CV RMSE per fold:", np.round(cv, 2), "mean =", round(np.mean(cv), 2))

# ---------- 8. FINAL MODEL: refit best on ALL data and save ----------
best_name = res.index[0]
print("Best model on test set:", best_name)
final = SARIMAX(y, order=(p_, d_, q_), seasonal_order=(P_, D_, Q_, 7),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
pass  # model is stored as its learned parameters (see info.json) - tiny file, rebuilt in app.py
json.dump({"best_by_test": best_name, "sarima_order": [p_, d_, q_], "seasonal_order": [P_, D_, Q_, 7],
           "params": [float(v) for v in final.params.values], "cv_rmse_mean": float(np.mean(cv)), "train_rmse": float(in_rmse)}, open("models/info.json", "w"), indent=2)
print("Saved models/info.json (contains fitted SARIMA parameters)")
