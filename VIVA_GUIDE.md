# Viva Guide – Retail Demand Forecasting

## 30-second pitch
"Retailers must know how much they'll sell tomorrow to stock correctly. I built a time-series model that forecasts
total daily sales of a store chain. I cleaned the data, explored it, tested stationarity, compared a baseline with
ARIMA, Holt-Winters and SARIMA, tuned SARIMA by grid search, validated on a time-based split, and deployed it as a web app."

## Pipeline (map each step to your lecture plan)
| Step | What I did | Lecture / CO |
|---|---|---|
| Data | Store-sales data: date, store, product family, sales, promotions, holiday (13,140 rows, 3 stores x 4 families x 3 years) | L5 |
| Cleaning | 120 missing sales -> time interpolation per series. Outliers -> IQR rule (Q1-1.5IQR, Q3+1.5IQR) and capped, not deleted (143 capped) | L6 |
| EDA | mean/median/variance/std, histogram, box plot, bar chart, heatmap, line plot. Finding: weekends sell more, mild upward trend, yearly wave | L7-9 |
| Features | month, is_weekend, lag_7, 7-day rolling mean | L10-11 |
| Time series | Decomposition (trend + seasonality + noise), ADF test: raw p=0.31 (non-stationary), after differencing p<0.001 (stationary) | L24 |
| Models | Seasonal naive baseline, ARIMA(2,1,2), Holt-Winters, SARIMA | L25-27 |
| Tuning | Grid search over p,q in 0-2 and P,Q in 0-1 (d=D=1, m=7), pick lowest AIC on training data -> SARIMA(0,1,2)x(1,1,1,7) | L35 |
| Validation | Last 60 days held out (no shuffling), RMSE / MAE / MAPE, plus 3-fold rolling-origin CV | L30-32 |
| Deployment | Final SARIMA refit on all data, stored as its fitted parameters in info.json (tiny file) and rebuilt in the app, Streamlit app with forecast slider, CI band, CSV download | - |

## Results (test set, 60 days)
| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| SARIMA(0,1,2)x(1,1,1,7) | 188.9 | 103.0 | 2.6% |
| Seasonal naive | 224.1 | 154.8 | 4.0% |
| Holt-Winters | 240.2 | 158.2 | 4.0% |
| ARIMA(2,1,2) | 373.9 | 326.2 | 8.5% |
CV RMSE mean = 158.4. Daily sales are about 3,500-5,000, so MAPE of 2.6% is good.

## Concepts you must be able to explain
- **ARIMA(p,d,q):** p = past values used (AR), d = times differenced to make it stationary, q = past errors used (MA).
- **SARIMA:** ARIMA + seasonal part (P,D,Q,m). m=7 because sales repeat weekly.
- **Why ARIMA did badly:** it has no seasonality, so its forecast flattens into a straight line. Seasonality is strong here.
- **Stationarity:** constant mean/variance over time. Needed for ARIMA. Checked with ADF test (null = non-stationary; p<0.05 rejects it).
- **Exponential smoothing (Holt-Winters):** weighted average giving more weight to recent data; has level, trend, seasonal parts.
- **Why a time-based split:** shuffling would let the model "see the future" (data leakage).
- **Baseline:** a model must beat a simple rule, otherwise it's not worth using.
- **RMSE vs MAE:** RMSE squares errors so punishes big misses; MAE is the average miss in sales units; MAPE is in %.
- **AIC:** model quality score that penalises complexity; lower is better. Used for grid search.
- **Overfitting:** train RMSE 117 vs test RMSE 189 -> some gap, but it still beats the baseline on unseen data, and CV RMSE (158) is stable across folds.
- **IQR outlier rule and why cap:** deleting rows would break the daily time order.
- **Poisson / Neg. Binomial / logistic (other syllabus units):** not used here because sales are continuous, not counts or classes.

## Likely questions + short answers
1. *Why SARIMA?* Data has a clear weekly seasonality and trend; it won on RMSE/MAE/MAPE.
2. *How did you choose the parameters?* Grid search by AIC on the training part only, then evaluated on held-out data.
3. *How do you know it isn't overfitted?* Held-out test + rolling CV, and it beats the baseline.
4. *What are the limitations?* Doesn't use promotions/holidays, assumes patterns continue, wider uncertainty for long horizons. Improvement: SARIMAX with exogenous variables or XGBoost with lag features.
5. *Ethical issues?* Over-forecasting causes waste; under-forecasting causes stock-outs; data privacy is low-risk here since no customer data.
6. *What does the confidence interval mean?* Range in which actual sales are expected to fall with 95% probability under model assumptions.
7. *How is it deployed?* Model stored as its fitted parameters in info.json (tiny file) and rebuilt in the app; Streamlit app loads it and calls get_forecast(); hosted on Streamlit Community Cloud from GitHub.

## Be ready to be honest about
The dataset I used has the same structure as Kaggle's Store Sales data but was simulated (weekly pattern, trend, yearly cycle, promotions, holidays, noise, with injected missing values and outliers). Say this plainly, and mention the pipeline runs on the real Kaggle file by just replacing data/store_sales.csv.
