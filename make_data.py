"""Generates a Kaggle 'Store Sales'-style dataset (synthetic, reproducible).
Columns: date, store_nbr, family, sales, onpromotion, holiday
Real data can replace this file: same column names -> rest of project works unchanged."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
dates = pd.date_range("2021-01-01", "2023-12-31", freq="D")
stores = {1: 1.3, 2: 1.0, 3: 0.7}
families = {"GROCERY": 400, "BEVERAGES": 220, "CLEANING": 120, "DAIRY": 160}
holidays = pd.to_datetime(["2021-01-01","2021-12-25","2022-01-01","2022-12-25","2023-01-01","2023-12-25",
                           "2021-11-04","2022-10-24","2023-11-12"])  # incl. Diwali dates
rows = []
t = np.arange(len(dates))
for s, sf in stores.items():
    for f, base in families.items():
        trend = 1 + 0.0004 * t
        weekly = 1 + 0.18 * (pd.Series(dates.dayofweek).isin([5, 6]).values) - 0.05 * (dates.dayofweek == 1)
        yearly = 1 + 0.10 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365.25)
        promo = rng.poisson(3, len(dates)) * (rng.random(len(dates)) < 0.35)
        hol = dates.isin(holidays).astype(int)
        sales = base * sf * trend * weekly * yearly * (1 + 0.02 * promo) * (1 + 0.25 * hol)
        sales = sales * rng.normal(1, 0.06, len(dates))
        df = pd.DataFrame({"date": dates, "store_nbr": s, "family": f,
                           "sales": sales.round(2), "onpromotion": promo, "holiday": hol})
        rows.append(df)
df = pd.concat(rows, ignore_index=True)
# Real-world messiness so preprocessing is meaningful
idx = rng.choice(len(df), 120, replace=False); df.loc[idx, "sales"] = np.nan
idx = rng.choice(len(df), 25, replace=False);  df.loc[idx, "sales"] *= 4   # outliers
df.to_csv("data/store_sales.csv", index=False)
print(df.shape); print(df.head())
