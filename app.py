import json
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd, plotly.graph_objects as go, streamlit as st

st.set_page_config(page_title="Retail Demand Forecasting", page_icon="📈", layout="wide")
st.title("📈 Retail Demand Forecasting")
st.caption("CSE3141 Predictive Analytics mini project · SARIMA model on daily store sales")

@st.cache_data
def load():
    return pd.read_csv("data/daily_clean.csv", parse_dates=["date"], index_col="date").asfreq("D")
info = json.load(open("models/info.json"))
daily = load()

@st.cache_resource
def load_model():
    # rebuild SARIMA with the saved parameters (no re-training, takes milliseconds)
    m = SARIMAX(daily["sales"], order=tuple(info["sarima_order"]), seasonal_order=tuple(info["seasonal_order"]),
                enforce_stationarity=False, enforce_invertibility=False)
    return m.smooth(np.array(info["params"]))

model = load_model()
metrics = pd.read_csv("outputs/metrics.csv", index_col=0)

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Forecast", "📊 EDA", "🏆 Model comparison", "ℹ️ About"])

with tab1:
    c1, c2 = st.columns([1, 3])
    with c1:
        horizon = st.slider("Days to forecast", 7, 90, 30)
        history = st.slider("Days of history to show", 30, 365, 120)
        conf = st.selectbox("Confidence interval", [0.80, 0.90, 0.95], index=2)
    fc = model.get_forecast(horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=1 - conf)
    mean.index = pd.date_range(daily.index[-1] + pd.Timedelta(days=1), periods=horizon)
    ci.index = mean.index
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=daily.index[-history:], y=daily["sales"].iloc[-history:], name="History", line=dict(color="gray"))
        fig.add_scatter(x=mean.index, y=mean, name="Forecast", line=dict(color="crimson", width=3))
        fig.add_scatter(x=ci.index, y=ci.iloc[:, 1], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=ci.index, y=ci.iloc[:, 0], fill="tonexty", fillcolor="rgba(220,20,60,0.15)",
                        line=dict(width=0), name=f"{int(conf*100)}% interval")
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Total daily sales")
        st.plotly_chart(fig, width="stretch")
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg forecast / day", f"{mean.mean():,.0f}")
    m2.metric("Peak day", mean.idxmax().strftime("%a %d %b"), f"{mean.max():,.0f}")
    m3.metric("Lowest day", mean.idxmin().strftime("%a %d %b"), f"{mean.min():,.0f}")
    out = pd.DataFrame({"forecast": mean.round(0), "lower": ci.iloc[:, 0].round(0), "upper": ci.iloc[:, 1].round(0)})
    st.dataframe(out, width="stretch")
    st.download_button("⬇️ Download forecast CSV", out.to_csv().encode(), "forecast.csv", "text/csv")

with tab2:
    st.subheader("Exploratory data analysis")
    a, b = st.columns(2)
    a.image("outputs/eda_distribution.png", caption="Distribution and box plot")
    b.image("outputs/eda_dayofweek.png", caption="Weekend effect")
    a.image("outputs/eda_heatmap.png", caption="Correlation heatmap")
    b.image("outputs/decomposition.png", caption="Trend / seasonality / noise decomposition")

with tab3:
    st.subheader("Test-set results (last 60 days)")
    st.dataframe(metrics, width="stretch")
    st.image("outputs/forecast_comparison.png")
    st.write(f"Time-series cross-validation mean RMSE: **{info['cv_rmse_mean']:.1f}**")

with tab4:
    st.markdown(f"""
**Pipeline:** data → cleaning (interpolation, IQR capping) → EDA → feature engineering → stationarity test (ADF)
→ ARIMA / SARIMA / Holt-Winters vs seasonal-naive baseline → grid search → time-based validation → deployment.

**Final model:** SARIMA{tuple(info['sarima_order'])}×{tuple(info['seasonal_order'])}, refit on all data.

**Limitations:** forecasts assume the past weekly pattern continues; promotions and holidays are not
used as inputs (SARIMAX with exogenous variables is the next step); uncertainty grows with the horizon.
""")
