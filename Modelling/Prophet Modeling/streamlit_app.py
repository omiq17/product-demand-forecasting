import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io
import zipfile

st.set_page_config(page_title="E-Commerce Weekly Forecast", layout="wide")
st.title("📦 Weekly Forecasting for E-Commerce Categories")

# === Upload Excel File ===
uploaded_file = st.file_uploader("Upload your e-commerce Excel file", type=["xlsx"])
if not uploaded_file:
    st.warning("Upload a file to proceed.")
    st.stop()

# === Load Dataset ===
df = pd.read_excel(uploaded_file)
df = df.dropna(subset=["category_name_1", "M-Y", "qty_ordered"])
df["M-Y"] = pd.to_datetime(df["M-Y"], errors="coerce")
df = df.dropna(subset=["M-Y"])

# === Weekly Aggregation ===
df["Week"] = df["M-Y"].dt.to_period("W-SUN").dt.to_timestamp()
weekly = df.groupby(["Week", "category_name_1"])["qty_ordered"].sum().reset_index()

# === Outlier Filtering (Top 15%) ===
q85 = weekly["qty_ordered"].quantile(0.85)
weekly = weekly[weekly["qty_ordered"] < q85]

# === Log + Min-Max Scaling + Smoothing ===
weekly["qty_ordered"] = np.log1p(weekly["qty_ordered"])
scaler = MinMaxScaler()
weekly["qty_ordered"] = scaler.fit_transform(weekly[["qty_ordered"]])
weekly["qty_ordered"] = weekly["qty_ordered"].rolling(window=3, min_periods=1).mean()

# === Top 3 Categories ===
top_categories = weekly.groupby("category_name_1")["qty_ordered"].sum().nlargest(3).index.tolist()
st.success(f"Top 3 categories for forecasting: {', '.join(top_categories)}")

train_weeks = 120
horizon = 6

metrics = {}
forecast_data = {}
chart_images = {}

# === Forecast for Each Category ===
for cat in top_categories:
    sub = weekly[weekly["category_name_1"] == cat].sort_values("Week").reset_index(drop=True)
    sub.rename(columns={"Week": "ds", "qty_ordered": "y"}, inplace=True)

    train_df = sub.iloc[:-horizon].copy()
    test_df = sub.iloc[-horizon:].copy()

    model = Prophet(weekly_seasonality=True, changepoint_prior_scale=0.00005)
    model.fit(train_df)

    forecast = model.predict(test_df[["ds"]])
    df_pred = forecast[["ds", "yhat"]].merge(test_df, on="ds")

    mae = mean_absolute_error(df_pred["y"], df_pred["yhat"])
    rmse = np.sqrt(mean_squared_error(df_pred["y"], df_pred["yhat"]))
    mape = (np.abs(df_pred["y"] - df_pred["yhat"]) / df_pred["y"]).mean() * 100

    metrics[cat] = {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}
    forecast_data[cat] = (train_df, test_df, df_pred)

    # === Plot Forecast ===
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_df["ds"], train_df["y"], "-o", label="Train", color="blue")
    ax.plot(test_df["ds"], test_df["y"], "-o", label="Actual", color="green")
    ax.plot(df_pred["ds"], df_pred["yhat"], "--x", label="Forecast", color="red")
    ax.axvline(train_df["ds"].max(), color="black", linestyle=":", label="Train/Test Split")
    ax.set_title(f"{cat} Forecast")
    ax.set_xlabel("Week")
    ax.set_ylabel("Demand (Normalized)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format="png", bbox_inches="tight")
    chart_images[f"{cat}_forecast.png"] = img_bytes.getvalue()

    forecast_data[cat] += (fig,)

# === Category Filter with "All" Option ===
options = ["All"] + top_categories
selected_cat = st.selectbox("🔎 Select Category to View Forecast", options)

if selected_cat == "All":
    for cat in top_categories:
        train_df, test_df, df_pred, fig = forecast_data[cat]
        st.subheader(f"📈 Forecast for: {cat}")
        st.pyplot(fig)

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{metrics[cat]['MAE']:.4f}")
        col2.metric("RMSE", f"{metrics[cat]['RMSE']:.4f}")
        col3.metric("MAPE (%)", f"{metrics[cat]['MAPE (%)']:.2f}")
        st.markdown("---")  # Divider between charts

else:
    train_df, test_df, df_pred, fig = forecast_data[selected_cat]
    st.subheader(f"📈 Forecast for: {selected_cat}")
    st.pyplot(fig)

    st.markdown("### 📋 Forecast Accuracy")
    st.metric("MAE", f"{metrics[selected_cat]['MAE']:.4f}")
    st.metric("RMSE", f"{metrics[selected_cat]['RMSE']:.4f}")
    st.metric("MAPE (%)", f"{metrics[selected_cat]['MAPE (%)']:.2f}")


# === All Summary Table ===
st.markdown("### 📊 All Categories Summary")
dfm = pd.DataFrame.from_dict(metrics, orient="index").reset_index().rename(columns={"index": "Category"})

styled_df = dfm.style.format({
    "MAE": "{:.4f}",
    "RMSE": "{:.4f}",
    "MAPE (%)": "{:.2f}"
})

st.dataframe(styled_df, use_container_width=True)

best_row = dfm.loc[dfm["MAPE (%)"].idxmin()]
st.success(f"✅ Best Model: `{best_row['Category']}` with MAPE = `{best_row['MAPE (%)']:.2f}%`")

# === Download ZIP of Charts ===
zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
    for filename, content in chart_images.items():
        zf.writestr(filename, content)

st.download_button("📥 Download All Forecast Charts (ZIP)", zip_buffer.getvalue(), file_name="forecast_charts.zip")
