
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === File paths ===
data_path = "C:/Users/User/Downloads/PRINCIPLES OF DATA SCIENCE/Datasets/e-commerce-scrubbed-data-filtered.xlsx"
output_dir = "C:/Users/User/Downloads/PRINCIPLES OF DATA SCIENCE/Datasets/Final-output"
os.makedirs(output_dir, exist_ok=True)

# === Load & preprocess ===
df = pd.read_excel(data_path)
df = df.dropna(subset=["category_name_1", "M-Y", "qty_ordered"])
df["M-Y"] = pd.to_datetime(df["M-Y"], errors="coerce")
df = df.dropna(subset=["M-Y"])

# === Weekly Aggregation ===
df["Week"] = df["M-Y"].dt.to_period("W-SUN").dt.to_timestamp()
weekly = df.groupby(["Week", "category_name_1"])["qty_ordered"].sum().reset_index()

# === Outlier Filtering (Top 15%) ===
q85 = weekly["qty_ordered"].quantile(0.85)
weekly = weekly[weekly["qty_ordered"] < q85]

# === Apply Log Scaling (no normalization) ===
weekly["qty_ordered"] = np.log1p(weekly["qty_ordered"])

# === Moving Average Smoothing ===
weekly["qty_ordered"] = weekly["qty_ordered"].rolling(window=3, min_periods=1).mean()

# === Top 3 categories ===
top_categories = (
    weekly.groupby("category_name_1")["qty_ordered"].sum().nlargest(3).index.tolist()
)
print(f"Initially selected top 3 categories: {top_categories}")

# === Forecast Parameters ===
train_weeks = 120
horizon = 6

metrics = []

# === Forecast Loop ===
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

    metrics.append({"Category": cat, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape})

    # === Plotting ===
    plt.figure(figsize=(12, 6))
    plt.plot(train_df["ds"], train_df["y"], "-o", label="Train Data", color="blue")
    plt.plot(test_df["ds"], test_df["y"], "-o", label="Actual Demand", color="green")
    plt.plot(df_pred["ds"], df_pred["yhat"], "--x", label="Forecasted Demand", color="red")

    plt.axvline(train_df["ds"].max(), color="black", linestyle=":", label="Train-Test Split")
    plt.xlabel("Week")
    plt.ylabel("Weekly Demand ")
    plt.title(f"{cat}: Weekly Forecast ")
    plt.grid(True, linestyle="--", alpha=0.6)

    # === Annotate Peak Forecast ===
    peak_week = df_pred.loc[df_pred["yhat"].idxmax()]
    plt.annotate("Highest Forecasted Demand",
                 xy=(peak_week["ds"], peak_week["yhat"]),
                 xytext=(peak_week["ds"], peak_week["yhat"] * 1.1),
                 arrowprops=dict(facecolor='red', shrink=0.05))

    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"{cat}_forecast.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"[{cat}] forecast done → {fig_path}")

# === Save Summary Metrics ===
if metrics:
    dfm = pd.DataFrame(metrics)
    excel_path = os.path.join(output_dir, "forecast_metrics.xlsx")
    dfm.to_excel(excel_path, index=False)

    print("\n=== SUMMARY ===")
    print(dfm.to_string(index=False))
    print(f"\nMetrics saved to: {excel_path}")
else:
    print("No forecasts generated.")
