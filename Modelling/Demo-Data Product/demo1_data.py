import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LSTM
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64


# --- Helper Functions ---

def smape(actual, forecast):
    """
    Calculates Symmetric Mean Absolute Percentage Error (SMAPE).
    """
    denominator = (np.abs(actual) + np.abs(forecast)) / 2
    mask = denominator != 0
    return np.mean(np.abs(actual[mask] - forecast[mask]) / denominator[mask]) * 100 if np.any(mask) else 0


def safe_mape(actual, forecast):
    """
    Calculates Mean Absolute Percentage Error (MAPE), handling zero values.
    """
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100 if np.any(mask) else 0


def load_and_prepare_data(uploaded_file):
    """
    Loads and preprocesses the data from the uploaded Excel file.
    """
    df = pd.read_excel(uploaded_file, usecols=['created_at', 'category_name_1', 'qty_ordered'])
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['week'] = df['created_at'].dt.to_period('W').apply(lambda r: r.start_time)

    weekly = df.groupby(['category_name_1', 'week'])['qty_ordered'].count().reset_index()
    pivot = weekly.pivot(index='week', columns='category_name_1', values='qty_ordered').fillna(0)
    pivot.index = pd.to_datetime(pivot.index)

    return pivot


def create_sequences(data, seq_length):
    """
    Creates sequences for time series forecasting.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


# --- Preprocessing Function for all models ---
def preprocess_for_model(data, category):
    """
    Applies log transform, outlier clipping, and train/test split for a given category.
    """
    cat_data = data[[category]].copy()
    pivot_log = np.log1p(cat_data)

    Q1 = pivot_log[category].quantile(0.25)
    Q3 = pivot_log[category].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    pivot_clean = pivot_log.clip(lower=lower, upper=upper)

    train_size = len(pivot_clean) - 6
    train_data = pivot_clean.iloc[:train_size]
    test_data = pivot_clean.iloc[train_size:]

    return train_data, test_data, pivot_clean


# --- Model Functions ---

@st.cache_data
def run_cnn_model(data, category):
    """
    Runs the CNN model for a given category.
    """
    train_data, test_data, _ = preprocess_for_model(data, category)

    scaler = RobustScaler()
    scaled_train = scaler.fit_transform(train_data)

    seq_len = 12
    X_train, y_train = create_sequences(scaled_train, seq_len)

    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_len, 1)),
        Flatten(),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)

    last_seq = scaled_train[-seq_len:]
    test_preds = []
    for _ in range(len(test_data)):
        pred = model.predict(last_seq.reshape(1, seq_len, 1), verbose=0)[0]
        test_preds.append(pred)
        last_seq = np.roll(last_seq, -1, axis=0)
        last_seq[-1] = pred

    test_preds_rescaled = scaler.inverse_transform(np.array(test_preds).reshape(-1, 1))
    test_preds_final = np.expm1(test_preds_rescaled).flatten()

    test_actual_final = np.expm1(test_data.values).flatten()
    mae = mean_absolute_error(test_actual_final, test_preds_final)
    rmse = np.sqrt(mean_squared_error(test_actual_final, test_preds_final))
    mape = safe_mape(test_actual_final, test_preds_final)
    smape_val = smape(test_actual_final, test_preds_final)

    metrics = {'Model': 'CNN', 'Category': category, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'SMAPE': smape_val}

    return metrics, test_preds_final


@st.cache_data
def run_prophet_model(data, category):
    """
    Runs the Prophet model for a given category.
    """
    train_data, test_data, _ = preprocess_for_model(data, category)

    prophet_train_data = train_data.reset_index().rename(columns={'week': 'ds', category: 'y'})

    model = Prophet()
    model.fit(prophet_train_data)

    future = model.make_future_dataframe(periods=6, freq='W')
    forecast = model.predict(future)

    y_test_pred_log = forecast.iloc[-6:]['yhat'].values
    y_test_pred = np.expm1(y_test_pred_log)

    y_test_actual = np.expm1(test_data.values).flatten()
    mae = mean_absolute_error(y_test_actual, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    mape = safe_mape(y_test_actual, y_test_pred)
    smape_val = smape(y_test_actual, y_test_pred)

    metrics = {'Model': 'Prophet', 'Category': category, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'SMAPE': smape_val}

    return metrics, y_test_pred


@st.cache_data
def run_lstm_model(data, category):
    """
    Runs the LSTM model for a given category.
    """
    train_data, test_data, _ = preprocess_for_model(data, category)

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_data)

    sequence_length = 18
    X_train, y_train = create_sequences(scaled_train, sequence_length)

    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], 1), activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=12, verbose=0)

    last_sequence = scaled_train[-sequence_length:]
    test_predictions = []
    for i in range(len(test_data)):
        next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)[0]
        test_predictions.append(next_pred)
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1] = next_pred

    test_predictions_rescaled = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
    test_predictions_final = np.expm1(test_predictions_rescaled).flatten()

    test_actual_final = np.expm1(test_data.values).flatten()
    mae = mean_absolute_error(test_actual_final, test_predictions_final)
    rmse = np.sqrt(mean_squared_error(test_actual_final, test_predictions_final))
    mape = safe_mape(test_actual_final, test_predictions_final)
    smape_val = smape(test_actual_final, test_predictions_final)

    metrics = {'Model': 'LSTM', 'Category': category, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'SMAPE': smape_val}

    return metrics, test_predictions_final


# --- Charting Function ---
def create_comparison_chart(category, train_data, test_data, forecasts):
    """
    Creates a single chart comparing different model forecasts for a category.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot training and actual test data
    ax.plot(train_data.index, train_data.values, label='Train Actual', color='blue')
    ax.plot(test_data.index, test_data.values, label='Test Actual', color='green')

    # Plot forecasts
    for model_name, forecast_values in forecasts.items():
        ax.plot(test_data.index, forecast_values, label=f'{model_name} Forecast', linestyle='--')

    ax.set_title(f'Forecast for {category}')
    ax.set_xlabel("Date (MM/YY)")
    ax.set_ylabel("Quantity Ordered")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    return fig


# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("Product Demand Forecasting Analysis")

st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload your data (Excel file)", type=["xlsx"])

if uploaded_file:
    try:
        data = load_and_prepare_data(uploaded_file)
        categories = data.columns.tolist()

        selected_models = st.sidebar.multiselect("Select Models", ["CNN", "Prophet", "LSTM"],
                                                 default=["CNN", "Prophet", "LSTM"])
        selected_categories = st.sidebar.multiselect("Select Categories", categories, default=categories)

        if st.sidebar.button("Start Analyzing"):
            if not selected_models or not selected_categories:
                st.warning("Please select at least one model and one category.")
            else:
                with st.spinner('Running analysis... This may take a moment.'):
                    all_metrics = []
                    all_charts = {}

                    for category_name in selected_categories:
                        forecasts = {}

                        # Get train/test data for charting
                        _, test_data_log, pivot_clean = preprocess_for_model(data, category_name)
                        train_data_raw = np.expm1(pivot_clean[[category_name]].iloc[:-6])
                        test_data_raw = np.expm1(test_data_log[[category_name]])

                        for model_name in selected_models:
                            if model_name == "CNN":
                                metrics, forecast_values = run_cnn_model(data, category_name)
                            elif model_name == "Prophet":
                                metrics, forecast_values = run_prophet_model(data, category_name)
                            elif model_name == "LSTM":
                                metrics, forecast_values = run_lstm_model(data, category_name)

                            all_metrics.append(metrics)
                            forecasts[model_name] = forecast_values

                        # Create comparison chart
                        fig = create_comparison_chart(category_name, train_data_raw, test_data_raw, forecasts)
                        all_charts[category_name] = fig

                    st.header("Forecasting Metrics")
                    metrics_df = pd.DataFrame(all_metrics).round(2)
                    st.dataframe(metrics_df)

                    # Download button for metrics table
                    csv = metrics_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Metrics as CSV",
                        data=csv,
                        file_name='forecasting_metrics.csv',
                        mime='text/csv',
                    )

                    st.header("Forecast Charts")
                    for category, fig in all_charts.items():
                        st.subheader(f"Category: {category}")
                        st.pyplot(fig)

                        # Download button for chart
                        buf = BytesIO()
                        fig.savefig(buf, format="png")
                        st.download_button(
                            label=f"Download Chart for {category}",
                            data=buf.getvalue(),
                            file_name=f"{category}_comparison_forecast.png",
                            mime="image/png"
                        )

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload an Excel file to begin.")
