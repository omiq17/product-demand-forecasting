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
import hashlib
import time


# --- Helper Functions ---

def get_file_hash(uploaded_file):
    """Generate a hash for the uploaded file to enable caching."""
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()

def initialize_session_state():
    """Initialize session state variables."""
    if 'data_hash' not in st.session_state:
        st.session_state.data_hash = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'categories' not in st.session_state:
        st.session_state.categories = []
    if 'cached_results' not in st.session_state:
        st.session_state.cached_results = {}
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

def clear_cache():
    """Clear all cached data and results."""
    st.session_state.cached_results = {}
    st.session_state.analysis_complete = False
    st.cache_data.clear()
    st.rerun()

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


@st.cache_data(persist=True)
def load_and_prepare_data(file_hash, file_content):
    """
    Loads and preprocesses the data from the uploaded Excel file.
    Uses file hash for better caching.
    """
    df = pd.read_excel(BytesIO(file_content), usecols=['created_at', 'category_name_1', 'qty_ordered'])
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

@st.cache_data(persist=True, show_spinner=False)
def run_cnn_model(data_hash, data, category):
    """
    Runs the CNN model for a given category.
    """
    train_data, test_data, _ = preprocess_for_model(data, category)

    scaler = RobustScaler()
    scaled_train = scaler.fit_transform(train_data)

    seq_len = 12
    X_train, y_train = create_sequences(scaled_train, seq_len)
    
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(seq_len, 1)),
        Flatten(),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)  # Reduced epochs for speed

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


@st.cache_data(persist=True, show_spinner=False)
def run_prophet_model(data_hash, data, category):
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


@st.cache_data(persist=True, show_spinner=False)
def run_lstm_model(data_hash, data, category):
    """
    Runs the LSTM model for a given category.
    """
    train_data, test_data, _ = preprocess_for_model(data, category)

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_data)

    sequence_length = 18
    X_train, y_train = create_sequences(scaled_train, sequence_length)
    
    model = Sequential([
        LSTM(32, input_shape=(X_train.shape[1], 1), activation='relu'),  # Reduced units for speed
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=12, verbose=0)  # Reduced epochs for speed

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
@st.cache_data(persist=True, show_spinner=False)
def create_comparison_chart(category, train_data, test_data, forecasts):
    """
    Creates a single chart comparing different model forecasts for a category.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot training and actual test data
    ax.plot(train_data.index, train_data.values, label='Train Actual', color='blue', linewidth=2)
    ax.plot(test_data.index, test_data.values, label='Test Actual', color='green', linewidth=2)

    # Plot forecasts with different styles
    colors = ['red', 'orange', 'purple', 'brown']
    linestyles = ['--', '-.', ':', '--']
    
    for i, (model_name, forecast_values) in enumerate(forecasts.items()):
        ax.plot(test_data.index, forecast_values, 
               label=f'{model_name} Forecast', 
               linestyle=linestyles[i % len(linestyles)],
               color=colors[i % len(colors)],
               linewidth=2)

    ax.set_title(f'Forecast for {category}', fontsize=14, fontweight='bold')
    ax.set_xlabel("Date (MM/YY)", fontsize=12)
    ax.set_ylabel("Quantity Ordered", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("Product Demand Forecasting Analysis")

# Initialize session state
initialize_session_state()

st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload your data (Excel file)", type=["xlsx"])

if uploaded_file:
    try:
        # Generate hash for caching
        file_hash = get_file_hash(uploaded_file)
        
        # Check if data needs to be reprocessed
        if st.session_state.data_hash != file_hash:
            with st.spinner('Loading and processing data...'):
                st.session_state.processed_data = load_and_prepare_data(file_hash, uploaded_file.getvalue())
                st.session_state.categories = st.session_state.processed_data.columns.tolist()
                st.session_state.data_hash = file_hash
                st.session_state.cached_results = {}  # Clear previous results
                st.session_state.analysis_complete = False
        
        data = st.session_state.processed_data
        categories = st.session_state.categories

        # Model and category selection
        selected_models = st.sidebar.multiselect("Select Models", ["CNN", "Prophet", "LSTM"],
                                                 default=["CNN", "Prophet", "LSTM"])
        selected_categories = st.sidebar.multiselect("Select Categories", categories, default=categories[:3])  # Limit default selection
          # Performance settings
        st.sidebar.subheader("Performance Settings")
        auto_run = st.sidebar.checkbox("Auto-run analysis when selections change", value=False)
        max_categories = st.sidebar.slider("Max categories to process", 1, len(categories), min(5, len(categories)))
        
        # Cache management
        if st.sidebar.button("🗑️ Clear Cache", help="Clear all cached results and start fresh"):
            clear_cache()
        
        # Show cache status
        cache_size = len(st.session_state.cached_results)
        if cache_size > 0:
            st.sidebar.success(f"📊 {cache_size} results cached")
        
        # Limit selections for performance
        if len(selected_categories) > max_categories:
            selected_categories = selected_categories[:max_categories]
            st.sidebar.warning(f"Limited to {max_categories} categories for performance.")

        # Show data preview
        if not st.session_state.analysis_complete:
            st.subheader("Data Preview")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Categories", len(categories))
                st.metric("Data Points per Category", len(data))
            with col2:
                st.metric("Selected Models", len(selected_models))
                st.metric("Selected Categories", len(selected_categories))
        
        # Auto-run logic
        should_run = auto_run and selected_models and selected_categories
        manual_run = st.sidebar.button("Start Analyzing", type="primary")
        
        if should_run or manual_run:
            if not selected_models or not selected_categories:
                st.warning("Please select at least one model and one category.")
            else:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_tasks = len(selected_categories) * len(selected_models)
                current_task = 0
                
                all_metrics = []
                all_charts = {}
                
                # Process each category
                for cat_idx, category_name in enumerate(selected_categories):
                    status_text.text(f"Processing category: {category_name}")
                    forecasts = {}

                    # Get train/test data for charting
                    _, test_data_log, pivot_clean = preprocess_for_model(data, category_name)
                    train_data_raw = np.expm1(pivot_clean[[category_name]].iloc[:-6])
                    test_data_raw = np.expm1(test_data_log[[category_name]])

                    # Process each model
                    for model_idx, model_name in enumerate(selected_models):
                        current_task += 1
                        status_text.text(f"Running {model_name} model for {category_name} ({current_task}/{total_tasks})")
                        
                        # Check cache first
                        cache_key = f"{file_hash}_{model_name}_{category_name}"
                        
                        if cache_key in st.session_state.cached_results:
                            metrics, forecast_values = st.session_state.cached_results[cache_key]
                        else:
                            # Run model
                            start_time = time.time()
                            if model_name == "CNN":
                                metrics, forecast_values = run_cnn_model(file_hash, data, category_name)
                            elif model_name == "Prophet":
                                metrics, forecast_values = run_prophet_model(file_hash, data, category_name)
                            elif model_name == "LSTM":
                                metrics, forecast_values = run_lstm_model(file_hash, data, category_name)
                            
                            # Cache results
                            st.session_state.cached_results[cache_key] = (metrics, forecast_values)
                            
                            # Show timing info
                            elapsed_time = time.time() - start_time
                            status_text.text(f"Completed {model_name} for {category_name} in {elapsed_time:.1f}s")

                        all_metrics.append(metrics)
                        forecasts[model_name] = forecast_values

                    # Create comparison chart
                    fig = create_comparison_chart(category_name, train_data_raw, test_data_raw, forecasts)
                    all_charts[category_name] = fig

                    # Update progress bar after each category
                    progress = (cat_idx + 1) / len(selected_categories)
                    progress_bar.progress(progress)

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                st.session_state.analysis_complete = True

                # Display results
                st.header("🔍 Forecasting Metrics")
                metrics_df = pd.DataFrame(all_metrics).round(2)
                
                # Enhanced metrics display
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.dataframe(metrics_df, use_container_width=True)
                with col2:
                    best_model = metrics_df.loc[metrics_df.groupby('Category')['MAE'].idxmin()]
                    st.subheader("Best Models by MAE")
                    for _, row in best_model.iterrows():
                        st.metric(f"{row['Category']}", f"{row['Model']}", f"MAE: {row['MAE']:.2f}")

                # Download button for metrics table
                csv = metrics_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Download Metrics as CSV",
                    data=csv,
                    file_name='forecasting_metrics.csv',
                    mime='text/csv',
                )

                st.header("📈 Forecast Charts")
                
                # Chart display options
                chart_cols = st.columns(2)
                with chart_cols[0]:
                    chart_layout = st.selectbox("Chart Layout", ["Single Column", "Two Columns"], index=1)
                with chart_cols[1]:
                    show_all = st.checkbox("Show all charts", value=True)
                
                if chart_layout == "Two Columns":
                    chart_columns = st.columns(2)
                    for idx, (category, fig) in enumerate(all_charts.items()):
                        with chart_columns[idx % 2]:
                            st.subheader(f"📊 {category}")
                            st.pyplot(fig, use_container_width=True)
                            
                            # Download button for chart
                            buf = BytesIO()
                            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                            st.download_button(                                label=f"💾 Download Chart",
                                data=buf.getvalue(),
                                file_name=f"{category}_comparison_forecast.png",
                                mime="image/png",
                                key=f"download_{category}"
                            )
                else:
                    for category, fig in all_charts.items():
                        st.subheader(f"📊 Category: {category}")
                        st.pyplot(fig, use_container_width=True)

                        # Download button for chart
                        buf = BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        st.download_button(
                            label=f"💾 Download Chart for {category}",
                            data=buf.getvalue(),
                            file_name=f"{category}_comparison_forecast.png",
                            mime="image/png",
                            key=f"download_{category}"
                        )
                
                # Summary statistics
                st.header("📋 Analysis Summary")
                summary_cols = st.columns(4)
                with summary_cols[0]:
                    st.metric("Categories Analyzed", len(selected_categories))
                with summary_cols[1]:
                    st.metric("Models Used", len(selected_models))
                with summary_cols[2]:
                    avg_smape = metrics_df['SMAPE'].mean()
                    st.metric("Average SMAPE", f"{avg_smape:.2f}%")
                with summary_cols[3]:
                    best_overall = metrics_df.loc[metrics_df['MAE'].idxmin(), 'Model']
                    st.metric("Best Overall Model", best_overall)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)  # Show detailed error for debugging

else:
    st.info("👆 Please upload an Excel file to begin the analysis.")
    st.markdown("""
    ### Expected Data Format:
    - **created_at**: Date column
    - **category_name_1**: Product category
    - **qty_ordered**: Quantity ordered
    
    ### Features:
    - 🔄 **Auto-run**: Enable automatic analysis when selections change
    - 💾 **Caching**: Results are cached for faster re-runs
    - 📊 **Multiple Models**: CNN, Prophet, and LSTM forecasting
    - 🎯 **Performance Metrics**: MAE, RMSE, MAPE, SMAPE
    """)
