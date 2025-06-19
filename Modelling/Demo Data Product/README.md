# Product Demand Forecasting Analysis

A comprehensive Streamlit web application for analyzing and forecasting product demand using advanced machine learning models. This tool helps businesses predict future demand patterns across different product categories using historical sales data.

## Features

- **Multiple Forecasting Models**: Compare results from three different forecasting approaches
  - **CNN (Convolutional Neural Networks)**: Deep learning model for pattern recognition in time series
  - **LSTM (Long Short-Term Memory)**: Specialized neural network for sequential data analysis
  - **Prophet**: Facebook's robust forecasting tool designed for business time series

- **Interactive Analysis**: 
  - Upload Excel files with sales data
  - Select specific product categories to analyze
  - Choose which models to run for comparison
  - Download results and visualizations

- **Comprehensive Metrics**: Evaluate model performance using multiple statistical measures
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Square Error)
  - MAPE (Mean Absolute Percentage Error)
  - SMAPE (Symmetric Mean Absolute Percentage Error)

- **Visual Insights**: Generate detailed comparison charts showing actual vs predicted values

## Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd product-demand-forecasting
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv forecasting_env
   source forecasting_env/bin/activate  # On Windows: forecasting_env\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the application**
   ```bash
   streamlit run demo1_data.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:8501`
   - The application will load with an intuitive sidebar interface

3. **Upload your data**
   - Click "Upload your data (Excel file)" in the sidebar
   - Ensure your Excel file contains these columns:
     - `created_at`: Date/timestamp of orders
     - `category_name_1`: Product category names
     - `qty_ordered`: Quantity ordered for each transaction

4. **Configure analysis**
   - Select which forecasting models to run
   - Choose specific product categories to analyze
   - Click "Start Analyzing" to begin the process

5. **View results**
   - Review performance metrics in the comparison table
   - Examine forecast visualizations for each category
   - Download results as CSV files or PNG images

## Data Requirements

Your Excel file should contain sales/order data with the following structure:

| created_at | category_name_1 | qty_ordered |
|------------|-----------------|-------------|
| 2023-01-01 | Electronics     | 5           |
| 2023-01-02 | Clothing        | 3           |
| 2023-01-03 | Electronics     | 8           |

- **created_at**: Date when the order was placed (any standard date format)
- **category_name_1**: Product category or classification
- **qty_ordered**: Number of items ordered (integer values)

## Model Details

### CNN (Convolutional Neural Network)
- Uses 1D convolutions to detect patterns in weekly sales data
- Sequence length: 12 weeks
- Preprocessing: Log transformation and robust scaling
- Best for: Detecting complex seasonal patterns

### LSTM (Long Short-Term Memory)
- Specialized for sequential data with long-term dependencies
- Sequence length: 18 weeks  
- Preprocessing: Log transformation and min-max scaling
- Best for: Capturing long-term trends and dependencies

### Prophet
- Facebook's forecasting tool designed for business metrics
- Handles seasonality, holidays, and trend changes automatically
- Robust to missing data and outliers
- Best for: Business forecasting with interpretable components

## Technical Notes

- **Data Processing**: Weekly aggregation with outlier detection and removal
- **Train/Test Split**: Last 6 weeks reserved for testing
- **Caching**: Model results are cached to improve performance
- **Error Handling**: Comprehensive error handling for data quality issues

## Output Files

The application generates downloadable files:
- **Metrics CSV**: Comparative performance metrics for all models
- **Forecast Charts**: PNG images of forecast visualizations
- **File naming**: Automatically named based on category and analysis type

## System Requirements

- Python 3.8 or higher
- Minimum 4GB RAM (8GB recommended for large datasets)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Troubleshooting

**Common Issues:**

1. **Installation errors**: Ensure you have the latest pip version
   ```bash
   pip install --upgrade pip
   ```

2. **Memory issues**: Reduce the number of categories analyzed simultaneously

3. **File upload problems**: Verify Excel file format and column names match requirements

4. **Model convergence**: Some categories with irregular patterns may show warnings

## Contributing

We welcome contributions to improve this forecasting tool. Please feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests
- Share feedback on model performance

## License

This project is available for educational and commercial use. Please refer to the license file for detailed terms.

---

**Built with ❤️ using Streamlit and modern machine learning techniques**
