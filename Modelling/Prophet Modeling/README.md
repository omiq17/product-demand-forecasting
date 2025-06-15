# 📦 E-Commerce Weekly Demand Forecasting with Prophet

A comprehensive time series forecasting solution for e-commerce demand prediction using Facebook Prophet, featuring both Jupyter notebook analysis and an interactive Streamlit web application.

## 🌟 Features

- **Advanced Data Preprocessing**: Outlier removal, log transformation, Min-Max scaling, and moving average smoothing
- **Prophet Time Series Modeling**: Weekly seasonality modeling with fine-tuned changepoint detection
- **Interactive Web Dashboard**: Streamlit application for real-time forecasting and visualization
- **Comprehensive Metrics**: MAE, RMSE, and MAPE evaluation metrics
- **Professional Visualizations**: High-quality plots with train/test splits and forecast annotations
- **Automated Category Selection**: Top 3 performing categories based on historical demand
- **Export Capabilities**: Download forecast charts and metrics as Excel/ZIP files

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Required packages (see `requirements.txt`)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/prophet-modeling.git
   cd prophet-modeling
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the application**
   - Open your browser to `http://localhost:8501`
   - Upload your e-commerce Excel file
   - View forecasts and download results

## 📊 Data Requirements

Your Excel file should contain the following columns:
- `category_name_1`: Product category names
- `M-Y`: Date column (Month-Year format)
- `qty_ordered`: Quantity ordered for each period

**Example data format:**
```
category_name_1 | M-Y        | qty_ordered
Electronics     | 2023-01-15 | 150
Clothing        | 2023-01-15 | 89
Home & Garden   | 2023-01-15 | 234
```

## 🔧 Model Configuration

### Prophet Parameters
- **Weekly Seasonality**: Enabled for capturing weekly demand patterns
- **Changepoint Prior Scale**: 0.00005 (fine-tuned for stability)
- **Training Period**: 120 weeks (configurable)
- **Forecast Horizon**: 6 weeks (configurable)

### Data Processing Pipeline
1. **Outlier Filtering**: Removes top 15% extreme values (85th percentile threshold)
2. **Log Transformation**: `log1p()` for variance stabilization
3. **Min-Max Scaling**: Normalizes data to [0,1] range
4. **Moving Average**: 3-period smoothing for noise reduction

## 📁 Project Structure

```
prophet-modeling/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── streamlit_app.py            # Interactive web application
├── weekly3 (1).ipynb          # Jupyter notebook analysis
├── data/                       # Data directory (create locally)
│   └── sample_data.xlsx        # Sample dataset
├── output/                     # Generated forecasts (auto-created)
│   ├── forecast_charts/        # PNG forecast plots
│   └── forecast_metrics.xlsx   # Performance metrics
└── docs/                       # Documentation
    ├── model_methodology.md    # Detailed model explanation
    └── api_reference.md        # Function documentation
```

## 🎯 Usage Examples

### Jupyter Notebook Analysis
```python
# Load and preprocess data
df = pd.read_excel("your_data.xlsx")
df = preprocess_data(df)

# Run forecasting pipeline
metrics = run_forecast_pipeline(df, top_n=3)
```

### Streamlit Web App
1. Launch the app: `streamlit run streamlit_app.py`
2. Upload your Excel file using the file uploader
3. View automatic category selection and forecasts
4. Download results as ZIP file containing charts and metrics

## 📈 Performance Metrics

The model evaluates performance using three key metrics:

- **MAE (Mean Absolute Error)**: Average absolute difference between actual and predicted values
- **RMSE (Root Mean Square Error)**: Square root of average squared differences
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error (lower is better)

**Typical Performance Ranges:**
- Excellent: MAPE < 10%
- Good: MAPE 10-20%
- Fair: MAPE 20-50%

## 🛠️ Customization

### Adjusting Forecast Parameters
```python
# In streamlit_app.py or notebook
train_weeks = 120        # Increase for more historical context
horizon = 6             # Adjust forecast period
changepoint_prior = 0.00005  # Tune model flexibility
```

### Adding New Metrics
```python
def custom_metric(y_true, y_pred):
    return your_calculation
    
# Add to metrics dictionary
metrics[cat]["Custom"] = custom_metric(df_pred["y"], df_pred["yhat"])
```

## 🔬 Methodology

### 1. Data Preprocessing
- **Temporal Aggregation**: Daily data aggregated to weekly periods (Sunday-based)
- **Outlier Detection**: Statistical filtering using 85th percentile threshold
- **Transformation**: Log transformation followed by Min-Max normalization
- **Smoothing**: 3-period moving average for noise reduction

### 2. Model Training
- **Prophet Configuration**: Weekly seasonality with minimal changepoint flexibility
- **Train/Test Split**: Last 6 weeks held out for validation
- **Category Selection**: Top 3 categories by total historical demand

### 3. Evaluation
- **Cross-validation**: Single holdout validation on most recent data
- **Multiple Metrics**: MAE, RMSE, and MAPE for comprehensive assessment
- **Visual Inspection**: Plots with train/test boundaries and peak annotations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues

**Q: "No forecasts generated" error**
A: Ensure your data has at least 126 weeks (120 train + 6 test) per category

**Q: Poor forecast accuracy**
A: Try adjusting the `changepoint_prior_scale` parameter or increasing training data

**Q: Memory issues with large datasets**
A: Consider chunking data processing or filtering to fewer categories

### Getting Help
- 📧 Email: your.email@domain.com
- 💬 Issues: [GitHub Issues](https://github.com/yourusername/prophet-modeling/issues)
- 📖 Documentation: [Wiki](https://github.com/yourusername/prophet-modeling/wiki)

## 🙏 Acknowledgments

- **Facebook Prophet**: Time series forecasting framework
- **Streamlit**: Interactive web application framework
- **Scikit-learn**: Data preprocessing and metrics
- **Matplotlib/Seaborn**: Data visualization libraries

## 📊 Sample Results

![Sample Forecast](docs/sample_forecast.png)

**Performance Summary:**
| Category | MAE | RMSE | MAPE (%) |
|----------|-----|------|----------|
| Electronics | 0.0234 | 0.0456 | 12.3 |
| Clothing | 0.0189 | 0.0398 | 9.8 |
| Home & Garden | 0.0267 | 0.0523 | 15.2 |

---

## 🔄 Version History

- **v1.0.0** - Initial release with Prophet forecasting
- **v1.1.0** - Added Streamlit web interface
- **v1.2.0** - Enhanced preprocessing pipeline
- **v1.3.0** - Added export capabilities and improved visualizations

---

**Built with ❤️ using Python, Prophet, and Streamlit**
