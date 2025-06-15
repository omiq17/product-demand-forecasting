# Model Methodology

## Overview

This document provides a detailed explanation of the forecasting methodology used in the Prophet modeling project.

## Data Preprocessing Pipeline

### 1. Data Cleaning
- **Missing Value Handling**: Removes rows with missing values in critical columns (`category_name_1`, `M-Y`, `qty_ordered`)
- **Date Parsing**: Converts `M-Y` column to datetime format with error handling
- **Data Validation**: Ensures data integrity before processing

### 2. Temporal Aggregation
```python
df["Week"] = df["M-Y"].dt.to_period("W-SUN").dt.to_timestamp()
weekly = df.groupby(["Week", "category_name_1"])["qty_ordered"].sum().reset_index()
```
- Aggregates daily/monthly data to weekly periods
- Uses Sunday as the week start (Prophet default)
- Groups by category for separate forecasting

### 3. Outlier Filtering
```python
q85 = weekly["qty_ordered"].quantile(0.85)
weekly = weekly[weekly["qty_ordered"] < q85]
```
- **Rationale**: Removes extreme spikes that could distort model training
- **Threshold**: 85th percentile (removes top 15% of values)
- **Impact**: Improves model stability and reduces overfitting to anomalies

### 4. Data Transformation
```python
weekly["qty_ordered"] = np.log1p(weekly["qty_ordered"])
```
- **Log Transformation**: `log1p()` handles zero values and stabilizes variance
- **Purpose**: Makes data more suitable for linear modeling assumptions
- **Benefit**: Reduces impact of large values while preserving relationships

### 5. Normalization
```python
scaler = MinMaxScaler()
weekly["qty_ordered"] = scaler.fit_transform(weekly[["qty_ordered"]])
```
- **Min-Max Scaling**: Normalizes data to [0,1] range
- **Advantage**: Ensures consistent scale across different categories
- **Requirement**: Must inverse transform for actual quantity predictions

### 6. Smoothing
```python
weekly["qty_ordered"] = weekly["qty_ordered"].rolling(window=3, min_periods=1).mean()
```
- **Moving Average**: 3-period window for noise reduction
- **Parameters**: `min_periods=1` handles edge cases
- **Effect**: Reduces short-term fluctuations while preserving trends

## Prophet Model Configuration

### Core Parameters
```python
model = Prophet(
    weekly_seasonality=True,
    changepoint_prior_scale=0.00005
)
```

### Weekly Seasonality
- **Enabled**: Captures day-of-week patterns in demand
- **Fourier Terms**: Automatic selection by Prophet
- **Use Case**: E-commerce typically shows weekly patterns (weekend vs weekday)

### Changepoint Detection
- **Prior Scale**: 0.00005 (very low flexibility)
- **Rationale**: Prevents overfitting to random fluctuations
- **Trade-off**: May miss genuine trend changes but provides stable forecasts

### Training Configuration
- **Training Period**: 120 weeks (approximately 2.3 years)
- **Test Period**: 6 weeks (validation horizon)
- **Minimum Data**: 126 total weeks required per category

## Category Selection Strategy

### Top-N Selection
```python
top_categories = (
    weekly.groupby("category_name_1")["qty_ordered"]
    .sum().nlargest(3).index.tolist()
)
```
- **Criteria**: Total historical demand volume
- **Count**: Top 3 categories (configurable)
- **Rationale**: Focus on high-impact categories for business value

### Data Sufficiency Check
- **Minimum Requirement**: 126 weeks of data
- **Handling**: Categories with insufficient data are skipped
- **Alternative**: Could implement interpolation for sparse data

## Evaluation Metrics

### Mean Absolute Error (MAE)
```python
mae = mean_absolute_error(df_pred["y"], df_pred["yhat"])
```
- **Interpretation**: Average absolute difference in normalized units
- **Advantage**: Robust to outliers, easy to interpret
- **Scale**: Same units as the transformed data

### Root Mean Square Error (RMSE)
```python
rmse = np.sqrt(mean_squared_error(df_pred["y"], df_pred["yhat"]))
```
- **Interpretation**: Standard deviation of prediction errors
- **Advantage**: Penalizes large errors more heavily
- **Use**: Good for detecting model instability

### Mean Absolute Percentage Error (MAPE)
```python
mape = (np.abs(df_pred["y"] - df_pred["yhat"]) / df_pred["y"]).mean() * 100
```
- **Interpretation**: Average percentage error
- **Advantage**: Scale-independent, business-friendly
- **Limitation**: Undefined for zero actual values

## Visualization Strategy

### Plot Components
1. **Training Data**: Blue line with markers
2. **Actual Test Data**: Green line with markers
3. **Forecasted Values**: Red dashed line with X markers
4. **Train/Test Split**: Vertical black dotted line
5. **Peak Annotation**: Highlight highest forecast value

### Design Principles
- **Clarity**: Clear distinction between data types
- **Grid Lines**: Improve readability
- **Legends**: Proper labeling for interpretation
- **Annotations**: Call attention to key insights

## Model Limitations

### Data Requirements
- **Volume**: Needs substantial historical data (120+ weeks)
- **Quality**: Sensitive to data quality and outliers
- **Frequency**: Designed for weekly aggregation

### Seasonal Patterns
- **Weekly Only**: Doesn't capture monthly/quarterly patterns
- **Holiday Effects**: No explicit holiday modeling
- **External Factors**: Doesn't account for promotions, marketing, etc.

### Scaling Considerations
- **Category Independence**: Each category modeled separately
- **Memory Usage**: Linear scaling with number of categories
- **Computation Time**: Can be slow for many categories

## Future Improvements

### Model Enhancements
1. **Holiday Integration**: Add country-specific holiday effects
2. **External Regressors**: Include marketing spend, weather, etc.
3. **Hierarchical Forecasting**: Model category relationships
4. **Ensemble Methods**: Combine multiple forecasting approaches

### Technical Improvements
1. **Automated Hyperparameter Tuning**: Grid search for optimal parameters
2. **Cross-Validation**: Time series cross-validation for robust evaluation
3. **Confidence Intervals**: Include uncertainty quantification
4. **Real-time Updates**: Streaming forecast updates

### Business Integration
1. **Inventory Optimization**: Link forecasts to stock planning
2. **Alert System**: Automated notifications for forecast anomalies
3. **A/B Testing**: Framework for model comparison
4. **Business Rules**: Incorporate domain knowledge constraints
