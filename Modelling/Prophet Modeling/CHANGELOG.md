# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive README with project overview and usage instructions
- Professional .gitignore for Python data science projects
- MIT License for open source distribution
- Contributing guidelines for community participation
- Detailed model methodology documentation

### Changed
- Improved project structure for professional development

## [1.3.0] - 2025-06-15

### Added
- Enhanced data preprocessing pipeline with outlier filtering
- Log transformation and Min-Max scaling for improved model stability
- Moving average smoothing for noise reduction
- Peak annotation in forecast visualizations
- Professional chart styling with grid lines and improved readability

### Changed
- Upgraded from SMAPE to MAPE for better interpretability
- Fine-tuned Prophet changepoint prior scale to 0.00005
- Extended training period to 120 weeks for better historical context
- Improved forecast accuracy through advanced preprocessing

### Fixed
- Date parsing errors in data loading
- Outlier handling for extreme demand spikes
- Chart export functionality in Streamlit app

## [1.2.0] - 2025-06-01

### Added
- Interactive Streamlit web application
- Real-time forecast visualization
- Category selection dropdown with "All" option
- Download functionality for forecast charts as ZIP
- Comprehensive metrics display (MAE, RMSE, MAPE)
- Automatic top 3 category selection

### Changed
- Migrated from standalone notebook to web application
- Improved user interface with professional styling
- Enhanced error handling and user feedback

## [1.1.0] - 2025-05-15

### Added
- Facebook Prophet integration for time series forecasting
- Weekly seasonality modeling
- Train/test split validation
- Multiple evaluation metrics (MAE, RMSE, SMAPE)
- Automated forecast chart generation
- Excel export for metrics and results

### Changed
- Switched from simple statistical methods to Prophet modeling
- Improved forecasting accuracy with advanced time series techniques

## [1.0.0] - 2025-05-01

### Added
- Initial project setup with Jupyter notebook
- Basic data loading and preprocessing
- Weekly demand aggregation
- Simple visualization capabilities
- Category-based analysis

### Features
- Data cleaning and validation
- Basic outlier detection
- Historical demand analysis
- Simple trend visualization

---

## Version Numbering

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality in a backward compatible manner
- **PATCH**: Backward compatible bug fixes

## Types of Changes

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Now removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes
