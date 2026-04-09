# Personal Medical Expense ML Pipeline

A comprehensive Machine Learning pipeline built with Streamlit for analyzing and predicting personal medical expenses.

## Features

- **Problem Type Selection**: Choose between Classification or Regression
- **Data Input**: Upload CSV files, select target features, visualize data shape with PCA
- **Exploratory Data Analysis (EDA)**: Comprehensive data analysis with visualizations
- **Data Engineering**: Handle missing values, detect and remove outliers using IQR, Isolation Forest, DBSCAN, and LOF
- **Feature Selection**: Variance threshold, correlation analysis, and information gain
- **Data Split**: Configurable train-test split
- **Model Selection**: Linear Regression, SVM, Random Forest, Logistic Regression, K-Means
- **Training & Validation**: K-Fold cross-validation
- **Performance Metrics**: Comprehensive metrics with overfitting/underfitting detection
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV support

## Installation

```bash
pip install streamlit pandas numpy plotly scikit-learn scipy
```

## Usage

```bash
streamlit run pipeline.py
```

## Dataset

Use medical expense datasets from:
- UCI Machine Learning Repository
- Kaggle Medical Cost Personal Datasets

## Screenshots

The application provides an intuitive step-by-step interface with horizontal navigation through all pipeline stages.

## License

MIT License