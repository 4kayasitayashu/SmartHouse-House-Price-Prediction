
# 🏡 SmartHouse — House Price Prediction & Property Intelligence System

A machine learning capstone project that predicts residential sale prices using property characteristics and provides interactive insights into housing data through a Streamlit dashboard.

## 📌 Project Overview

SmartHouse is an end-to-end Data Science project built using Python and machine learning. It explores historical house sales, prepares data for modeling, compares regression algorithms, predicts house prices, and visualizes the factors associated with model predictions.

The project uses the **House Prices: Advanced Regression Techniques** dataset from Kaggle.

### Key Features

- Exploratory Data Analysis (EDA)
- Data quality and missing-value analysis
- Numerical feature imputation and categorical encoding
- Leakage-conscious training and validation workflow
- Regression model comparison
- House price prediction through an interactive web dashboard
- Market insights by overall quality, kitchen quality, and neighborhood
- Feature importance visualization
- Downloadable prediction results in CSV format

## 🎯 Project Objectives

1. Understand the characteristics of residential property sales.
2. Investigate relationships between property features and sale prices.
3. Build and compare machine learning regression models.
4. Evaluate model performance using validation data.
5. Develop an interactive application for property price estimation.
6. Explain model behavior using feature importance.

## 🗂️ Project Structure

```text
House-Price-Prediction/
│
├── assets/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── data_description.txt
│   │   └── sample_submission.csv
│   │
│   └── processed/
│       ├── X_train.csv
│       ├── X_valid.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_valid.csv
│
├── models/
│   ├── house_price_model.joblib
│   └── model_info.json
│
├── reports/
│   ├── data_quality_report.csv
│   ├── data_quality_summary.txt
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── saleprice_distribution.png
│   ├── correlation_heatmap.png
│   └── price_group_comparisons.png
│
├── src/
│   ├── data/
│   │   ├── inspect_data.py
│   │   ├── quality_report.py
│   │   └── prepare_data.py
│   │
│   ├── features/
│   │
│   ├── models/
│   │   ├── train_model.py
│   │   └── feature_importance.py
│   │
│   └── utils/
│
├── tests/
├── README.md
└── requirements.txt
```

*Note: The structure above reflects the project's intended layout. Some directories may be empty, and generated files appear after their corresponding scripts have been run.*

## 📊 Dataset

**Source:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

The dataset contains historical residential property sales from Ames, Iowa.

| Dataset | Rows | Columns | Purpose |
|---|---:|---:|---|
| Training data | 1,460 | 81 | Model development and evaluation |
| Test data | 1,459 | 80 | Unlabeled properties supplied with the competition |

The training data includes `SalePrice`, the target variable. The competition test data does not include sale prices.

### Important Features

| Feature | Description |
|---|---|
| `OverallQual` | Overall material and finish quality |
| `GrLivArea` | Above-ground living area |
| `TotalBsmtSF` | Total basement area |
| `GarageCars` | Garage capacity in cars |
| `GarageArea` | Garage area |
| `YearBuilt` | Original construction year |
| `LotArea` | Lot size |
| `Neighborhood` | Physical location within Ames |
| `KitchenQual` | Kitchen quality |
| `SalePrice` | Actual sale price; prediction target |

## 🔎 Exploratory Data Analysis

The exploratory analysis examines sale prices, missing values, correlations, and differences between property groups.

Key observations from the training data:

- The target variable, `SalePrice`, is right-skewed, with a measured skewness of approximately **1.883**.
- The average sale price is approximately **$180,921**.
- The median sale price is approximately **$163,000**.
- Sale prices range from approximately **$34,900 to $755,000**.
- `OverallQual` and `GrLivArea` have strong positive correlations with `SalePrice` in this dataset.
- Median sale prices vary across neighborhoods and kitchen-quality categories.
- Missing values require careful handling because some missing entries represent the absence of a property feature rather than unknown information.

These are descriptive observations from this historical dataset, not conclusions about every housing market.

### Generated Visualizations

- `reports/saleprice_distribution.png` — distribution of sale prices
- `reports/correlation_heatmap.png` — feature correlation heatmap
- `reports/price_group_comparisons.png` — comparisons across selected property groups

## ⚙️ Data Preprocessing

The project uses a preprocessing pipeline to prepare data for machine learning.

- Numerical missing values are imputed using the median.
- Categorical missing values are handled using a dedicated missing-value category.
- Categorical variables are one-hot encoded.
- Unseen categories are handled by the encoder.
- Training and validation data are split before fitting the preprocessing workflow used for model evaluation.
- The selected model and preprocessing steps are saved together in a reusable pipeline.

This design helps keep preprocessing consistent between training and prediction and reduces the risk of validation data leakage.

## 🤖 Model Development

Three regression algorithms were compared during model development:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

The models were evaluated using a held-out validation set.

### Validation Results

The following are the recorded results from the initial model comparison. Values are rounded to two decimal places.

| Model | RMSE | MAE |
|---|---:|---:|
| Linear Regression | $29,389.18 | $18,665.09 |
| Random Forest | $28,898.29 | $17,497.73 |
| Gradient Boosting | $29,288.27 | $17,174.60 |

**Metric definitions:**

- **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger errors more heavily. Lower is better.
- **MAE (Mean Absolute Error):** Measures the average absolute difference between predicted and actual sale prices. Lower is better.

The current training workflow selects a model using validation RMSE and saves the fitted pipeline to `models/house_price_model.joblib`.

The table records the earlier comparison results. If the training script or dataset changes, rerun training and consult `reports/model_comparison.csv` for the latest results.

## 🧠 Feature Importance

The trained Random Forest model's feature importance output identifies which features contribute most to its predictions according to the model's importance measure.

The leading features in the current output are:

| Feature | Importance |
|---|---:|
| Overall Quality | 0.5863 |
| Above-ground Living Area | 0.1104 |
| Total Basement Area | 0.0387 |
| Second-floor Area | 0.0352 |
| Basement Finished Area 1 | 0.0296 |
| First-floor Area | 0.0218 |
| Garage Capacity | 0.0207 |

The values above come from `reports/feature_importance.csv` and may change if the model is retrained.

**Interpretation note:** Feature importance describes how the fitted model uses information. It does not prove causation, establish the direction of a feature's effect, or guarantee that changing a feature will change a real property's market value by a specific amount.

## 🖥️ Interactive Dashboard

SmartHouse includes a Streamlit application with four tabs.

### 1. Price Prediction

- Enter selected property characteristics.
- Generate a predicted sale price.
- View estimated price per square foot of above-ground living area.
- Download the entered features and prediction as a CSV file.

### 2. Market Insights

- Explore average and median sale prices.
- View the sale-price distribution.
- Compare median prices by overall quality and kitchen quality.
- Explore neighborhood-level median sale prices.

### 3. Model Evaluation

- Review the selected model and validation metrics.
- Compare regression model results.
- Visualize the leading features by model importance.

### 4. About

- Review the project objectives, workflow, dataset, and limitations.

### Prediction Limitation

The current dashboard exposes a selected subset of the model's input features. Other features retain values from a baseline row in the training dataset. Consequently, predictions are useful for demonstrating the model and comparing scenarios, but they are not fully customized property valuations.

## 🚀 Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the installed libraries
- pip
- Visual Studio Code or another Python development environment

### 1. Clone the Repository

If the project has been uploaded to GitHub:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd House-Price-Prediction
```

Alternatively, open your existing project folder directly.

### 2. Create and Activate a Virtual Environment

**Windows Command Prompt:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

If `requirements.txt` is present and complete:

```bash
pip install -r requirements.txt
```

Otherwise, install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn streamlit joblib
```

### 4. Prepare the Dataset

Download the competition dataset from Kaggle and extract the files into:

```text
data/raw/
```

Ensure the following files are present:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/data_description.txt
data/raw/sample_submission.csv
```

### 5. Run the Data Inspection Scripts

```bash
python src/data/inspect_data.py
python src/data/quality_report.py
python src/data/prepare_data.py
```

### 6. Train the Model

```bash
python src/models/train_model.py
```

This script compares the candidate models, selects a model according to validation RMSE, and saves the fitted pipeline and model metadata.

### 7. Generate Feature Importance

```bash
python src/models/feature_importance.py
```

The results are saved to:

```text
reports/feature_importance.csv
```

### 8. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Open the local URL printed in the terminal by Streamlit.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Exploratory data visualizations |
| Scikit-learn | Preprocessing, regression, and evaluation |
| Joblib | Saving and loading the trained pipeline |
| Streamlit | Interactive web dashboard |
| Git and GitHub | Version control and project sharing |

## 📁 Generated Outputs

| File | Purpose |
|---|---|
| `reports/data_quality_report.csv` | Feature-level missing-value report |
| `reports/data_quality_summary.txt` | Summary of data-quality checks |
| `reports/model_comparison.csv` | Model evaluation results |
| `reports/feature_importance.csv` | Feature importance values |
| `models/house_price_model.joblib` | Saved model pipeline |
| `models/model_info.json` | Model metadata and validation metrics |
| `data/processed/` | Prepared training, validation, and test data |

## ⚠️ Limitations and Future Improvements

### Current Limitations

- The dataset contains historical sales from one location and period.
- The application does not use live property listings or current market feeds.
- The prediction form does not expose every model feature.
- Validation results may not reflect performance on future properties.
- Feature importance does not establish causality.
- The model does not replace a professional property appraisal.

### Future Improvements

- Add more property inputs to the prediction form.
- Investigate log-transforming the skewed sale-price target.
- Add cross-validation and hyperparameter tuning.
- Evaluate residuals and prediction errors across price ranges.
- Add explainability tools such as permutation importance or SHAP.
- Add confidence or prediction intervals, with appropriate calibration.
- Build a more complete automated test suite.
- Deploy the dashboard to a hosting service.
- Add a reproducible dependency file and continuous integration checks.

## 👨‍💻 Project Purpose

SmartHouse was developed as a Data Science capstone to demonstrate an end-to-end workflow involving data inspection, exploratory analysis, preprocessing, regression modeling, evaluation, feature interpretation, and application development.

## 📜 Dataset Attribution

The data is provided through the Kaggle competition:

[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

Please consult Kaggle's competition rules and dataset terms before redistributing data or using the project for other purposes.