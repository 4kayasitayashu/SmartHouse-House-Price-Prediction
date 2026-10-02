
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

ROOT = Path(".")
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Load original training data
df = pd.read_csv("data/raw/train.csv")
X = df.drop(columns=["SalePrice", "Id"])
y = df["SalePrice"]

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_cols),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical_cols),
])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05,
        l2_regularization=1.0, random_state=42
    ),
}

results = []
best_rmse = float("inf")
best_name = None
best_pipeline = None

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_valid)
    rmse = root_mean_squared_error(y_valid, predictions)
    mae = mean_absolute_error(y_valid, predictions)

    results.append({"Model": name, "RMSE": rmse, "MAE": mae})
    print(f"{name}: RMSE=${rmse:,.2f} | MAE=${mae:,.2f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_name = name
        best_pipeline = pipeline

pd.DataFrame(results).sort_values("RMSE").to_csv(
    REPORT_DIR / "model_comparison.csv", index=False
)

# Refit the selected model on all labeled data for final deployment
best_pipeline.fit(X, y)
joblib.dump(best_pipeline, MODEL_DIR / "house_price_model.joblib")

with open(MODEL_DIR / "model_info.json", "w", encoding="utf-8") as f:
    json.dump({
        "model": best_name,
        "validation_rmse": round(best_rmse, 2),
        "training_rows": len(X),
    }, f, indent=2)

print(f"\nSelected model by validation RMSE: {best_name}")
print(f"Saved model: {MODEL_DIR / 'house_price_model.joblib'}")