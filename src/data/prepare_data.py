
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

X = train.drop(columns=["SalePrice", "Id"])
y = train["SalePrice"]
X_test = test.drop(columns=["Id"])

# Identify numerical and categorical columns
numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

# Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols),
])

# Split raw training data before fitting preprocessing to avoid leakage
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_processed = preprocessor.fit_transform(X_train)
X_valid_processed = preprocessor.transform(X_valid)
X_test_processed = preprocessor.transform(X_test)

# Save processed datasets
feature_names = preprocessor.get_feature_names_out()

pd.DataFrame(X_train_processed, columns=feature_names).to_csv(
    OUTPUT_DIR / "X_train.csv", index=False
)
pd.DataFrame(X_valid_processed, columns=feature_names).to_csv(
    OUTPUT_DIR / "X_valid.csv", index=False
)
pd.DataFrame(X_test_processed, columns=feature_names).to_csv(
    OUTPUT_DIR / "X_test.csv", index=False
)

y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_valid.to_csv(OUTPUT_DIR / "y_valid.csv", index=False)

print("PREPROCESSING COMPLETE")
print(f"Training rows: {X_train_processed.shape[0]}")
print(f"Validation rows: {X_valid_processed.shape[0]}")
print(f"Test rows: {X_test_processed.shape[0]}")
print(f"Processed features: {X_train_processed.shape[1]}")
print(f"Saved files to: {OUTPUT_DIR.resolve()}")