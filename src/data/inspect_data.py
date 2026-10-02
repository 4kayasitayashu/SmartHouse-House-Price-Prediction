
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
DESCRIPTION_PATH = RAW_DATA_DIR / "data_description.txt"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "sample_submission.csv"


def main():
    # Verify required files before loading anything
    required_files = [
        TRAIN_PATH,
        TEST_PATH,
        DESCRIPTION_PATH,
        SAMPLE_SUBMISSION_PATH,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    print("=" * 60)
    print("SMARTHOUSE - DATA UNDERSTANDING")
    print("=" * 60)

    # Load the datasets
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    # Basic dataset information
    print("\n1. DATASET SHAPES")
    print("-" * 40)
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    print(f"Sample submission: {submission_df.shape}")

    # Inspect target availability
    print("\n2. TARGET VARIABLE CHECK")
    print("-" * 40)

    target = "SalePrice"

    if target in train_df.columns:
        print(f"{target} exists in training data.")
        print(f"Missing target values: {train_df[target].isna().sum()}")
        print(f"Minimum price: {train_df[target].min()}")
        print(f"Maximum price: {train_df[target].max()}")
        print(f"Mean price: {train_df[target].mean():,.2f}")
        print(f"Median price: {train_df[target].median():,.2f}")
    else:
        print(f"WARNING: {target} is missing from training data.")

    print(f"\nDoes test data contain {target}? "
          f"{target in test_df.columns}")

    # Inspect columns and data types
    print("\n3. TRAINING DATA COLUMNS AND DATA TYPES")
    print("-" * 40)
    print(train_df.dtypes.to_string())

    # Missing values
    print("\n4. MISSING VALUES")
    print("-" * 40)

    missing = train_df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values found in training data.")
    else:
        missing_report = pd.DataFrame({
            "MissingCount": missing,
            "MissingPercent": (
                missing / len(train_df) * 100
            ).round(2),
        })
        print(missing_report.to_string())

    # Duplicate rows
    print("\n5. DUPLICATE ROWS")
    print("-" * 40)
    print(f"Training duplicate rows: {train_df.duplicated().sum()}")
    print(f"Test duplicate rows: {test_df.duplicated().sum()}")

    # Unique values
    print("\n6. UNIQUE VALUE COUNTS")
    print("-" * 40)

    for column in train_df.columns:
        print(f"{column}: {train_df[column].nunique(dropna=True)}")

    # Column differences between train and test
    print("\n7. TRAIN/TEST COLUMN COMPARISON")
    print("-" * 40)

    train_features = set(train_df.columns) - {target}
    test_features = set(test_df.columns)

    print("Features in training but not test:",
          sorted(train_features - test_features))
    print("Features in test but not training:",
          sorted(test_features - train_features))

    # Dataset description preview
    print("\n8. DATA DESCRIPTION PREVIEW")
    print("-" * 40)

    with open(DESCRIPTION_PATH, "r", encoding="utf-8") as file:
        description = file.read()

    print(description[:3000])
    print("\n[Preview limited to the first 3000 characters]")

    # Sample submission structure
    print("\n9. SAMPLE SUBMISSION PREVIEW")
    print("-" * 40)
    print(submission_df.head().to_string())
    print("\nSample submission columns:",
          submission_df.columns.tolist())

    print("\n" + "=" * 60)
    print("INITIAL DATA INSPECTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()