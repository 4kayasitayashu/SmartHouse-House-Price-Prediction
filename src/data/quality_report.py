
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
TARGET = "SalePrice"


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Build one row per training-data column.
    quality_report = pd.DataFrame({
        "column": train_df.columns,
        "data_type": [
            str(train_df[column].dtype)
            for column in train_df.columns
        ],
        "non_missing_count": [
            train_df[column].notna().sum()
            for column in train_df.columns
        ],
        "missing_count": [
            train_df[column].isna().sum()
            for column in train_df.columns
        ],
        "missing_percent": [
            round(train_df[column].isna().mean() * 100, 2)
            for column in train_df.columns
        ],
        "unique_values": [
            train_df[column].nunique(dropna=True)
            for column in train_df.columns
        ],
    })

    quality_report["duplicate_column_flag"] = (
        quality_report["column"].duplicated()
    )

    quality_report.to_csv(
        REPORTS_DIR / "data_quality_report.csv",
        index=False,
    )

    train_features = set(train_df.columns) - {TARGET}
    test_features = set(test_df.columns)

    lines = [
        "SMARTHOUSE - DATA QUALITY SUMMARY",
        "=" * 45,
        "",
        f"Training shape: {train_df.shape}",
        f"Test shape: {test_df.shape}",
        f"Training duplicate rows: {train_df.duplicated().sum()}",
        f"Test duplicate rows: {test_df.duplicated().sum()}",
        "",
        "TARGET VARIABLE",
        "-" * 45,
    ]

    if TARGET in train_df.columns:
        lines.extend([
            f"Target: {TARGET}",
            f"Missing target values: {train_df[TARGET].isna().sum()}",
            f"Minimum: {train_df[TARGET].min()}",
            f"Maximum: {train_df[TARGET].max()}",
            f"Mean: {train_df[TARGET].mean():.2f}",
            f"Median: {train_df[TARGET].median():.2f}",
        ])

    lines.extend([
        "",
        "TRAIN/TEST FEATURE CONSISTENCY",
        "-" * 45,
        f"Features in training but not test: "
        f"{sorted(train_features - test_features)}",
        f"Features in test but not training: "
        f"{sorted(test_features - train_features)}",
        "",
        "MISSING VALUES",
        "-" * 45,
    ])

    missing_columns = quality_report[
        quality_report["missing_count"] > 0
    ].sort_values("missing_count", ascending=False)

    if missing_columns.empty:
        lines.append("No missing values in training data.")
    else:
        lines.append(
            missing_columns[
                ["column", "missing_count", "missing_percent"]
            ].to_string(index=False)
        )

    summary_path = REPORTS_DIR / "data_quality_summary.txt"
    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("Data-quality reports created successfully.")
    print(f"CSV report: {REPORTS_DIR / 'data_quality_report.csv'}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()