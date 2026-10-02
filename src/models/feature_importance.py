
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "train.csv"
MODEL_PATH = ROOT_DIR / "models" / "house_price_model.joblib"
MODEL_INFO_PATH = ROOT_DIR / "models" / "model_info.json"
COMPARISON_PATH = ROOT_DIR / "reports" / "model_comparison.csv"
IMPORTANCE_PATH = ROOT_DIR / "reports" / "feature_importance.csv"


st.set_page_config(
    page_title="SmartHouse | Property Intelligence",
    page_icon="🏡",
    layout="wide",
)


# --------------------------------------------------
# Load project assets
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_training_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_model_info():
    if MODEL_INFO_PATH.exists():
        with open(MODEL_INFO_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


@st.cache_data
def load_comparison():
    if COMPARISON_PATH.exists():
        return pd.read_csv(COMPARISON_PATH)
    return pd.DataFrame()


@st.cache_data
def load_feature_importance():
    if IMPORTANCE_PATH.exists():
        return pd.read_csv(IMPORTANCE_PATH)
    return pd.DataFrame()


# --------------------------------------------------
# Check required files
# --------------------------------------------------
st.title("🏡 SmartHouse")
st.subheader("House Price Prediction & Property Intelligence System")
st.write(
    "Explore property characteristics, estimate house prices, "
    "and understand which features influence the trained model."
)

required_files = [DATA_PATH, MODEL_PATH]

missing_files = [path.name for path in required_files if not path.exists()]

if missing_files:
    st.error(
        "Required project files are missing: "
        + ", ".join(missing_files)
        + ". Check your project folders and train the model if necessary."
    )
    st.stop()

try:
    model = load_model()
    df = load_training_data()
    model_info = load_model_info()
    comparison_df = load_comparison()
    importance_df = load_feature_importance()
except Exception as error:
    st.error(f"Could not load the project assets: {error}")
    st.stop()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("SmartHouse")
st.sidebar.caption("Property Intelligence")
st.sidebar.markdown("---")
st.sidebar.write("Use the tabs to explore the project.")


# --------------------------------------------------
# Tabs
# --------------------------------------------------
prediction_tab, insights_tab, evaluation_tab, about_tab = st.tabs(
    [
        "💰 Price Prediction",
        "📊 Market Insights",
        "🧠 Model Evaluation",
        "ℹ️ About",
    ]
)


# ==================================================
# TAB 1: PRICE PREDICTION
# ==================================================
with prediction_tab:
    st.header("Estimate a Property's Sale Price")

    st.info(
        "Enter the property details below. The model uses a baseline "
        "training-data record for features that are not exposed in this "
        "form, so the estimate is an indicative model prediction rather "
        "than a professional valuation."
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            overall_qual = st.slider(
                "Overall Quality",
                min_value=1,
                max_value=10,
                value=7,
                help="Overall material and finish quality, rated from 1 to 10.",
            )
            living_area = st.number_input(
                "Above-ground Living Area (sq ft)",
                min_value=300,
                max_value=10000,
                value=1800,
                step=50,
            )
            lot_area = st.number_input(
                "Lot Area (sq ft)",
                min_value=500,
                max_value=200000,
                value=8000,
                step=250,
            )
            year_built = st.number_input(
                "Year Built",
                min_value=1870,
                max_value=2026,
                value=2000,
                step=1,
            )

        with col2:
            garage_cars = st.selectbox(
                "Garage Capacity (cars)",
                options=[0, 1, 2, 3, 4],
                index=2,
            )
            garage_area = st.number_input(
                "Garage Area (sq ft)",
                min_value=0,
                max_value=2000,
                value=500,
                step=25,
            )
            total_basement = st.number_input(
                "Total Basement Area (sq ft)",
                min_value=0,
                max_value=4000,
                value=900,
                step=50,
            )
            full_bath = st.selectbox(
                "Full Bathrooms",
                options=[0, 1, 2, 3, 4],
                index=2,
            )

        with col3:
            first_floor = st.number_input(
                "First Floor Area (sq ft)",
                min_value=0,
                max_value=4000,
                value=1000,
                step=50,
            )
            second_floor = st.number_input(
                "Second Floor Area (sq ft)",
                min_value=0,
                max_value=3000,
                value=500,
                step=50,
            )
            kitchen_qual = st.selectbox(
                "Kitchen Quality",
                options=["Ex", "Gd", "TA", "Fa", "Po"],
                index=2,
                help="Ex = Excellent, Gd = Good, TA = Typical/Average.",
            )
            neighborhood = st.selectbox(
                "Neighborhood",
                options=sorted(df["Neighborhood"].dropna().unique().tolist()),
                index=(
                    sorted(df["Neighborhood"].dropna().unique().tolist()).index(
                        "NAmes"
                    )
                    if "NAmes" in df["Neighborhood"].dropna().unique()
                    else 0
                ),
            )

        submitted = st.form_submit_button(
            "Predict House Price",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            # Use an existing training row as the baseline for unexposed
            # features, then replace fields supplied by the user.
            input_data = df.drop(columns=["SalePrice"]).iloc[[0]].copy()

            user_values = {
                "OverallQual": overall_qual,
                "GrLivArea": living_area,
                "LotArea": lot_area,
                "YearBuilt": year_built,
                "GarageCars": garage_cars,
                "GarageArea": garage_area,
                "TotalBsmtSF": total_basement,
                "FullBath": full_bath,
                "1stFlrSF": first_floor,
                "2ndFlrSF": second_floor,
                "KitchenQual": kitchen_qual,
                "Neighborhood": neighborhood,
            }

            for column, value in user_values.items():
                if column in input_data.columns:
                    input_data.loc[input_data.index[0], column] = value

            predicted_price = float(model.predict(input_data)[0])

            st.success("Prediction completed.")
            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.metric(
                    "Estimated Sale Price",
                    f"${predicted_price:,.0f}",
                )

            with metric_col2:
                st.metric(
                    "Estimated Price per Living Sq Ft",
                    (
                        f"${predicted_price / living_area:,.2f}"
                        if living_area > 0
                        else "N/A"
                    ),
                )

            prediction_summary = pd.DataFrame(
                {
                    "Property Feature": [
                        "Overall Quality",
                        "Living Area (sq ft)",
                        "Lot Area (sq ft)",
                        "Year Built",
                        "Garage Capacity",
                        "Garage Area (sq ft)",
                        "Basement Area (sq ft)",
                        "Full Bathrooms",
                        "Kitchen Quality",
                        "Neighborhood",
                    ],
                    "Value": [
                        overall_qual,
                        living_area,
                        lot_area,
                        year_built,
                        garage_cars,
                        garage_area,
                        total_basement,
                        full_bath,
                        kitchen_qual,
                        neighborhood,
                    ],
                }
            )

            with st.expander("View entered property details"):
                st.dataframe(prediction_summary, use_container_width=True)

            download_df = pd.DataFrame(
                [
                    {
                        **user_values,
                        "PredictedSalePrice": round(predicted_price, 2),
                    }
                ]
            )

            st.download_button(
                "Download Prediction as CSV",
                data=download_df.to_csv(index=False).encode("utf-8"),
                file_name="smarthouse_prediction.csv",
                mime="text/csv",
            )

            st.caption(
                "This is a machine-learning estimate based on historical "
                "training data. Actual market prices may differ."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ==================================================
# TAB 2: MARKET INSIGHTS
# ==================================================
with insights_tab:
    st.header("Explore the Housing Market")

    if "SalePrice" not in df.columns:
        st.warning("SalePrice is not available in the training dataset.")
    else:
        metric1, metric2, metric3, metric4 = st.columns(4)

        metric1.metric("Properties", f"{len(df):,}")
        metric2.metric("Median Sale Price", f"${df['SalePrice'].median():,.0f}")
        metric3.metric("Average Sale Price", f"${df['SalePrice'].mean():,.0f}")
        metric4.metric(
            "Price Range",
            f"${df['SalePrice'].min():,.0f} – ${df['SalePrice'].max():,.0f}",
        )

        st.subheader("Sale Price Distribution")
        st.caption("Distribution of sale prices in the training dataset.")
        st.histogram if False else None

        price_chart = pd.DataFrame({"Sale Price": df["SalePrice"]})
        st.bar_chart(
            price_chart["Sale Price"]
            .value_counts(bins=20, sort=False)
            .sort_index()
            .rename_axis("Price Range")
            .to_frame("Properties")
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Median Price by Overall Quality")
            quality_prices = (
                df.groupby("OverallQual", as_index=False)["SalePrice"]
                .median()
                .sort_values("OverallQual")
            )
            st.line_chart(
                quality_prices.set_index("OverallQual")["SalePrice"]
            )

        with col2:
            st.subheader("Median Price by Kitchen Quality")
            kitchen_prices = (
                df.groupby("KitchenQual")["SalePrice"]
                .median()
                .reindex(["Po", "Fa", "TA", "Gd", "Ex"])
                .dropna()
            )
            st.bar_chart(kitchen_prices)

        st.subheader("Neighborhood Comparison")
        neighborhood_prices = (
            df.groupby("Neighborhood", as_index=False)
            .agg(
                MedianSalePrice=("SalePrice", "median"),
                PropertyCount=("SalePrice", "size"),
            )
            .sort_values("MedianSalePrice", ascending=False)
        )

        st.bar_chart(
            neighborhood_prices.set_index("Neighborhood")["MedianSalePrice"]
        )

        with st.expander("View neighborhood statistics"):
            st.dataframe(neighborhood_prices, use_container_width=True)

        st.caption(
            "These summaries describe the supplied training dataset; "
            "they are not live market listings or current appraisals."
        )


# ==================================================
# TAB 3: MODEL EVALUATION
# ==================================================
with evaluation_tab:
    st.header("Model Performance and Feature Importance")

    rmse = model_info.get("validation_rmse")
    mae = model_info.get("validation_mae")
    model_name = model_info.get("model", "Selected model")

    st.subheader("Selected Model")
    st.write(f"**Model:** {model_name}")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Validation RMSE",
            f"${float(rmse):,.0f}" if rmse is not None else "Not available",
            help="Root Mean Squared Error on the validation set. Lower values indicate smaller errors overall, with larger errors penalized more.",
        )

    with metric2:
        st.metric(
            "Validation MAE",
            f"${float(mae):,.0f}" if mae is not None else "Not available",
            help="Mean Absolute Error on the validation set.",
        )

    with metric3:
        training_rows = model_info.get("training_rows")
        st.metric(
            "Training Rows",
            f"{int(training_rows):,}" if training_rows is not None else f"{len(df):,}",
        )

    st.caption(
        "Validation metrics measure performance on a held-out portion of the "
        "training dataset. They do not guarantee the same error on future homes."
    )

    st.divider()
    st.subheader("Model Comparison")

    if not comparison_df.empty:
        st.dataframe(comparison_df, use_container_width=True)

        # Detect common column names without assuming a specific capitalization.
        column_lookup = {str(column).lower(): column for column in comparison_df.columns}
        rmse_column = column_lookup.get("rmse")

        if rmse_column is not None:
            model_column = next(
                (
                    column
                    for column in comparison_df.columns
                    if str(column).lower() in ["model", "model_name", "name"]
                ),
                None,
            )

            chart_df = comparison_df.copy()

            if model_column is not None:
                chart_df = chart_df.set_index(model_column)

            st.subheader("Validation RMSE by Model")
            st.bar_chart(chart_df[rmse_column])

    else:
        st.info(
            "Model comparison file not found. Run "
            "python src/models/train_model.py to generate it."
        )

    st.divider()
    st.subheader("What Drives House Prices?")

    if not importance_df.empty:
        required_columns = {"Feature", "Importance"}

        if required_columns.issubset(importance_df.columns):
            top_features = (
                importance_df.sort_values("Importance", ascending=False)
                .head(15)
                .sort_values("Importance", ascending=True)
            )

            st.bar_chart(
                top_features.set_index("Feature")["Importance"],
                horizontal=True,
            )

            st.dataframe(
                importance_df.sort_values("Importance", ascending=False).head(15),
                use_container_width=True,
            )

            st.caption(
                "Feature importance describes how much the fitted model relies "
                "on each feature when making predictions. It does not establish "
                "causation or indicate whether a feature increases or decreases "
                "the predicted price."
            )
        else:
            st.warning(
                "The feature importance CSV must contain 'Feature' and "
                "'Importance' columns."
            )
    else:
        st.info(
            "Feature importance is not available yet. Run "
            "python src/models/feature_importance.py to generate it."
        )


# ==================================================
# TAB 4: ABOUT
# ==================================================
with about_tab:
    st.header("About SmartHouse")

    st.markdown(
        """
        **SmartHouse — House Price Prediction & Property Intelligence System**

        This capstone project uses supervised machine learning to estimate
        residential sale prices from property characteristics.

        ### Project components
        - **Data inspection:** review dataset structure and data quality.
        - **Exploratory analysis:** investigate sale prices and property features.
        - **Preprocessing:** impute missing values and encode categorical data.
        - **Model comparison:** evaluate candidate regressors using validation data.
        - **Prediction:** estimate a sale price for an entered property profile.
        - **Interpretability:** inspect feature importance for the selected model.

        ### Dataset
        The project uses the Kaggle *House Prices: Advanced Regression
        Techniques* training dataset. The training data contains historical
        sales records and is not a live property-market feed.

        ### Important limitations
        - Predictions are estimates, not professional valuations.
        - The prediction form exposes only some model inputs. Unexposed features
          retain values from a baseline training record.
        - Feature importance is not proof that a feature causes a price change.
        - Model performance on historical validation data may differ from
          performance on future properties.
        """
    )

    st.markdown("---")
    st.caption("SmartHouse | Data Science Capstone Project")