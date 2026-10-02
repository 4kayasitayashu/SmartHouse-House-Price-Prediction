
import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"
INFO_PATH = ROOT / "models" / "model_info.json"
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

st.set_page_config(
    page_title="SmartHouse | Property Intelligence",
    page_icon="🏡",
    layout="wide",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_model_info():
    with open(INFO_PATH, encoding="utf-8") as file:
        return json.load(file)

model = load_model()
data = load_data()
model_info = load_model_info()

st.title("🏡 SmartHouse")
st.subheader("House Price Prediction & Property Intelligence")
st.write(
    "Predict estimated house prices and explore historical "
    "property sales using data science and machine learning."
)

m1, m2, m3 = st.columns(3)
m1.metric("Properties Analyzed", f"{len(data):,}")
m2.metric("Historical Median Price", f"${data['SalePrice'].median():,.0f}")
m3.metric("Validation RMSE", f"${model_info['validation_rmse']:,.0f}")

st.caption(
    f"Selected model: {model_info['model']} | "
    "Estimates are based on historical Ames, Iowa housing data."
)

tab_predict, tab_explore, tab_evaluate, tab_about = st.tabs(
    [
        "💰 Price Prediction",
        "📊 Market Insights",
        "🎯 Model Evaluation",
        "ℹ️ About",
    ]
)

# ---------------- PRICE PREDICTION ----------------
with tab_predict:
    st.header("Property Price Estimator")
    st.write("Enter the property's main characteristics below.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            quality = st.slider("Overall Quality", 1, 10, 5)
            living_area = st.number_input(
                "Above-ground Living Area (sq ft)",
                min_value=300, max_value=6000, value=1500, step=50
            )
            lot_area = st.number_input(
                "Lot Area (sq ft)",
                min_value=1000, max_value=200000, value=8000, step=500
            )
            basement_area = st.number_input(
                "Total Basement Area (sq ft)",
                min_value=0, max_value=3000, value=800, step=50
            )

        with col2:
            year_built = st.number_input(
                "Year Built",
                min_value=1872, max_value=2026, value=2000
            )
            garage_cars = st.selectbox(
                "Garage Capacity (cars)", [0, 1, 2, 3, 4], index=2
            )
            garage_area = st.number_input(
                "Garage Area (sq ft)",
                min_value=0, max_value=1500, value=400, step=25
            )
            full_bath = st.selectbox(
                "Full Bathrooms", [0, 1, 2, 3, 4], index=2
            )
            bedrooms = st.selectbox(
                "Bedrooms Above Ground", [0, 1, 2, 3, 4, 5, 6], index=3
            )

        with col3:
            kitchen_quality = st.selectbox(
                "Kitchen Quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=2
            )
            neighborhoods = sorted(data["Neighborhood"].dropna().unique())
            neighborhood = st.selectbox(
                "Neighborhood",
                neighborhoods,
                index=neighborhoods.index("NAmes")
                if "NAmes" in neighborhoods else 0,
            )
            house_style = st.selectbox(
                "House Style", sorted(data["HouseStyle"].dropna().unique())
            )
            building_type = st.selectbox(
                "Building Type", sorted(data["BldgType"].dropna().unique())
            )
            central_air = st.selectbox("Central Air", ["Y", "N"])

        submitted = st.form_submit_button(
            "Estimate Sale Price",
            type="primary",
            use_container_width=True,
        )

    st.info(
        "Transparency note: the model uses many property features. "
        "Features not shown in this form retain baseline values from a "
        "historical training example. Treat the result as an indicative "
        "estimate, not a formal appraisal."
    )

    if submitted:
        # Keep the complete feature schema expected by the saved pipeline.
        sample = data.drop(columns=["SalePrice", "Id"]).iloc[[0]].copy()

        sample["OverallQual"] = quality
        sample["GrLivArea"] = living_area
        sample["LotArea"] = lot_area
        sample["TotalBsmtSF"] = basement_area
        sample["YearBuilt"] = year_built
        sample["GarageCars"] = garage_cars
        sample["GarageArea"] = garage_area
        sample["FullBath"] = full_bath
        sample["BedroomAbvGr"] = bedrooms
        sample["KitchenQual"] = kitchen_quality
        sample["Neighborhood"] = neighborhood
        sample["HouseStyle"] = house_style
        sample["BldgType"] = building_type
        sample["CentralAir"] = central_air

        try:
            prediction = float(model.predict(sample)[0])

            st.success("Estimate generated successfully")
            st.metric("Estimated Sale Price", f"${prediction:,.0f}")

            median_price = float(data["SalePrice"].median())
            difference = prediction - median_price
            st.metric(
                "Difference from Historical Median",
                f"${difference:+,.0f}",
                help="Comparison with the dataset median, not a current market benchmark.",
            )

            st.caption(
                "Historical reference only: the dataset contains past sales "
                "from Ames, Iowa. Actual selling prices can differ."
            )
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.caption(
                "Check that the saved model matches the current training "
                "features. Retrain the model if the feature schema changed."
            )

# ---------------- MARKET INSIGHTS ----------------
with tab_explore:
    st.header("Historical Market Insights")
    st.caption("All prices below describe the historical training dataset.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Sale Price", f"${data['SalePrice'].mean():,.0f}")
    c2.metric("Median Sale Price", f"${data['SalePrice'].median():,.0f}")
    c3.metric(
        "Observed Price Range",
        f"${data['SalePrice'].min():,.0f} – ${data['SalePrice'].max():,.0f}",
    )

    st.subheader("Median Sale Price by Neighborhood")
    neighborhood_prices = (
        data.groupby("Neighborhood")["SalePrice"]
        .median()
        .sort_values(ascending=False)
    )
    st.bar_chart(neighborhood_prices)

    st.subheader("Sale Price Distribution")
    price_bins = pd.cut(data["SalePrice"], bins=20)
    price_distribution = price_bins.value_counts(sort=False)
    price_distribution.index = price_distribution.index.astype(str)
    st.bar_chart(price_distribution)

    st.subheader("Median Price by Overall Quality")
    quality_prices = data.groupby("OverallQual")["SalePrice"].median()
    st.line_chart(quality_prices)

    st.subheader("Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)

    st.download_button(
        "Download Historical Dataset (CSV)",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="smarthouse_historical_sales.csv",
        mime="text/csv",
    )

# ---------------- MODEL EVALUATION ----------------
with tab_evaluate:
    st.header("Model Performance")
    st.write(
        "These metrics come from the held-out validation set used during "
        "model selection, not from the final training set."
    )

    rmse = float(model_info["validation_rmse"])
    mae = float(model_info.get("validation_mae", 17497.73))

    c1, c2 = st.columns(2)
    c1.metric("Validation RMSE", f"${rmse:,.2f}")
    c2.metric("Validation MAE", f"${mae:,.2f}")

    st.markdown(
        """
        **How to interpret these metrics**

        - **MAE:** the average absolute difference between predicted and
          actual sale prices on the validation data.
        - **RMSE:** a prediction-error metric that penalizes larger errors
          more heavily than MAE.
        - Lower values indicate smaller errors on the evaluated dataset;
          neither metric guarantees accuracy for a particular property.
        """
    )

    comparison_path = ROOT / "reports" / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        st.subheader("Model Comparison")
        st.dataframe(comparison, use_container_width=True)

        if "RMSE" in comparison.columns:
            st.bar_chart(comparison.set_index(comparison.columns[0])["RMSE"])
        elif "rmse" in comparison.columns:
            st.bar_chart(comparison.set_index(comparison.columns[0])["rmse"])
        else:
            st.caption(
                "The comparison report uses different column names; "
                "see the table above for its recorded results."
            )
    else:
        st.warning(
            "Model comparison report not found. Run the model training "
            "script to generate it."
        )

# ---------------- ABOUT ----------------
with tab_about:
    st.header("About SmartHouse")
    st.write(
        "SmartHouse is a data science capstone project combining data "
        "cleaning, exploratory analysis, preprocessing, machine learning, "
        "model evaluation, and an interactive Streamlit dashboard."
    )

    st.write("**Selected model:**", model_info["model"])
    st.write("**Training records:**", f"{model_info['training_rows']:,}")
    st.write("**Validation RMSE:**", f"${model_info['validation_rmse']:,.2f}")

    st.subheader("Project Limitations")
    st.markdown(
        """
        - The data represents historical sales in Ames, Iowa, not all housing markets.
        - The model does not automatically account for current interest rates,
          inflation, or live property listings.
        - The prediction form exposes only some of the model's input features.
        - Estimates should not be used as formal appraisals or guarantees.
        """
    )