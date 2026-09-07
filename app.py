import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Concrete Mix Optimizer",
    page_icon="🏗️",
    layout="wide"
)

# ==========================================
# LOAD ML MODEL
# ==========================================

model = XGBRegressor()
model.load_model("concrete_strength_model.json")

# ==========================================
# TITLE
# ==========================================

st.title("🏗️ AI-Driven Sustainable Concrete Mix Design")
st.markdown("**Created by: Ashyam Haqeel**")
st.markdown("**B.E. Civil Engineering**")
st.markdown("**Bearys Institute of Technology, Mangaluru**")
st.markdown("**Major Project**")
st.subheader(
    "Machine Learning-Based Concrete Strength Prediction "
    "and Sustainable Mix Optimization"
)

st.write(
    "Enter the concrete mix parameters below to predict "
    "compressive strength using the trained XGBoost model."
)

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("Concrete Mix Parameters")

cement = st.sidebar.number_input(
    "Cement (kg/m³)",
    min_value=0.0,
    max_value=1000.0,
    value=300.0
)

slag = st.sidebar.number_input(
    "Blast Furnace Slag (kg/m³)",
    min_value=0.0,
    max_value=400.0,
    value=100.0
)

fly_ash = st.sidebar.number_input(
    "Fly Ash (kg/m³)",
    min_value=0.0,
    max_value=300.0,
    value=50.0
)

water = st.sidebar.number_input(
    "Water (kg/m³)",
    min_value=0.0,
    max_value=400.0,
    value=180.0
)

superplasticizer = st.sidebar.number_input(
    "Superplasticizer (kg/m³)",
    min_value=0.0,
    max_value=50.0,
    value=5.0
)

coarse_aggregate = st.sidebar.number_input(
    "Coarse Aggregate (kg/m³)",
    min_value=0.0,
    max_value=1500.0,
    value=1000.0
)

fine_aggregate = st.sidebar.number_input(
    "Fine Aggregate (kg/m³)",
    min_value=0.0,
    max_value=1200.0,
    value=800.0
)

age = st.sidebar.number_input(
    "Age (days)",
    min_value=1,
    max_value=365,
    value=28
)

# ==========================================
# CREATE INPUT DATA
# ==========================================

input_data = pd.DataFrame({
    "Cement": [cement],
    "Blast Furnace Slag": [slag],
    "Fly Ash": [fly_ash],
    "Water": [water],
    "Superplasticizer": [superplasticizer],
    "Coarse Aggregate": [coarse_aggregate],
    "Fine Aggregate": [fine_aggregate],
    "Age": [age]
})

# ==========================================
# PREDICTION
# ==========================================

st.header("🤖 Concrete Strength Prediction")

if st.button("Predict Compressive Strength", type="primary"):

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Compressive Strength: {prediction:.2f} MPa"
    )

    # ======================================
    # MIX CALCULATIONS
    # ======================================

    total_binder = cement + slag + fly_ash

    if total_binder > 0:
        water_binder_ratio = water / total_binder
    else:
        water_binder_ratio = 0

    total_mix_mass = (
        cement +
        slag +
        fly_ash +
        water +
        superplasticizer +
        coarse_aggregate +
        fine_aggregate
    )

    # ======================================
    # DISPLAY
    # ======================================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Binder",
        f"{total_binder:.2f} kg/m³"
    )

    col2.metric(
        "Water/Binder Ratio",
        f"{water_binder_ratio:.3f}"
    )

    col3.metric(
        "Total Mix Mass",
        f"{total_mix_mass:.2f} kg/m³"
    )

    # ======================================
    # TARGET CHECK
    # ======================================

    st.subheader("Strength Assessment")

    if prediction >= 40:
        st.success("✅ Target strength of 40 MPa achieved.")
    else:
        st.warning("⚠️ Target strength of 40 MPa not achieved.")

# ==========================================
# MIX SUMMARY
# ==========================================

st.header("📋 Mix Composition")

st.dataframe(
    input_data.T.rename(columns={0: "Value"}),
    use_container_width=True
)

# ==========================================
# SUSTAINABILITY
# ==========================================

st.header("🌱 Sustainability Analysis")

material_cost = {
    "Cement": 7.0,
    "Blast Furnace Slag": 2.0,
    "Fly Ash": 2.0,
    "Water": 0.05,
    "Superplasticizer": 80.0,
    "Coarse Aggregate": 1.0,
    "Fine Aggregate": 1.0
}

co2_factor = {
    "Cement": 0.90,
    "Blast Furnace Slag": 0.07,
    "Fly Ash": 0.02,
    "Water": 0.0003,
    "Superplasticizer": 0.50,
    "Coarse Aggregate": 0.005,
    "Fine Aggregate": 0.005
}

estimated_cost = sum(
    input_data.iloc[0][material] * material_cost[material]
    for material in material_cost
)

estimated_co2 = sum(
    input_data.iloc[0][material] * co2_factor[material]
    for material in co2_factor
)

col1, col2 = st.columns(2)

col1.metric(
    "Estimated Cost",
    f"₹{estimated_cost:.2f}/m³"
)

col2.metric(
    "Estimated CO₂",
    f"{estimated_co2:.2f} kg/m³"
)

# ==========================================
# DISCLAIMER
# ==========================================

st.divider()

st.caption(
    "⚠️ ML-based prediction only. Laboratory validation "
    "is required before practical construction use."
)
