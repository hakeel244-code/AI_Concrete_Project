import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Concrete Mix Optimizer",
    page_icon="🏗️",
    layout="wide"
)

# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model("concrete_strength_model.json")
    return model

model = load_model()

# ============================================================
# MATERIAL COST ASSUMPTIONS
# ============================================================

material_cost = {
    "Cement": 7.0,
    "Blast Furnace Slag": 2.0,
    "Fly Ash": 2.0,
    "Water": 0.05,
    "Superplasticizer": 80.0,
    "Coarse Aggregate": 1.0,
    "Fine Aggregate": 1.0
}

# ============================================================
# CO2 FACTORS
# Academic/project assumptions
# ============================================================

co2_factor = {
    "Cement": 0.90,
    "Blast Furnace Slag": 0.07,
    "Fly Ash": 0.02,
    "Water": 0.0003,
    "Superplasticizer": 0.50,
    "Coarse Aggregate": 0.005,
    "Fine Aggregate": 0.005
}

# ============================================================
# FUNCTIONS
# ============================================================

def calculate_cost(mix):
    return sum(
        mix[name] * material_cost[name]
        for name in material_cost
    )


def calculate_co2(mix):
    return sum(
        mix[name] * co2_factor[name]
        for name in co2_factor
    )


def calculate_properties(mix):
    binder = (
        mix["Cement"]
        + mix["Blast Furnace Slag"]
        + mix["Fly Ash"]
    )

    if binder > 0:
        wb_ratio = mix["Water"] / binder
    else:
        wb_ratio = 0

    total_mass = sum(
        mix[name]
        for name in material_cost
    )

    return binder, wb_ratio, total_mass


def predict_strength(mix, age):
    data = pd.DataFrame({
        "Cement": [mix["Cement"]],
        "Blast Furnace Slag": [mix["Blast Furnace Slag"]],
        "Fly Ash": [mix["Fly Ash"]],
        "Water": [mix["Water"]],
        "Superplasticizer": [mix["Superplasticizer"]],
        "Coarse Aggregate": [mix["Coarse Aggregate"]],
        "Fine Aggregate": [mix["Fine Aggregate"]],
        "Age": [age]
    })

    prediction = model.predict(data)[0]

    return float(prediction)


# ============================================================
# SIMPLE OPTIMIZATION SEARCH
# ============================================================

def optimize_mix(
    current_mix,
    age,
    target_strength,
    current_cost,
    current_co2
):

    best_mix = None
    best_score = float("inf")

    # --------------------------------------------------------
    # Search ranges
    # --------------------------------------------------------

    cement_values = np.arange(180, min(current_mix["Cement"], 400) + 1, 10)

    if len(cement_values) == 0:
        cement_values = [180]

    slag_values = np.arange(0, 151, 15)
    flyash_values = np.arange(0, 101, 15)
    water_values = np.arange(120, 221, 10)
    sp_values = np.arange(0, 11, 2)

    # Aggregate values are calculated approximately
    # to maintain realistic total concrete mass.

    for cement in cement_values:

        for slag in slag_values:

            for fly_ash in flyash_values:

                binder = cement + slag + fly_ash

                if binder < 250:
                    continue

                for water in water_values:

                    wb_ratio = water / binder

                    if wb_ratio > 0.60:
                        continue

                    for sp in sp_values:

                        # Approximate aggregate quantities
                        remaining_mass = 2200 - (
                            cement
                            + slag
                            + fly_ash
                            + water
                            + sp
                        )

                        if remaining_mass < 1700:
                            continue

                        # Split aggregate approximately 45/55
                        fine = remaining_mass * 0.45
                        coarse = remaining_mass * 0.55

                        # Keep aggregates in reasonable range
                        if fine < 650 or fine > 1000:
                            continue

                        if coarse < 850 or coarse > 1200:
                            continue

                        candidate = {
                            "Cement": float(cement),
                            "Blast Furnace Slag": float(slag),
                            "Fly Ash": float(fly_ash),
                            "Water": float(water),
                            "Superplasticizer": float(sp),
                            "Coarse Aggregate": float(coarse),
                            "Fine Aggregate": float(fine)
                        }

                        binder, wb, total_mass = calculate_properties(candidate)

                        # Total mass constraint
                        if total_mass < 2200 or total_mass > 2500:
                            continue

                        # ------------------------------------------------
                        # Predict strength using REAL ML model
                        # ------------------------------------------------

                        strength = predict_strength(candidate, age)

                        # Required strength
                        if strength < target_strength:
                            continue

                        cost = calculate_cost(candidate)
                        co2 = calculate_co2(candidate)

                        # ------------------------------------------------
                        # Prefer a solution that improves current mix
                        # ------------------------------------------------

                        cost_saving = max(
                            0,
                            current_cost - cost
                        )

                        co2_saving = max(
                            0,
                            current_co2 - co2
                        )

                        cement_saving = max(
                            0,
                            current_mix["Cement"] - cement
                        )

                        # Penalize solutions that do not improve
                        # the current mix.
                        penalty = 0

                        if cost >= current_cost:
                            penalty += 5000

                        if co2 >= current_co2:
                            penalty += 5000

                        if cement >= current_mix["Cement"]:
                            penalty += 5000

                        # ------------------------------------------------
                        # Objective
                        # ------------------------------------------------

                        score = (
                            cost
                            + co2 * 5
                            + cement * 2
                            - strength * 2
                            - cost_saving * 2
                            - co2_saving * 2
                            - cement_saving
                            + penalty
                        )

                        if score < best_score:

                            best_score = score

                            best_mix = {
                                **candidate,
                                "Strength": strength,
                                "Cost": cost,
                                "CO2": co2,
                                "Binder": binder,
                                "WB Ratio": wb,
                                "Total Mass": total_mass
                            }

    return best_mix


# ============================================================
# TITLE
# ============================================================

st.title("🏗️ AI-Driven Sustainable Concrete Mix Design")

st.markdown("### Machine Learning-Based Concrete Strength Prediction and Mix Optimization")

st.markdown(
    "**Created by: Ashyam Haqeel**  \n"
    "**B.E. Civil Engineering**  \n"
    "**Bearys Institute of Technology, Mangaluru**  \n"
    "**Major Project**"
)

st.write(
    "This application uses a trained XGBoost machine learning model "
    "to predict concrete compressive strength and identify a more "
    "resource-efficient concrete mix."
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🏗️ Concrete Mix Design")

# ============================================================
# CONCRETE GRADE
# ============================================================

st.sidebar.subheader("Concrete Grade")

grade = st.sidebar.selectbox(
    "Select Concrete Grade",
    ["M20", "M25", "M30", "M35", "M40"],
    index=0
)

target_strength = {
    "M20": 20,
    "M25": 25,
    "M30": 30,
    "M35": 35,
    "M40": 40
}[grade]

st.sidebar.info(
    f"{grade} target strength = {target_strength} MPa"
)

# ============================================================
# CURRENT MIX INPUT
# ============================================================

st.sidebar.subheader("Current Mix")

cement = st.sidebar.number_input(
    "Cement (kg/m³)",
    min_value=0.0,
    max_value=1000.0,
    value=300.0,
    step=5.0
)

slag = st.sidebar.number_input(
    "Blast Furnace Slag (kg/m³)",
    min_value=0.0,
    max_value=400.0,
    value=100.0,
    step=5.0
)

fly_ash = st.sidebar.number_input(
    "Fly Ash (kg/m³)",
    min_value=0.0,
    max_value=300.0,
    value=50.0,
    step=5.0
)

water = st.sidebar.number_input(
    "Water (kg/m³)",
    min_value=0.0,
    max_value=400.0,
    value=180.0,
    step=5.0
)

superplasticizer = st.sidebar.number_input(
    "Superplasticizer (kg/m³)",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.5
)

coarse_aggregate = st.sidebar.number_input(
    "Coarse Aggregate (kg/m³)",
    min_value=0.0,
    max_value=1500.0,
    value=1000.0,
    step=10.0
)

fine_aggregate = st.sidebar.number_input(
    "Fine Aggregate (kg/m³)",
    min_value=0.0,
    max_value=1200.0,
    value=800.0,
    step=10.0
)

age = st.sidebar.number_input(
    "Age (days)",
    min_value=1,
    max_value=365,
    value=28,
    step=1
)

slump = st.sidebar.number_input(
    "Required Slump (mm)",
    min_value=0.0,
    max_value=250.0,
    value=100.0,
    step=5.0
)

# ============================================================
# CURRENT MIX DICTIONARY
# ============================================================

current_mix = {
    "Cement": cement,
    "Blast Furnace Slag": slag,
    "Fly Ash": fly_ash,
    "Water": water,
    "Superplasticizer": superplasticizer,
    "Coarse Aggregate": coarse_aggregate,
    "Fine Aggregate": fine_aggregate
}

# ============================================================
# BUTTON
# ============================================================

st.divider()

run_button = st.button(
    "🚀 Analyze & Optimize Concrete Mix",
    type="primary",
    use_container_width=True
)

# ============================================================
# MAIN APPLICATION
# ============================================================

if run_button:

    # ========================================================
    # CURRENT MIX PREDICTION
    # ========================================================

    current_strength = predict_strength(
        current_mix,
        age
    )

    current_binder, current_wb, current_mass = calculate_properties(
        current_mix
    )

    current_cost = calculate_cost(
        current_mix
    )

    current_co2 = calculate_co2(
        current_mix
    )

    # ========================================================
    # CURRENT MIX RESULTS
    # ========================================================

    st.header("🤖 Current Mix — AI Prediction")

    st.success(
        f"Predicted Compressive Strength: "
        f"{current_strength:.2f} MPa"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Strength",
        f"{current_strength:.2f} MPa"
    )

    col2.metric(
        "Cost",
        f"₹{current_cost:.2f}/m³"
    )

    col3.metric(
        "CO₂",
        f"{current_co2:.2f} kg/m³"
    )

    col4.metric(
        "Water/Binder",
        f"{current_wb:.3f}"
    )

    # ========================================================
    # CURRENT MIX DETAILS
    # ========================================================

    st.subheader("📋 Current Mix Composition")

    current_table = pd.DataFrame({
        "Material": [
            "Cement",
            "Blast Furnace Slag",
            "Fly Ash",
            "Water",
            "Superplasticizer",
            "Coarse Aggregate",
            "Fine Aggregate"
        ],
        "Quantity (kg/m³)": [
            cement,
            slag,
            fly_ash,
            water,
            superplasticizer,
            coarse_aggregate,
            fine_aggregate
        ]
    })

    st.dataframe(
        current_table,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # STRENGTH ASSESSMENT
    # ========================================================

    st.subheader("🎯 Grade Assessment")

    if current_strength >= target_strength:

        st.success(
            f"✅ The predicted strength of "
            f"{current_strength:.2f} MPa meets the "
            f"{grade} requirement of {target_strength} MPa."
        )

    else:

        st.warning(
            f"⚠️ The predicted strength of "
            f"{current_strength:.2f} MPa is below the "
            f"{grade} requirement of {target_strength} MPa."
        )

    # ========================================================
    # SLUMP
    # ========================================================

    st.subheader("📏 Slump Requirement")

    st.metric(
        "Required Slump",
        f"{slump:.0f} mm"
    )

    st.caption(
        "Slump is treated as a design requirement. "
        "The current UCI concrete dataset does not contain "
        "slump measurements, so slump is not predicted by the ML model."
    )

    # ========================================================
    # OPTIMIZATION
    # ========================================================

    st.header("⚙️ Automatic Sustainable Mix Optimization")

    st.write(
        f"The optimizer searches for a mix capable of achieving "
        f"the selected {grade} target of {target_strength} MPa "
        f"while reducing material use, estimated cost and CO₂ "
        f"where a feasible solution exists."
    )

    with st.spinner(
        "🔄 AI is searching for an improved concrete mix..."
    ):

        optimized_mix = optimize_mix(
            current_mix=current_mix,
            age=age,
            target_strength=target_strength,
            current_cost=current_cost,
            current_co2=current_co2
        )

    # ========================================================
    # NO SOLUTION
    # ========================================================

    if optimized_mix is None:

        st.error(
            "❌ No feasible optimized mix was found within the "
            "current search constraints."
        )

        st.info(
            "Try increasing the current cement/binder quantities "
            "or selecting a lower concrete grade."
        )

    else:

        # ====================================================
        # OPTIMIZED RESULT
        # ====================================================

        st.success(
            "✅ Optimized concrete mix found."
        )

        st.subheader("🏆 Recommended AI-Optimized Mix")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Predicted Strength",
            f"{optimized_mix['Strength']:.2f} MPa"
        )

        col2.metric(
            "Cost",
            f"₹{optimized_mix['Cost']:.2f}/m³"
        )

        col3.metric(
            "CO₂",
            f"{optimized_mix['CO2']:.2f} kg/m³"
        )

        col4.metric(
            "Water/Binder",
            f"{optimized_mix['WB Ratio']:.3f}"
        )

        # ====================================================
        # OPTIMIZED MIX TABLE
        # ====================================================

        st.subheader("📋 Optimized Mix Composition")

        optimized_table = pd.DataFrame({
            "Material": [
                "Cement",
                "Blast Furnace Slag",
                "Fly Ash",
                "Water",
                "Superplasticizer",
                "Coarse Aggregate",
                "Fine Aggregate"
            ],
            "Current (kg/m³)": [
                current_mix["Cement"],
                current_mix["Blast Furnace Slag"],
                current_mix["Fly Ash"],
                current_mix["Water"],
                current_mix["Superplasticizer"],
                current_mix["Coarse Aggregate"],
                current_mix["Fine Aggregate"]
            ],
            "Optimized (kg/m³)": [
                optimized_mix["Cement"],
                optimized_mix["Blast Furnace Slag"],
                optimized_mix["Fly Ash"],
                optimized_mix["Water"],
                optimized_mix["Superplasticizer"],
                optimized_mix["Coarse Aggregate"],
                optimized_mix["Fine Aggregate"]
            ]
        })

        st.dataframe(
            optimized_table.round(2),
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # OPTIMIZED MIX PROPERTIES
        # ====================================================

        st.subheader("📊 Optimized Mix Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Binder",
            f"{optimized_mix['Binder']:.2f} kg/m³"
        )

        col2.metric(
            "Total Mix Mass",
            f"{optimized_mix['Total Mass']:.2f} kg/m³"
        )

        col3.metric(
            "Strength",
            f"{optimized_mix['Strength']:.2f} MPa"
        )

        # ====================================================
        # SAVINGS CALCULATION
        # ====================================================

        cement_saving = (
            current_mix["Cement"]
            - optimized_mix["Cement"]
        )

        cost_saving = (
            current_cost
            - optimized_mix["Cost"]
        )

        co2_saving = (
            current_co2
            - optimized_mix["CO2"]
        )

        binder_saving = (
            current_binder
            - optimized_mix["Binder"]
        )

        mass_saving = (
            current_mass
            - optimized_mix["Total Mass"]
        )

        # Percentage savings
        cement_saving_pct = (
            cement_saving / cement * 100
            if cement > 0 else 0
        )

        cost_saving_pct = (
            cost_saving / current_cost * 100
            if current_cost > 0 else 0
        )

        co2_saving_pct = (
            co2_saving / current_co2 * 100
            if current_co2 > 0 else 0
        )

        binder_saving_pct = (
            binder_saving / current_binder * 100
            if current_binder > 0 else 0
        )

        # ====================================================
        # RESOURCE SAVINGS
        # ====================================================

        st.header("🌱 Resource & Sustainability Savings")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Cement Reduction",
            f"{cement_saving:.2f} kg/m³",
            f"{cement_saving_pct:.1f}%"
        )

        col2.metric(
            "Cost Saving",
            f"₹{cost_saving:.2f}/m³",
            f"{cost_saving_pct:.1f}%"
        )

        col3.metric(
            "CO₂ Reduction",
            f"{co2_saving:.2f} kg/m³",
            f"{co2_saving_pct:.1f}%"
        )

        col4.metric(
            "Binder Reduction",
            f"{binder_saving:.2f} kg/m³",
            f"{binder_saving_pct:.1f}%"
        )

        # ====================================================
        # COMPARISON
        # ====================================================

        st.header("📈 Current vs Optimized Mix")

        comparison = pd.DataFrame({
            "Parameter": [
                "Predicted Strength (MPa)",
                "Cement (kg/m³)",
                "Total Binder (kg/m³)",
                "Water/Binder Ratio",
                "Total Mix Mass (kg/m³)",
                "Estimated Cost (₹/m³)",
                "Estimated CO₂ (kg/m³)"
            ],
            "Current Mix": [
                current_strength,
                cement,
                current_binder,
                current_wb,
                current_mass,
                current_cost,
                current_co2
            ],
            "Optimized Mix": [
                optimized_mix["Strength"],
                optimized_mix["Cement"],
                optimized_mix["Binder"],
                optimized_mix["WB Ratio"],
                optimized_mix["Total Mass"],
                optimized_mix["Cost"],
                optimized_mix["CO2"]
            ]
        })

        st.dataframe(
            comparison.round(3),
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # FINAL ASSESSMENT
        # ====================================================

        st.header("🎯 Optimization Result")

        if optimized_mix["Strength"] >= target_strength:

            st.success(
                f"✅ Optimized mix achieves the {grade} target. "
                f"Predicted strength = "
                f"{optimized_mix['Strength']:.2f} MPa."
            )

        else:

            st.warning(
                "⚠️ Optimized mix did not achieve the required target."
            )

        if (
            optimized_mix["Cost"] < current_cost
            and optimized_mix["CO2"] < current_co2
            and optimized_mix["Cement"] < cement
        ):

            st.success(
                "🌱 The optimized mix reduces cement, estimated "
                "cost and estimated CO₂ compared with the current mix."
            )

        else:

            st.info(
                "ℹ️ The optimizer found the best feasible solution "
                "within the defined constraints, but all resource "
                "indicators may not be lower than the current mix."
            )

        # ====================================================
        # IMPORTANT ENGINEERING DISCLAIMER
        # ====================================================

        st.warning(
            "⚠️ Engineering Disclaimer: This application provides "
            "ML-based prediction and academic optimization only. "
            "The optimized mix must be verified through laboratory "
            "trial mixes, workability testing, durability checks and "
            "applicable concrete design standards before construction use."
        )

# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("📚 Project Information")

st.write(
    "**Dataset:** UCI Concrete Compressive Strength Dataset"
)

st.write(
    "**Machine Learning Model:** Optimized XGBoost Regressor"
)

st.write(
    "**Optimization Goal:** Strength + Cost + CO₂ + Resource Efficiency"
)

st.write(
    "**Explainability:** SHAP-based model interpretation "
    "can be integrated into the next version."
)

st.caption(
    "AI-Driven Sustainable Concrete Mix Design | "
    "B.E. Civil Engineering Major Project"
)
