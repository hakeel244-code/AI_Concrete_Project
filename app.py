import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Concrete Mix Optimizer",
    page_icon="🏗️",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model("concrete_strength_model.json")
    return model


model = load_model()

# ============================================================
# MATERIAL COST
# ============================================================

MATERIAL_COST = {
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
# ============================================================

CO2_FACTOR = {
    "Cement": 0.90,
    "Blast Furnace Slag": 0.07,
    "Fly Ash": 0.02,
    "Water": 0.0003,
    "Superplasticizer": 0.50,
    "Coarse Aggregate": 0.005,
    "Fine Aggregate": 0.005
}

# ============================================================
# PREDICT STRENGTH
# ============================================================

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

    return float(model.predict(data)[0])


# ============================================================
# BATCH PREDICTION
# ============================================================

def predict_candidates(candidates, age):

    data = candidates[
        [
            "Cement",
            "Blast Furnace Slag",
            "Fly Ash",
            "Water",
            "Superplasticizer",
            "Coarse Aggregate",
            "Fine Aggregate"
        ]
    ].copy()

    data["Age"] = age

    return model.predict(data)


# ============================================================
# COST CALCULATION
# ============================================================

def calculate_cost(mix):

    return sum(
        mix[name] * MATERIAL_COST[name]
        for name in MATERIAL_COST
    )


# ============================================================
# CO2 CALCULATION
# ============================================================

def calculate_co2(mix):

    return sum(
        mix[name] * CO2_FACTOR[name]
        for name in CO2_FACTOR
    )


# ============================================================
# MIX PROPERTIES
# ============================================================

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
        for name in MATERIAL_COST
    )

    return binder, wb_ratio, total_mass


# ============================================================
# FAST AI OPTIMIZER
# ============================================================

def optimize_mix_fast(
    current_mix,
    age,
    target_strength
):

    rng = np.random.default_rng(42)

    N = 25000

    candidates = pd.DataFrame({

        "Cement": rng.uniform(
            160,
            max(320, current_mix["Cement"] + 20),
            N
        ),

        "Blast Furnace Slag": rng.uniform(
            0,
            150,
            N
        ),

        "Fly Ash": rng.uniform(
            0,
            100,
            N
        ),

        "Water": rng.uniform(
            120,
            210,
            N
        ),

        "Superplasticizer": rng.uniform(
            0,
            10,
            N
        )
    })

    candidates["Binder"] = (
        candidates["Cement"]
        + candidates["Blast Furnace Slag"]
        + candidates["Fly Ash"]
    )

    candidates["WB"] = (
        candidates["Water"]
        / candidates["Binder"]
    )

    target_mass = 2250

    aggregate_mass = (
        target_mass
        - candidates["Cement"]
        - candidates["Blast Furnace Slag"]
        - candidates["Fly Ash"]
        - candidates["Water"]
        - candidates["Superplasticizer"]
    )

    candidates["Fine Aggregate"] = (
        aggregate_mass * 0.43
    )

    candidates["Coarse Aggregate"] = (
        aggregate_mass * 0.57
    )

    candidates["Total Mass"] = (
        candidates["Cement"]
        + candidates["Blast Furnace Slag"]
        + candidates["Fly Ash"]
        + candidates["Water"]
        + candidates["Superplasticizer"]
        + candidates["Coarse Aggregate"]
        + candidates["Fine Aggregate"]
    )

    valid = (
        (candidates["Cement"] >= 180)
        &
        (candidates["Binder"] >= 250)
        &
        (candidates["WB"] <= 0.60)
        &
        (candidates["WB"] >= 0.35)
        &
        (candidates["Fine Aggregate"] >= 650)
        &
        (candidates["Fine Aggregate"] <= 1000)
        &
        (candidates["Coarse Aggregate"] >= 850)
        &
        (candidates["Coarse Aggregate"] <= 1250)
        &
        (candidates["Total Mass"] >= 2200)
        &
        (candidates["Total Mass"] <= 2500)
    )

    candidates = candidates.loc[
        valid
    ].reset_index(drop=True)

    if len(candidates) == 0:
        return None

    candidates["Strength"] = predict_candidates(
        candidates,
        age
    )

    candidates = candidates[
        candidates["Strength"] >= target_strength
    ].copy()

    if len(candidates) == 0:
        return None

    candidates["Cost"] = (
        candidates["Cement"] * 7.0
        + candidates["Blast Furnace Slag"] * 2.0
        + candidates["Fly Ash"] * 2.0
        + candidates["Water"] * 0.05
        + candidates["Superplasticizer"] * 80.0
        + candidates["Coarse Aggregate"] * 1.0
        + candidates["Fine Aggregate"] * 1.0
    )

    candidates["CO2"] = (
        candidates["Cement"] * 0.90
        + candidates["Blast Furnace Slag"] * 0.07
        + candidates["Fly Ash"] * 0.02
        + candidates["Water"] * 0.0003
        + candidates["Superplasticizer"] * 0.50
        + candidates["Coarse Aggregate"] * 0.005
        + candidates["Fine Aggregate"] * 0.005
    )

    candidates["Strength Excess"] = (
        candidates["Strength"]
        - target_strength
    )

    def normalize(series):

        minimum = series.min()
        maximum = series.max()

        if maximum == minimum:
            return pd.Series(
                np.zeros(len(series)),
                index=series.index
            )

        return (
            series - minimum
        ) / (
            maximum - minimum
        )

    candidates["Cement Score"] = normalize(
        candidates["Cement"]
    )

    candidates["Cost Score"] = normalize(
        candidates["Cost"]
    )

    candidates["CO2 Score"] = normalize(
        candidates["CO2"]
    )

    candidates["Strength Score"] = normalize(
        candidates["Strength Excess"]
    )

    candidates["Score"] = (
        candidates["Cement Score"] * 0.35
        + candidates["Cost Score"] * 0.30
        + candidates["CO2 Score"] * 0.25
        + candidates["Strength Score"] * 0.10
    )

    best = candidates.loc[
        candidates["Score"].idxmin()
    ]

    return {
        "Cement": float(best["Cement"]),
        "Blast Furnace Slag": float(best["Blast Furnace Slag"]),
        "Fly Ash": float(best["Fly Ash"]),
        "Water": float(best["Water"]),
        "Superplasticizer": float(best["Superplasticizer"]),
        "Coarse Aggregate": float(best["Coarse Aggregate"]),
        "Fine Aggregate": float(best["Fine Aggregate"]),
        "Strength": float(best["Strength"]),
        "Cost": float(best["Cost"]),
        "CO2": float(best["CO2"]),
        "Binder": float(best["Binder"]),
        "WB Ratio": float(best["WB"]),
        "Total Mass": float(best["Total Mass"])
    }


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(
    grade,
    target_strength,
    age,
    slump,
    current_mix,
    current_strength,
    current_binder,
    current_wb,
    current_mass,
    current_cost,
    current_co2,
    optimized_mix,
    cement_saving,
    binder_saving,
    cost_saving,
    co2_saving,
    cement_pct,
    binder_pct,
    cost_pct,
    co2_pct
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "AI-Driven Sustainable Concrete Mix Design",
            styles["Title"]
        )
    )

    story.append(
        Paragraph(
            "Machine Learning-Based Concrete Strength Prediction "
            "and Sustainable Mix Optimization",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "1. Project Details",
            styles["Heading2"]
        )
    )

    project_data = [
        ["Project", "AI-Driven Sustainable Concrete Mix Design"],
        ["Degree", "B.E. Civil Engineering"],
        ["Student", "Ashyam Haqeel"],
        ["Institute", "Bearys Institute of Technology, Mangaluru"],
        ["Selected Grade", grade],
        ["Target Strength", f"{target_strength:.0f} MPa"],
        ["Age", f"{age} days"],
        ["Required Slump", f"{slump:.0f} mm"]
    ]

    table = Table(
        project_data,
        colWidths=[150, 350]
    )

    table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ])
    )

    story.append(table)

    story.append(Spacer(1, 14))

    # CURRENT MIX

    story.append(
        Paragraph(
            "2. Current Mix",
            styles["Heading2"]
        )
    )

    materials = [
        "Cement",
        "Blast Furnace Slag",
        "Fly Ash",
        "Water",
        "Superplasticizer",
        "Coarse Aggregate",
        "Fine Aggregate"
    ]

    current_data = [
        ["Material", "Quantity (kg/m3)"]
    ]

    for name in materials:
        current_data.append([
            name,
            f"{current_mix[name]:.2f}"
        ])

    table = Table(
        current_data,
        colWidths=[280, 220]
    )

    table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)
        ])
    )

    story.append(table)

    story.append(Spacer(1, 10))

    current_properties = [
        ["Parameter", "Current Value"],
        ["ML Predicted Strength", f"{current_strength:.2f} MPa"],
        ["Total Binder", f"{current_binder:.2f} kg/m3"],
        ["Water/Binder Ratio", f"{current_wb:.3f}"],
        ["Total Mix Mass", f"{current_mass:.2f} kg/m3"],
        ["Estimated Cost", f"Rs. {current_cost:.2f}/m3"],
        ["Estimated CO2", f"{current_co2:.2f} kg/m3"]
    ]

    table = Table(
        current_properties,
        colWidths=[280, 220]
    )

    table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)
        ])
    )

    story.append(table)

    story.append(Spacer(1, 14))

    # OPTIMIZED MIX

    story.append(
        Paragraph(
            "3. AI-Optimized Mix",
            styles["Heading2"]
        )
    )

    optimized_data = [
        ["Material", "Optimized Quantity (kg/m3)"]
    ]

    for name in materials:
        optimized_data.append([
            name,
            f"{optimized_mix[name]:.2f}"
        ])

    table = Table(
        optimized_data,
        colWidths=[280, 220]
    )

    table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)
        ])
    )

    story.append(table)

    story.append(Spacer(1, 10))

    optimized_properties = [
        ["Parameter", "Optimized Value"],
        ["Predicted Strength", f"{optimized_mix['Strength']:.2f} MPa"],
        ["Target Strength", f"{target_strength:.2f} MPa"],
        ["Total Binder", f"{optimized_mix['Binder']:.2f} kg/m3"],
        ["Water/Binder Ratio", f"{optimized_mix['WB Ratio']:.3f}"],
        ["Total Mix Mass", f"{optimized_mix['Total Mass']:.2f} kg/m3"],
        ["Estimated Cost", f"Rs. {optimized_mix['Cost']:.2f}/m3"],
        ["Estimated CO2", f"{optimized_mix['CO2']:.2f} kg/m3"]
    ]

    table = Table(
        optimized_properties,
        colWidths=[280, 220]
    )

    table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)
        ])
    )

    story.append(table)

    story.append(Spacer(1, 14))

    # SAVINGS

    story.append(
        Paragraph(
            "4. Resource Savings",
            styles["Heading2"]
        )
    )

    savings_data = [
        ["Parameter", "Saving", "Reduction"],
        ["Cement", f"{cement_saving:.2f} kg/m3", f"{cement_pct:.1f}%"],
        ["Binder", f"{binder_saving:.2f} kg/m3", f"{binder_pct:.1f}%"],
        ["Cost", f"Rs. {cost_saving:.2f}/m3", f"{cost_pct:.1f}%"],
        ["CO2", f"{co2_saving:.2f} kg/m3", f"{co2_pct:.1f}%"]
    ]

    table = Table(
        savings_data,
        colWidths=[180, 180, 140]
    )

    table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)
        ])
    )

    story.append(table)

    story.append(Spacer(1, 14))

    # COMPARISON

    story.append(
        Paragraph(
            "5. Current vs Optimized",
            styles["Heading2"]
        )
    )

    comparison_data = [
        ["Parameter", "Current", "Optimized"],
        ["Predicted Strength (MPa)", f"{current_strength:.2f}", f"{optimized_mix['Strength']:.2f}"],
        ["Target Strength (MPa)", f"{target_strength:.2f}", f"{target_strength:.2f}"],
        ["Cement (kg/m3)", f"{current_mix['Cement']:.2f}", f"{optimized_mix['Cement']:.2f}"],
        ["Binder (kg/m3)", f"{current_binder:.2f}", f"{optimized_mix['Binder']:.2f}"],
        ["Water/Binder", f"{current_wb:.3f}", f"{optimized_mix['WB Ratio']:.3f}"],
        ["Total Mass (kg/m3)", f"{current_mass:.2f}", f"{optimized_mix['Total Mass']:.2f}"],
        ["Cost (Rs./m3)", f"{current_cost:.2f}", f"{optimized_mix['Cost']:.2f}"],
        ["CO2 (kg/m3)", f"{current_co2:.2f}", f"{optimized_mix['CO2']:.2f}"]
    ]

    table = Table(
        comparison_data,
        colWidths=[230, 135, 135]
    )

    table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)
        ])
    )

    story.append(table)

    story.append(Spacer(1, 14))

    story.append(
        Paragraph(
            "6. Engineering Disclaimer",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            "This application is an ML-based academic decision-support "
            "system. The optimized mix must be verified through laboratory "
            "trial mixes, workability testing, durability testing and "
            "applicable concrete design standards before construction use. "
            "Cost and CO2 values are estimated using academic assumptions. "
            "Slump is a design requirement and is not predicted by the "
            "current ML model.",
            styles["BodyText"]
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# TITLE
# ============================================================

st.title(
    "🏗️ AI-Driven Sustainable Concrete Mix Design"
)

st.subheader(
    "Machine Learning-Based Concrete Strength Prediction "
    "and Sustainable Mix Optimization"
)

st.markdown(
    "**Created by: Ashyam Haqeel**  \n"
    "**B.E. Civil Engineering**  \n"
    "**Bearys Institute of Technology, Mangaluru**  \n"
    "**Major Project**"
)

st.write(
    "Enter the current concrete mix. The trained XGBoost model "
    "predicts its compressive strength, and the AI optimizer "
    "searches for a more resource-efficient mix."
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🏗️ Concrete Mix Design")

grade = st.sidebar.selectbox(
    "Select Grade",
    ["M20", "M25", "M30", "M35", "M40"],
    index=0
)

target_strength = {
    "M20": 20.0,
    "M25": 25.0,
    "M30": 30.0,
    "M35": 35.0,
    "M40": 40.0
}[grade]

st.sidebar.info(
    f"{grade} target = {target_strength:.0f} MPa"
)

# ============================================================
# INPUTS
# ============================================================

st.sidebar.subheader("Current Mix")

cement = st.sidebar.number_input(
    "Cement (kg/m³)",
    0.0, 1000.0, 300.0, 5.0
)

slag = st.sidebar.number_input(
    "Blast Furnace Slag (kg/m³)",
    0.0, 400.0, 100.0, 5.0
)

fly_ash = st.sidebar.number_input(
    "Fly Ash (kg/m³)",
    0.0, 300.0, 50.0, 5.0
)

water = st.sidebar.number_input(
    "Water (kg/m³)",
    0.0, 400.0, 180.0, 5.0
)

superplasticizer = st.sidebar.number_input(
    "Superplasticizer (kg/m³)",
    0.0, 50.0, 5.0, 0.5
)

coarse_aggregate = st.sidebar.number_input(
    "Coarse Aggregate (kg/m³)",
    0.0, 1500.0, 1000.0, 10.0
)

fine_aggregate = st.sidebar.number_input(
    "Fine Aggregate (kg/m³)",
    0.0, 1200.0, 800.0, 10.0
)

age = st.sidebar.number_input(
    "Age (days)",
    1, 365, 28, 1
)

slump = st.sidebar.number_input(
    "Required Slump (mm)",
    0.0, 250.0, 100.0, 5.0
)

# ============================================================
# CURRENT MIX
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

analyze = st.button(
    "🚀 Analyze & Optimize",
    type="primary",
    use_container_width=True
)

# ============================================================
# MAIN
# ============================================================

if analyze:

    current_strength = predict_strength(
        current_mix,
        age
    )

    current_binder, current_wb, current_mass = \
        calculate_properties(current_mix)

    current_cost = calculate_cost(
        current_mix
    )

    current_co2 = calculate_co2(
        current_mix
    )

    # ========================================================
    # CURRENT RESULT
    # ========================================================

    st.header(
        "🤖 Current Mix — Actual ML Prediction"
    )

    st.success(
        f"Predicted Compressive Strength: "
        f"{current_strength:.2f} MPa"
    )

    st.caption(
        "This is the actual prediction produced by the trained "
        "XGBoost model. The value is not forced to equal the "
        "selected concrete grade."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Predicted Strength",
        f"{current_strength:.2f} MPa"
    )

    col2.metric(
        "Grade Target",
        f"{target_strength:.0f} MPa"
    )

    col3.metric(
        "Estimated Cost",
        f"₹{current_cost:.2f}/m³"
    )

    col4.metric(
        "Estimated CO₂",
        f"{current_co2:.2f} kg/m³"
    )

    # ========================================================
    # GRADE CHECK
    # ========================================================

    st.subheader("🎯 Grade Assessment")

    if current_strength >= target_strength:

        st.success(
            f"✅ Current mix satisfies the {grade} target "
            f"of {target_strength:.0f} MPa."
        )

    else:

        st.error(
            f"❌ Current mix is below the {grade} target "
            f"of {target_strength:.0f} MPa."
        )

    # ========================================================
    # CURRENT PROPERTIES
    # ========================================================

    st.subheader("📊 Current Mix Properties")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Binder",
        f"{current_binder:.2f} kg/m³"
    )

    col2.metric(
        "Water/Binder",
        f"{current_wb:.3f}"
    )

    col3.metric(
        "Total Mix Mass",
        f"{current_mass:.2f} kg/m³"
    )

    # ========================================================
    # SLUMP
    # ========================================================

    st.subheader("📏 Workability Requirement")

    st.metric(
        "Required Slump",
        f"{slump:.0f} mm"
    )

    st.caption(
        "Slump is treated as an input requirement. The current "
        "ML dataset does not contain slump data, so this "
        "application does not predict slump."
    )

    # ========================================================
    # OPTIMIZATION
    # ========================================================

    st.header("⚙️ Fast AI Mix Optimization")

    st.write(
        f"Searching for a mix that achieves at least "
        f"{target_strength:.0f} MPa while reducing cement, "
        f"cost and CO₂."
    )

    with st.spinner(
        "🔄 AI is searching for the best feasible mix..."
    ):

        optimized_mix = optimize_mix_fast(
            current_mix,
            age,
            target_strength
        )

    # ========================================================
    # OPTIMIZED RESULT
    # ========================================================

    if optimized_mix is None:

        st.error(
            "❌ No feasible optimized mix was found within "
            "the current search range."
        )

        st.info(
            "Try increasing the current cement/binder content "
            "or selecting a lower grade."
        )

    else:

        st.success("✅ Optimized mix found.")

        st.header(
            "🏆 Recommended AI-Optimized Mix"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Predicted Strength",
            f"{optimized_mix['Strength']:.2f} MPa"
        )

        col2.metric(
            "Target",
            f"{target_strength:.0f} MPa"
        )

        col3.metric(
            "Estimated Cost",
            f"₹{optimized_mix['Cost']:.2f}/m³"
        )

        col4.metric(
            "Estimated CO₂",
            f"{optimized_mix['CO2']:.2f} kg/m³"
        )

        # ====================================================
        # MIX TABLE
        # ====================================================

        st.subheader(
            "📋 Optimized Mix Composition"
        )

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
                cement,
                slag,
                fly_ash,
                water,
                superplasticizer,
                coarse_aggregate,
                fine_aggregate
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
        # OPTIMIZED PROPERTIES
        # ====================================================

        st.subheader(
            "📊 Optimized Mix Properties"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Binder",
            f"{optimized_mix['Binder']:.2f} kg/m³"
        )

        col2.metric(
            "Water/Binder",
            f"{optimized_mix['WB Ratio']:.3f}"
        )

        col3.metric(
            "Total Mass",
            f"{optimized_mix['Total Mass']:.2f} kg/m³"
        )

        col4.metric(
            "Strength",
            f"{optimized_mix['Strength']:.2f} MPa"
        )

        # ====================================================
        # SAVINGS
        # ====================================================

        cement_saving = (
            cement - optimized_mix["Cement"]
        )

        binder_saving = (
            current_binder - optimized_mix["Binder"]
        )

        cost_saving = (
            current_cost - optimized_mix["Cost"]
        )

        co2_saving = (
            current_co2 - optimized_mix["CO2"]
        )

        cement_pct = (
            cement_saving / cement * 100
            if cement > 0 else 0
        )

        binder_pct = (
            binder_saving / current_binder * 100
            if current_binder > 0 else 0
        )

        cost_pct = (
            cost_saving / current_cost * 100
            if current_cost > 0 else 0
        )

        co2_pct = (
            co2_saving / current_co2 * 100
            if current_co2 > 0 else 0
        )

        # ====================================================
        # RESOURCE SAVINGS
        # ====================================================

        st.header(
            "🌱 Resource Savings"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Cement Saving",
            f"{cement_saving:.2f} kg/m³",
            f"{cement_pct:.1f}%"
        )

        col2.metric(
            "Binder Saving",
            f"{binder_saving:.2f} kg/m³",
            f"{binder_pct:.1f}%"
        )

        col3.metric(
            "Cost Saving",
            f"₹{cost_saving:.2f}/m³",
            f"{cost_pct:.1f}%"
        )

        col4.metric(
            "CO₂ Saving",
            f"{co2_saving:.2f} kg/m³",
            f"{co2_pct:.1f}%"
        )

        # ====================================================
        # COMPARISON
        # ====================================================

        st.header(
            "📈 Current vs Optimized"
        )

        comparison = pd.DataFrame({

            "Parameter": [
                "Grade",
                "Predicted Strength (MPa)",
                "Target Strength (MPa)",
                "Cement (kg/m³)",
                "Binder (kg/m³)",
                "Water/Binder",
                "Total Mass (kg/m³)",
                "Cost (₹/m³)",
                "CO₂ (kg/m³)"
            ],

            "Current": [
                grade,
                current_strength,
                target_strength,
                cement,
                current_binder,
                current_wb,
                current_mass,
                current_cost,
                current_co2
            ],

            "Optimized": [
                grade,
                optimized_mix["Strength"],
                target_strength,
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
        # FINAL RECOMMENDATION
        # ====================================================

        st.header(
            "🎯 Final Recommendation"
        )

        st.success(
            f"Recommended {grade} mix\n\n"
            f"Predicted strength = "
            f"{optimized_mix['Strength']:.2f} MPa\n\n"
            f"Required strength = "
            f"{target_strength:.0f} MPa"
        )

        if (
            optimized_mix["Cement"] < cement
            and optimized_mix["Cost"] < current_cost
            and optimized_mix["CO2"] < current_co2
        ):

            st.success(
                "🌱 The optimized mix reduces cement, estimated "
                "cost and estimated CO₂ while meeting the selected "
                "strength target."
            )

        else:

            st.info(
                "ℹ️ A feasible mix was found, but reducing all "
                "resources simultaneously is not guaranteed for "
                "every starting mix."
            )

        # ====================================================
        # PDF DOWNLOAD
        # ====================================================

        st.divider()

        st.header(
            "📄 Download Complete PDF Report"
        )

        pdf_file = create_pdf_report(
            grade,
            target_strength,
            age,
            slump,
            current_mix,
            current_strength,
            current_binder,
            current_wb,
            current_mass,
            current_cost,
            current_co2,
            optimized_mix,
            cement_saving,
            binder_saving,
            cost_saving,
            co2_saving,
            cement_pct,
            binder_pct,
            cost_pct,
            co2_pct
        )

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_file,
            file_name=f"{grade}_AI_Concrete_Mix_Report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

        st.caption(
            "The PDF contains the current mix, actual XGBoost "
            "prediction, optimized mix, cost, CO₂ and savings."
        )

        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.warning(
            "⚠️ Engineering Disclaimer: This application is an "
            "ML-based academic decision-support system. The "
            "optimized mix must be verified through laboratory "
            "trial mixes, workability testing, durability testing "
            "and applicable concrete design standards before "
            "construction use."
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Driven Sustainable Concrete Mix Design | "
    "B.E. Civil Engineering Major Project"
)
