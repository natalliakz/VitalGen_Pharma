"""
Clinical Trial Enrollment Dashboard - Streamlit

Alternative interactive dashboard using Streamlit for analyzing clinical trial
enrollment data and predicting site enrollment success.

This project contains synthetic data and analysis created for demonstration purposes only.
"""

import streamlit as st
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="VitalGen Clinical Trial Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for branding
st.markdown("""
    <style>
    .main-header {
        color: #003D7A;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1em;
        border-radius: 0.5em;
        border-left: 4px solid #00A3A1;
    }
    .success-text {
        color: #7AB800;
        font-weight: bold;
    }
    .warning-text {
        color: #F47B20;
        font-weight: bold;
    }
    .danger-text {
        color: #D32F2F;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load synthetic clinical trial data."""
    # Check if data exists, generate if not
    data_path = Path("data/synthetic-trials.csv")
    if not data_path.exists():
        st.info("Generating synthetic data...")
        import subprocess
        subprocess.run(["python", "data/generate_data.py"], check=True)

    trials = pl.read_csv("data/synthetic-trials.csv")
    sites = pl.read_csv("data/synthetic-sites.csv")
    enrollment = pl.read_csv("data/synthetic-enrollment.csv")
    demographics = pl.read_csv("data/synthetic-patient-demographics.csv")

    return trials, sites, enrollment, demographics


@st.cache_resource
def load_model():
    """Load trained enrollment prediction model."""
    model = joblib.load("ml/enrollment_model.joblib")
    scaler = joblib.load("ml/scaler.joblib")
    encoders = joblib.load("ml/encoders.joblib")
    return model, scaler, encoders


# Load data and model
trials, sites, enrollment, demographics = load_data()
model, scaler, encoders = load_model()

# Main title
st.markdown('<div class="main-header">🏥 VitalGen Clinical Trial Dashboard</div>', unsafe_allow_html=True)
st.markdown("*This project contains synthetic data and analysis created for demonstration purposes only.*")
st.divider()

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Site Performance", "Enrollment Predictor"],
    index=0
)

# Overview Page
if page == "Overview":
    st.header("Portfolio Overview")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trials", len(trials))

    with col2:
        st.metric("Total Sites", len(sites))

    with col3:
        st.metric("Total Patients", len(demographics))

    with col4:
        avg_enrollment = (
            enrollment.group_by("site_id")
            .agg([(pl.col("cumulative_enrolled").max() / pl.col("month").max()).alias("rate")])
            .select(pl.col("rate").mean())
            .item()
        )
        st.metric("Avg Enrollment Rate", f"{avg_enrollment:.2f} pts/mo")

    st.divider()

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Enrollment by Phase")
        phase_counts = trials.group_by("phase").agg(pl.count("trial_id").alias("count")).to_pandas()

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#003D7A", "#00A3A1", "#5CB3CC"]
        ax.bar(phase_counts["phase"], phase_counts["count"], color=colors)
        ax.set_xlabel("Phase")
        ax.set_ylabel("Number of Trials")
        ax.set_title("Active Trials by Phase")
        st.pyplot(fig)

    with col2:
        st.subheader("Trials by Therapeutic Area")
        ta_counts = trials.group_by("therapeutic_area").agg(
            pl.count("trial_id").alias("count")
        ).sort("count", descending=True).to_pandas()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(ta_counts["therapeutic_area"], ta_counts["count"], color="#00A3A1")
        ax.set_xlabel("Number of Trials")
        ax.set_title("Trials by Therapeutic Area")
        st.pyplot(fig)

    # Enrollment timeline
    st.subheader("Enrollment Timeline")
    enrollment_ts = (
        enrollment
        .with_columns([
            pl.col("enrollment_date").str.strptime(pl.Date, "%Y-%m-%d")
        ])
        .group_by("enrollment_date")
        .agg([
            pl.col("monthly_enrolled").sum().alias("total_enrolled")
        ])
        .sort("enrollment_date")
        .with_columns([
            pl.col("total_enrolled").cum_sum().alias("cumulative_total")
        ])
        .to_pandas()
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(enrollment_ts["enrollment_date"], enrollment_ts["cumulative_total"],
            color="#003D7A", linewidth=2)
    ax.fill_between(enrollment_ts["enrollment_date"], enrollment_ts["cumulative_total"],
                    alpha=0.3, color="#003D7A")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Patients Enrolled")
    ax.set_title("Cumulative Patient Enrollment Over Time")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Site Performance Page
elif page == "Site Performance":
    st.header("Site Performance Analysis")

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        country_filter = st.selectbox(
            "Filter by Country:",
            ["All"] + sorted(sites["country"].unique().to_list())
        )

    with col2:
        site_type_filter = st.selectbox(
            "Filter by Site Type:",
            ["All"] + sorted(sites["site_type"].unique().to_list())
        )

    # Calculate site enrollment rates
    site_enroll = (
        enrollment
        .group_by("site_id")
        .agg([
            pl.col("cumulative_enrolled").max().alias("total_enrolled"),
            pl.col("month").max().alias("months_active")
        ])
        .join(sites, on="site_id")
        .with_columns([
            (pl.col("total_enrolled") / pl.col("months_active")).alias("patients_per_month")
        ])
    )

    # Apply filters
    filtered = site_enroll
    if country_filter != "All":
        filtered = filtered.filter(pl.col("country") == country_filter)
    if site_type_filter != "All":
        filtered = filtered.filter(pl.col("site_type") == site_type_filter)

    st.divider()

    # Performance by site type
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Enrollment Rate by Site Type")
        perf_by_type = filtered.group_by("site_type").agg(
            pl.col("patients_per_month").mean().alias("avg_rate")
        ).sort("avg_rate", descending=True).to_pandas()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(perf_by_type["site_type"], perf_by_type["avg_rate"], color="#003D7A")
        ax.set_ylabel("Average Patients per Month")
        ax.set_title("Average Enrollment Rate by Site Type")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Enrollment by Country")
        country_perf = filtered.group_by("country").agg([
            pl.col("total_enrolled").sum().alias("total_patients")
        ]).sort("total_patients", descending=True).to_pandas()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(country_perf["country"], country_perf["total_patients"], color="#00A3A1")
        ax.set_xlabel("Total Patients Enrolled")
        ax.set_title("Patient Enrollment by Country")
        st.pyplot(fig)

    # Top performing sites
    st.subheader("Top 10 Performing Sites")
    top_sites = (
        filtered
        .sort("patients_per_month", descending=True)
        .head(10)
        .select(["site_id", "country", "site_type", "total_enrolled", "patients_per_month"])
        .to_pandas()
    )

    top_sites["patients_per_month"] = top_sites["patients_per_month"].round(2)
    top_sites.columns = ["Site ID", "Country", "Site Type", "Total Enrolled", "Rate (pts/month)"]

    st.dataframe(top_sites, use_container_width=True)

    # Investigator experience impact
    st.subheader("Investigator Experience Impact")
    site_enroll_exp = filtered.with_columns([
        pl.when(pl.col("investigator_experience_years") < 5)
        .then(pl.lit("< 5 years"))
        .when(pl.col("investigator_experience_years") < 10)
        .then(pl.lit("5-10 years"))
        .when(pl.col("investigator_experience_years") < 15)
        .then(pl.lit("10-15 years"))
        .otherwise(pl.lit("15+ years"))
        .alias("experience_category")
    ])

    experience_impact = site_enroll_exp.group_by("experience_category").agg(
        pl.col("patients_per_month").mean().alias("avg_enrollment_rate")
    ).to_pandas()

    order = ["< 5 years", "5-10 years", "10-15 years", "15+ years"]
    experience_impact["experience_category"] = pd.Categorical(
        experience_impact["experience_category"], categories=order, ordered=True
    )
    experience_impact = experience_impact.sort_values("experience_category")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(experience_impact["experience_category"], experience_impact["avg_enrollment_rate"],
           color="#003D7A")
    ax.set_xlabel("Investigator Experience")
    ax.set_ylabel("Average Patients per Month")
    ax.set_title("Enrollment Rate by Investigator Experience")
    st.pyplot(fig)

# Enrollment Predictor Page
elif page == "Enrollment Predictor":
    st.header("Enrollment Success Predictor")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Site Characteristics")

        # Trial characteristics
        st.markdown("**Trial Characteristics**")
        pred_phase = st.selectbox("Trial Phase:", sorted(trials["phase"].unique().to_list()))
        pred_therapeutic_area = st.selectbox("Therapeutic Area:", sorted(trials["therapeutic_area"].unique().to_list()))

        st.divider()

        # Site characteristics
        st.markdown("**Site Characteristics**")
        pred_country = st.selectbox("Country:", sorted(sites["country"].unique().to_list()))
        pred_site_type = st.selectbox("Site Type:", sorted(sites["site_type"].unique().to_list()))
        pred_population_density = st.selectbox("Population Density:", ["Urban", "Suburban", "Rural"])

        st.divider()

        # Numeric inputs
        st.markdown("**Site Capabilities**")
        pred_investigator_exp = st.slider("Investigator Experience (years):", 1, 25, 10)
        pred_staff_count = st.slider("Site Staff Count:", 3, 30, 15)
        pred_prior_trials = st.slider("Prior Trials Completed:", 0, 50, 10)
        pred_database_size = st.number_input("Patient Database Size:", min_value=500, max_value=50000, value=10000, step=500)
        pred_distance = st.slider("Distance to Metro (km):", 0, 200, 20)

        st.divider()

        # Protocol complexity
        st.markdown("**Protocol Complexity**")
        pred_inclusion = st.slider("Inclusion Criteria Count:", 5, 20, 10)
        pred_exclusion = st.slider("Exclusion Criteria Count:", 3, 15, 8)
        pred_amendments = st.slider("Protocol Amendments:", 0, 5, 1)
        pred_target = st.number_input("Target Enrollment per Site:", min_value=1, max_value=500, value=50)

        st.divider()

        predict_btn = st.button("🔮 Predict Enrollment Success", type="primary", use_container_width=True)

    with col2:
        st.subheader("Prediction Results")

        if predict_btn:
            # Prepare features
            features = [
                encoders["phase"].transform([pred_phase])[0],
                encoders["therapeutic_area"].transform([pred_therapeutic_area])[0],
                encoders["country"].transform([pred_country])[0],
                encoders["site_type"].transform([pred_site_type])[0],
                encoders["population_density"].transform([pred_population_density])[0],
                pred_investigator_exp,
                pred_staff_count,
                pred_prior_trials,
                pred_database_size,
                pred_distance,
                pred_inclusion,
                pred_exclusion,
                pred_amendments,
                pred_target
            ]

            X = np.array(features).reshape(1, -1)
            X_scaled = scaler.transform(X)

            # Make prediction
            prob = model.predict_proba(X_scaled)[0, 1]
            prediction = "Success" if prob >= 0.5 else "At Risk"

            # Determine color and interpretation
            if prob >= 0.7:
                color_class = "success-text"
                interpretation = "✅ High likelihood of success"
                recommendation = "This site has excellent characteristics for meeting enrollment targets."
            elif prob >= 0.4:
                color_class = "warning-text"
                interpretation = "⚠️ Moderate risk - consider optimization strategies"
                recommendation = "Consider providing additional support or resources to improve enrollment success."
            else:
                color_class = "danger-text"
                interpretation = "❌ High risk - significant intervention needed"
                recommendation = "This site may struggle to meet targets. Consider additional sites or protocol modifications."

            # Display results
            st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<h1 class="{color_class}">{prob*100:.1f}%</h1>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size: 1.2em;">Probability of meeting enrollment target</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.divider()

            st.markdown(f"**Prediction:** {prediction}")
            st.markdown(f"**Interpretation:** {interpretation}")
            st.markdown(f"**Recommendation:** {recommendation}")

            st.divider()

            # Show top factors
            st.subheader("Key Contributing Factors")
            st.markdown("""
            The model considers these key factors (in order of importance):
            1. **Target enrollment per site** - Higher targets increase difficulty
            2. **Investigator experience** - More experienced investigators perform better
            3. **Site staff count** - Adequate staffing is critical
            4. **Trial phase** - Phase affects complexity and enrollment
            5. **Patient database size** - Larger databases improve recruitment
            """)

        else:
            st.info("👈 Adjust the site characteristics in the sidebar and click 'Predict Enrollment Success' to see results.")

            st.subheader("About the Model")
            st.markdown("""
            This machine learning model predicts the probability that a clinical trial site
            will meet its enrollment target based on various site and trial characteristics.

            **Model Details:**
            - Algorithm: Random Forest Classifier
            - Test Accuracy: 97.9%
            - Test ROC-AUC: 0.82

            **Note:** Model trained on synthetic data for demonstration purposes only.
            """)

# Footer
st.divider()
st.markdown("*Dashboard built with Streamlit • Data is synthetic and AI-generated for demonstration purposes*")
