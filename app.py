"""
Clinical Trial Enrollment Dashboard - Shiny for Python

Interactive dashboard for analyzing clinical trial enrollment data
and predicting site enrollment success.

This project contains synthetic data and analysis created for demonstration purposes only.
"""

from shiny import App, ui, render, reactive
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from pathlib import Path

# Load data
def load_data():
    """Load synthetic clinical trial data."""
    # Check if data exists, generate if not
    data_path = Path("data/synthetic-trials.csv")
    if not data_path.exists():
        print("Generating synthetic data...")
        import subprocess
        subprocess.run(["python", "data/generate_data.py"], check=True)

    trials = pl.read_csv("data/synthetic-trials.csv")
    sites = pl.read_csv("data/synthetic-sites.csv")
    enrollment = pl.read_csv("data/synthetic-enrollment.csv")
    demographics = pl.read_csv("data/synthetic-patient-demographics.csv")

    return trials, sites, enrollment, demographics

# Load model artifacts
def load_model():
    """Load trained enrollment prediction model."""
    model = joblib.load("ml/enrollment_model.joblib")
    scaler = joblib.load("ml/scaler.joblib")
    encoders = joblib.load("ml/encoders.joblib")
    return model, scaler, encoders

trials, sites, enrollment, demographics = load_data()
model, scaler, encoders = load_model()

# UI Definition
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Overview",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Portfolio Summary"),
                ui.output_text("total_trials"),
                ui.output_text("total_sites"),
                ui.output_text("total_patients"),
                ui.hr(),
                ui.p("This project contains synthetic data and analysis created for demonstration purposes only.",
                     style="font-size: 0.85em; color: #666;")
            ),
            ui.card(
                ui.card_header("Enrollment by Phase"),
                ui.output_plot("phase_plot")
            ),
            ui.card(
                ui.card_header("Trials by Therapeutic Area"),
                ui.output_plot("therapeutic_area_plot")
            )
        )
    ),
    ui.nav_panel(
        "Site Performance",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "country_filter",
                    "Filter by Country:",
                    choices=["All"] + sorted(sites["country"].unique().to_list())
                ),
                ui.input_select(
                    "site_type_filter",
                    "Filter by Site Type:",
                    choices=["All"] + sorted(sites["site_type"].unique().to_list())
                )
            ),
            ui.card(
                ui.card_header("Site Enrollment Performance"),
                ui.output_plot("site_performance_plot")
            ),
            ui.card(
                ui.card_header("Top Performing Sites"),
                ui.output_table("top_sites_table")
            )
        )
    ),
    ui.nav_panel(
        "Enrollment Predictor",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Site Characteristics"),
                ui.input_select(
                    "pred_phase",
                    "Trial Phase:",
                    choices=sorted(trials["phase"].unique().to_list())
                ),
                ui.input_select(
                    "pred_therapeutic_area",
                    "Therapeutic Area:",
                    choices=sorted(trials["therapeutic_area"].unique().to_list())
                ),
                ui.input_select(
                    "pred_country",
                    "Country:",
                    choices=sorted(sites["country"].unique().to_list())
                ),
                ui.input_select(
                    "pred_site_type",
                    "Site Type:",
                    choices=sorted(sites["site_type"].unique().to_list())
                ),
                ui.input_select(
                    "pred_population_density",
                    "Population Density:",
                    choices=["Urban", "Suburban", "Rural"]
                ),
                ui.input_slider(
                    "pred_investigator_exp",
                    "Investigator Experience (years):",
                    min=1, max=25, value=10
                ),
                ui.input_slider(
                    "pred_staff_count",
                    "Site Staff Count:",
                    min=3, max=30, value=15
                ),
                ui.input_slider(
                    "pred_prior_trials",
                    "Prior Trials Completed:",
                    min=0, max=50, value=10
                ),
                ui.input_numeric(
                    "pred_database_size",
                    "Patient Database Size:",
                    value=10000, min=500, max=50000
                ),
                ui.input_slider(
                    "pred_distance",
                    "Distance to Metro (km):",
                    min=0, max=200, value=20
                ),
                ui.input_slider(
                    "pred_inclusion",
                    "Inclusion Criteria Count:",
                    min=5, max=20, value=10
                ),
                ui.input_slider(
                    "pred_exclusion",
                    "Exclusion Criteria Count:",
                    min=3, max=15, value=8
                ),
                ui.input_slider(
                    "pred_amendments",
                    "Protocol Amendments:",
                    min=0, max=5, value=1
                ),
                ui.input_numeric(
                    "pred_target",
                    "Target Enrollment per Site:",
                    value=50, min=1, max=500
                ),
                ui.input_action_button("predict_btn", "Predict Enrollment Success", class_="btn-primary")
            ),
            ui.card(
                ui.card_header("Enrollment Success Prediction"),
                ui.output_ui("prediction_result")
            ),
            ui.card(
                ui.card_header("About the Model"),
                ui.markdown(
                    """
                    This model predicts the probability that a clinical trial site will meet its enrollment target
                    based on site and trial characteristics.

                    **Top Factors:**
                    - Target enrollment per site
                    - Investigator experience
                    - Site staff count
                    - Trial phase
                    - Patient database size

                    **Note:** Model trained on synthetic data for demonstration purposes only.
                    """
                )
            )
        )
    ),
    title="VitalGen Clinical Trial Dashboard",
    theme=ui.Theme.from_brand(__file__)
)


# Server Logic
def server(input, output, session):

    @output
    @render.text
    def total_trials():
        return f"Total Trials: {len(trials)}"

    @output
    @render.text
    def total_sites():
        return f"Total Sites: {len(sites)}"

    @output
    @render.text
    def total_patients():
        return f"Total Patients: {len(demographics)}"

    @output
    @render.plot
    def phase_plot():
        phase_counts = trials.group_by("phase").agg(pl.count("trial_id").alias("count")).to_pandas()

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#003D7A", "#00A3A1", "#5CB3CC"]
        ax.bar(phase_counts["phase"], phase_counts["count"], color=colors)
        ax.set_xlabel("Phase")
        ax.set_ylabel("Number of Trials")
        ax.set_title("Active Trials by Phase")
        return fig

    @output
    @render.plot
    def therapeutic_area_plot():
        ta_counts = trials.group_by("therapeutic_area").agg(pl.count("trial_id").alias("count")).sort("count", descending=True).to_pandas()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(ta_counts["therapeutic_area"], ta_counts["count"], color="#00A3A1")
        ax.set_xlabel("Number of Trials")
        ax.set_title("Trials by Therapeutic Area")
        return fig

    @output
    @render.plot
    def site_performance_plot():
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
        if input.country_filter() != "All":
            filtered = filtered.filter(pl.col("country") == input.country_filter())
        if input.site_type_filter() != "All":
            filtered = filtered.filter(pl.col("site_type") == input.site_type_filter())

        perf_by_type = filtered.group_by("site_type").agg(
            pl.col("patients_per_month").mean().alias("avg_rate")
        ).sort("avg_rate", descending=True).to_pandas()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(perf_by_type["site_type"], perf_by_type["avg_rate"], color="#003D7A")
        ax.set_ylabel("Average Patients per Month")
        ax.set_title("Average Enrollment Rate by Site Type")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    @output
    @render.table
    def top_sites_table():
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
        if input.country_filter() != "All":
            filtered = filtered.filter(pl.col("country") == input.country_filter())
        if input.site_type_filter() != "All":
            filtered = filtered.filter(pl.col("site_type") == input.site_type_filter())

        top_sites = (
            filtered
            .sort("patients_per_month", descending=True)
            .head(10)
            .select(["site_id", "country", "site_type", "total_enrolled", "patients_per_month"])
            .to_pandas()
        )

        top_sites["patients_per_month"] = top_sites["patients_per_month"].round(2)
        top_sites.columns = ["Site ID", "Country", "Site Type", "Total Enrolled", "Rate (pts/month)"]

        return top_sites

    @output
    @render.ui
    @reactive.event(input.predict_btn)
    def prediction_result():
        # Prepare features
        features = [
            encoders["phase"].transform([input.pred_phase()])[0],
            encoders["therapeutic_area"].transform([input.pred_therapeutic_area()])[0],
            encoders["country"].transform([input.pred_country()])[0],
            encoders["site_type"].transform([input.pred_site_type()])[0],
            encoders["population_density"].transform([input.pred_population_density()])[0],
            input.pred_investigator_exp(),
            input.pred_staff_count(),
            input.pred_prior_trials(),
            input.pred_database_size(),
            input.pred_distance(),
            input.pred_inclusion(),
            input.pred_exclusion(),
            input.pred_amendments(),
            input.pred_target()
        ]

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        # Make prediction
        prob = model.predict_proba(X_scaled)[0, 1]
        prediction = "Success" if prob >= 0.5 else "At Risk"

        # Determine color
        color = "#7AB800" if prob >= 0.7 else "#F47B20" if prob >= 0.4 else "#D32F2F"

        return ui.div(
            ui.h3(f"Prediction: {prediction}", style=f"color: {color};"),
            ui.h2(f"{prob*100:.1f}%", style=f"color: {color}; font-size: 3em;"),
            ui.p(f"Probability of meeting enrollment target"),
            ui.hr(),
            ui.p(
                f"""
                **Interpretation:**
                {"✓ High likelihood of success" if prob >= 0.7 else
                 "⚠ Moderate risk - consider optimization strategies" if prob >= 0.4 else
                 "✗ High risk - significant intervention needed"}
                """
            )
        )


app = App(app_ui, server)
