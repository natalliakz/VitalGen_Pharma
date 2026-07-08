"""
Clinical Trial Enrollment Dashboard - Shiny for Python

Interactive dashboard for analyzing clinical trial enrollment data
and predicting site enrollment success.

This project contains synthetic data and analysis created for demonstration
purposes only.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from shiny import App, reactive, render, ui


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load synthetic clinical trial datasets."""
    data_path = Path("data/synthetic-trials.csv")
    if not data_path.exists():
        import subprocess
        subprocess.run(["python", "data/generate_data.py"], check=True)

    trials = pl.read_csv("data/synthetic-trials.csv")
    sites = pl.read_csv("data/synthetic-sites.csv")
    enrollment = pl.read_csv("data/synthetic-enrollment.csv")
    demographics = pl.read_csv("data/synthetic-patient-demographics.csv")
    return trials, sites, enrollment, demographics


def load_model():
    """Load trained enrollment prediction model artifacts."""
    model = joblib.load("ml/enrollment_model.joblib")
    scaler = joblib.load("ml/scaler.joblib")
    encoders = joblib.load("ml/encoders.joblib")
    return model, scaler, encoders


trials, sites, enrollment, demographics = load_data()
model, scaler, encoders = load_model()

# Pre-compute summary statistics used in the Overview KPI cards
total_trials = len(trials)
total_sites = len(sites)
total_patients = len(demographics)

site_enroll_base = (
    enrollment.group_by("site_id")
    .agg(
        pl.col("cumulative_enrolled").max().alias("total_enrolled"),
        pl.col("month").max().alias("months_active"),
    )
    .join(sites, on="site_id")
    .with_columns(
        (pl.col("total_enrolled") / pl.col("months_active")).alias("patients_per_month")
    )
)


# ---------------------------------------------------------------------------
# Brand colours (from _brand.yml)
# ---------------------------------------------------------------------------

DEEP_BLUE = "#003D7A"
TEAL = "#00A3A1"
LIGHT_BLUE = "#5CB3CC"
ORANGE = "#F47B20"
GREEN = "#7AB800"
GRAY = "#6D6E71"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

app_ui = ui.page_navbar(
    # ── Overview ──────────────────────────────────────────────────────────
    ui.nav_panel(
        "Overview",
        ui.layout_column_wrap(
            ui.value_box(
                title="Active Trials",
                value=str(total_trials),
                showcase=ui.tags.i(class_="bi bi-clipboard2-pulse",
                                   style="font-size:1.8rem;"),
                theme="primary",
            ),
            ui.value_box(
                title="Clinical Sites",
                value=f"{total_sites:,}",
                showcase=ui.tags.i(class_="bi bi-hospital",
                                   style="font-size:1.8rem;"),
                theme="info",
            ),
            ui.value_box(
                title="Enrolled Patients",
                value=f"{total_patients:,}",
                showcase=ui.tags.i(class_="bi bi-people",
                                   style="font-size:1.8rem;"),
                theme="success",
            ),
            width=1 / 3,
            fill=False,
        ),
        ui.layout_column_wrap(
            ui.card(
                ui.card_header("Trials by Phase"),
                ui.output_plot("phase_plot"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header("Trials by Therapeutic Area"),
                ui.output_plot("therapeutic_area_plot"),
                full_screen=True,
            ),
            width=1 / 2,
        ),
        ui.card(
            ui.card_header("Monthly Enrollment Trend"),
            ui.output_plot("enrollment_trend_plot"),
            full_screen=True,
        ),
    ),
    # ── Site Performance ─────────────────────────────────────────────────
    ui.nav_panel(
        "Site Performance",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Filters"),
                ui.input_select(
                    "country_filter",
                    "Country:",
                    choices=["All"] + sorted(sites["country"].unique().to_list()),
                ),
                ui.input_select(
                    "site_type_filter",
                    "Site Type:",
                    choices=["All"] + sorted(sites["site_type"].unique().to_list()),
                ),
            ),
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("Enrollment Rate by Site Type"),
                    ui.output_plot("site_performance_plot"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header("Performance by Country"),
                    ui.output_plot("country_performance_plot"),
                    full_screen=True,
                ),
                width=1 / 2,
            ),
            ui.card(
                ui.card_header("Top 10 Sites"),
                ui.output_table("top_sites_table"),
            ),
        ),
    ),
    # ── Patient Demographics ──────────────────────────────────────────────
    ui.nav_panel(
        "Demographics",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Filters"),
                ui.input_select(
                    "demo_trial_filter",
                    "Trial:",
                    choices=["All"] + sorted(
                        demographics["trial_id"].unique().to_list()
                    ),
                ),
                ui.input_select(
                    "demo_gender_filter",
                    "Gender:",
                    choices=["All"] + sorted(
                        demographics["gender"].unique().to_list()
                    ),
                ),
            ),
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("Age Distribution"),
                    ui.output_plot("age_distribution_plot"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header("Gender Breakdown"),
                    ui.output_plot("gender_plot"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header("Ethnicity Distribution"),
                    ui.output_plot("ethnicity_plot"),
                    full_screen=True,
                ),
                width=1 / 3,
            ),
            ui.card(
                ui.card_header("BMI Distribution by Gender"),
                ui.output_plot("bmi_plot"),
                full_screen=True,
            ),
        ),
    ),
    # ── Enrollment Predictor ─────────────────────────────────────────────
    ui.nav_panel(
        "Predictor",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Site & Trial Characteristics"),
                ui.input_select(
                    "pred_phase",
                    "Trial Phase:",
                    choices=sorted(trials["phase"].unique().to_list()),
                ),
                ui.input_select(
                    "pred_therapeutic_area",
                    "Therapeutic Area:",
                    choices=sorted(trials["therapeutic_area"].unique().to_list()),
                ),
                ui.input_select(
                    "pred_country",
                    "Country:",
                    choices=sorted(sites["country"].unique().to_list()),
                ),
                ui.input_select(
                    "pred_site_type",
                    "Site Type:",
                    choices=sorted(sites["site_type"].unique().to_list()),
                ),
                ui.input_select(
                    "pred_population_density",
                    "Population Density:",
                    choices=["Urban", "Suburban", "Rural"],
                ),
                ui.input_slider(
                    "pred_investigator_exp",
                    "Investigator Experience (yrs):",
                    min=1, max=25, value=10,
                ),
                ui.input_slider(
                    "pred_staff_count",
                    "Site Staff Count:",
                    min=3, max=30, value=15,
                ),
                ui.input_slider(
                    "pred_prior_trials",
                    "Prior Trials Completed:",
                    min=0, max=50, value=10,
                ),
                ui.input_numeric(
                    "pred_database_size",
                    "Patient Database Size:",
                    value=10000, min=500, max=50000,
                ),
                ui.input_slider(
                    "pred_distance",
                    "Distance to Metro (km):",
                    min=0, max=200, value=20,
                ),
                ui.input_slider(
                    "pred_inclusion",
                    "Inclusion Criteria Count:",
                    min=5, max=20, value=10,
                ),
                ui.input_slider(
                    "pred_exclusion",
                    "Exclusion Criteria Count:",
                    min=3, max=15, value=8,
                ),
                ui.input_slider(
                    "pred_amendments",
                    "Protocol Amendments:",
                    min=0, max=5, value=1,
                ),
                ui.input_numeric(
                    "pred_target",
                    "Target Enrollment per Site:",
                    value=50, min=1, max=500,
                ),
                ui.input_action_button(
                    "predict_btn",
                    "Predict Enrollment Success",
                    class_="btn-primary w-100 mt-2",
                ),
            ),
            ui.card(
                ui.card_header("Prediction Result"),
                ui.output_ui("prediction_result"),
            ),
            ui.card(
                ui.card_header("About the Model"),
                ui.markdown(
                    """
Predicts the probability that a site will meet its enrollment target
based on site and trial characteristics.

**Key factors:** investigator experience, staff count, patient
database size, trial phase, and target enrollment.

*Model trained on synthetic data for demonstration purposes only.*
                    """
                ),
            ),
        ),
    ),
    title=ui.tags.span(
        ui.tags.strong("VitalGen", style=f"color:{DEEP_BLUE};"),
        " Clinical Trial Dashboard",
    ),
    theme=ui.Theme.from_brand(__file__),
    fillable=True,
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def server(input, output, session):

    # ── Filtered helpers ──────────────────────────────────────────────────

    @reactive.calc
    def filtered_demo():
        df = demographics
        if input.demo_trial_filter() != "All":
            df = df.filter(pl.col("trial_id") == input.demo_trial_filter())
        if input.demo_gender_filter() != "All":
            df = df.filter(pl.col("gender") == input.demo_gender_filter())
        return df

    @reactive.calc
    def filtered_site_enroll():
        df = site_enroll_base
        if input.country_filter() != "All":
            df = df.filter(pl.col("country") == input.country_filter())
        if input.site_type_filter() != "All":
            df = df.filter(pl.col("site_type") == input.site_type_filter())
        return df

    # ── Overview ──────────────────────────────────────────────────────────

    @output
    @render.plot
    def phase_plot():
        phase_counts = (
            trials.group_by("phase")
            .agg(pl.len().alias("count"))
            .sort("phase")
            .to_pandas()
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(phase_counts["phase"], phase_counts["count"],
               color=[DEEP_BLUE, TEAL, LIGHT_BLUE])
        ax.set_xlabel("Phase")
        ax.set_ylabel("Number of Trials")
        ax.set_title("Active Trials by Phase")
        fig.tight_layout()
        return fig

    @output
    @render.plot
    def therapeutic_area_plot():
        ta_counts = (
            trials.group_by("therapeutic_area")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .to_pandas()
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(ta_counts["therapeutic_area"], ta_counts["count"], color=TEAL)
        ax.set_xlabel("Number of Trials")
        ax.set_title("Trials by Therapeutic Area")
        fig.tight_layout()
        return fig

    @output
    @render.plot
    def enrollment_trend_plot():
        trend = (
            enrollment.group_by("enrollment_date")
            .agg(pl.col("monthly_enrolled").sum().alias("total_enrolled"))
            .sort("enrollment_date")
            .to_pandas()
        )
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(trend["enrollment_date"], trend["total_enrolled"],
                color=DEEP_BLUE, linewidth=1.5)
        ax.fill_between(trend["enrollment_date"], trend["total_enrolled"],
                        alpha=0.15, color=DEEP_BLUE)
        ax.set_xlabel("Date")
        ax.set_ylabel("Patients Enrolled")
        ax.set_title("Monthly Enrollment Across All Trials")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        return fig

    # ── Site Performance ──────────────────────────────────────────────────

    @output
    @render.plot
    def site_performance_plot():
        perf = (
            filtered_site_enroll()
            .group_by("site_type")
            .agg(pl.col("patients_per_month").mean().alias("avg_rate"))
            .sort("avg_rate", descending=True)
            .to_pandas()
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(perf["site_type"], perf["avg_rate"], color=DEEP_BLUE)
        ax.set_ylabel("Avg Patients / Month")
        ax.set_title("Average Enrollment Rate by Site Type")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        return fig

    @output
    @render.plot
    def country_performance_plot():
        perf = (
            filtered_site_enroll()
            .group_by("country")
            .agg(pl.col("patients_per_month").mean().alias("avg_rate"))
            .sort("avg_rate", descending=True)
            .to_pandas()
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(perf["country"], perf["avg_rate"], color=TEAL)
        ax.set_xlabel("Avg Patients / Month")
        ax.set_title("Average Enrollment Rate by Country")
        fig.tight_layout()
        return fig

    @output
    @render.table
    def top_sites_table():
        top = (
            filtered_site_enroll()
            .sort("patients_per_month", descending=True)
            .head(10)
            .select("site_id", "country", "site_type",
                    "total_enrolled", "patients_per_month")
            .with_columns(pl.col("patients_per_month").round(2))
            .to_pandas()
        )
        top.columns = [
            "Site ID", "Country", "Site Type",
            "Total Enrolled", "Rate (pts/mo)",
        ]
        return top

    # ── Demographics ──────────────────────────────────────────────────────

    @output
    @render.plot
    def age_distribution_plot():
        ages = filtered_demo().get_column("age").to_list()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(ages, bins=20, color=LIGHT_BLUE, edgecolor="white")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.set_title("Patient Age Distribution")
        fig.tight_layout()
        return fig

    @output
    @render.plot
    def gender_plot():
        counts = (
            filtered_demo()
            .group_by("gender")
            .agg(pl.len().alias("count"))
            .to_pandas()
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            counts["count"],
            labels=counts["gender"],
            autopct="%1.1f%%",
            colors=[DEEP_BLUE, TEAL, LIGHT_BLUE],
            startangle=90,
        )
        ax.set_title("Gender Breakdown")
        fig.tight_layout()
        return fig

    @output
    @render.plot
    def ethnicity_plot():
        counts = (
            filtered_demo()
            .group_by("ethnicity")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .to_pandas()
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(counts["ethnicity"], counts["count"], color=ORANGE)
        ax.set_xlabel("Count")
        ax.set_title("Ethnicity Distribution")
        fig.tight_layout()
        return fig

    @output
    @render.plot
    def bmi_plot():
        df_pd = filtered_demo().select("bmi", "gender").to_pandas()
        genders = df_pd["gender"].unique()
        colors_map = {g: c for g, c in zip(genders, [DEEP_BLUE, TEAL, LIGHT_BLUE])}
        fig, ax = plt.subplots(figsize=(10, 4))
        for gender, grp in df_pd.groupby("gender"):
            ax.hist(grp["bmi"], bins=25, alpha=0.6,
                    label=gender, color=colors_map.get(gender, GRAY))
        ax.set_xlabel("BMI")
        ax.set_ylabel("Count")
        ax.set_title("BMI Distribution by Gender")
        ax.legend()
        fig.tight_layout()
        return fig

    # ── Predictor ─────────────────────────────────────────────────────────

    @output
    @render.ui
    @reactive.event(input.predict_btn)
    def prediction_result():
        features = [
            encoders["phase"].transform([input.pred_phase()])[0],
            encoders["therapeutic_area"].transform(
                [input.pred_therapeutic_area()]
            )[0],
            encoders["country"].transform([input.pred_country()])[0],
            encoders["site_type"].transform([input.pred_site_type()])[0],
            encoders["population_density"].transform(
                [input.pred_population_density()]
            )[0],
            input.pred_investigator_exp(),
            input.pred_staff_count(),
            input.pred_prior_trials(),
            input.pred_database_size(),
            input.pred_distance(),
            input.pred_inclusion(),
            input.pred_exclusion(),
            input.pred_amendments(),
            input.pred_target(),
        ]

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0, 1]
        prediction = "Success" if prob >= 0.5 else "At Risk"
        color = GREEN if prob >= 0.7 else ORANGE if prob >= 0.4 else "#D32F2F"
        interp = (
            "High likelihood of meeting enrollment target."
            if prob >= 0.7
            else "Moderate risk — consider optimisation strategies."
            if prob >= 0.4
            else "High risk — significant intervention likely needed."
        )

        return ui.div(
            ui.h3(f"Prediction: {prediction}",
                  style=f"color:{color}; font-weight:600;"),
            ui.h2(
                f"{prob * 100:.1f}%",
                style=f"color:{color}; font-size:3rem; margin:0;",
            ),
            ui.p(
                "Probability of meeting enrollment target",
                style="color:#666; margin-bottom:1rem;",
            ),
            ui.hr(),
            ui.p(interp),
        )


app = App(app_ui, server)
