"""
Utility functions for clinical trial enrollment prediction model.

This module contains helper functions for model training, evaluation, and deployment.
All models use synthetic data for demonstration purposes only.
"""

import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path


def load_and_prepare_data():
    """
    Load synthetic clinical trial data and prepare features for modeling.

    Returns:
        tuple: (X, y, feature_names, encoders) for model training
    """
    # Get project root directory (parent of ml directory if we're in ml)
    import os
    current_dir = Path.cwd()
    if current_dir.name == "ml":
        project_root = current_dir.parent
    else:
        project_root = current_dir

    # Check if data exists, generate if not
    data_path = project_root / "data" / "synthetic-trials.csv"
    if not data_path.exists():
        print("Generating synthetic data...")
        import subprocess
        subprocess.run(["python", str(project_root / "data" / "generate_data.py")], check=True, cwd=str(project_root))

    # Load data
    trials = pl.read_csv(str(project_root / "data" / "synthetic-trials.csv"))
    sites = pl.read_csv(str(project_root / "data" / "synthetic-sites.csv"))
    enrollment = pl.read_csv(str(project_root / "data" / "synthetic-enrollment.csv"))

    # Calculate actual enrollment per site
    site_enrollment = (
        enrollment
        .group_by(["trial_id", "site_id"])
        .agg([
            pl.col("cumulative_enrolled").max().alias("actual_enrolled"),
            pl.col("target_per_site").first().alias("target_per_site"),
            pl.col("month").max().alias("months_active")
        ])
    )

    # Join with site and trial characteristics
    features = (
        site_enrollment
        .join(sites, on=["trial_id", "site_id"])
        .join(trials.select([
            "trial_id", "phase", "therapeutic_area",
            "inclusion_criteria_count", "exclusion_criteria_count",
            "protocol_amendments"
        ]), on="trial_id")
    )

    # Calculate target variable: enrollment success rate
    # 1 if site met or exceeded target, 0 otherwise
    features = features.with_columns([
        (pl.col("actual_enrolled") >= pl.col("target_per_site")).cast(pl.Int32).alias("enrollment_success"),
        (pl.col("actual_enrolled") / pl.col("target_per_site") * 100).alias("enrollment_pct"),
        (pl.col("actual_enrolled") / pl.col("months_active")).alias("enrollment_rate")
    ])

    # Convert to pandas for sklearn
    df = features.to_pandas()

    # Encode categorical variables
    encoders = {}
    categorical_cols = ["phase", "therapeutic_area", "country", "site_type", "population_density"]

    for col in categorical_cols:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col])
        encoders[col] = le

    # Select features for modeling
    feature_cols = [
        "phase_encoded",
        "therapeutic_area_encoded",
        "country_encoded",
        "site_type_encoded",
        "population_density_encoded",
        "investigator_experience_years",
        "site_staff_count",
        "prior_trials_completed",
        "patient_database_size",
        "distance_to_metro_km",
        "inclusion_criteria_count",
        "exclusion_criteria_count",
        "protocol_amendments",
        "target_per_site"
    ]

    X = df[feature_cols].values
    y = df["enrollment_success"].values

    return X, y, feature_cols, encoders, df


def prepare_site_features(site_data, encoders):
    """
    Prepare features for a new site prediction.

    Args:
        site_data (dict): Dictionary containing site characteristics
        encoders (dict): Label encoders for categorical variables

    Returns:
        np.array: Feature vector for prediction
    """
    # Encode categorical variables
    features = [
        encoders["phase"].transform([site_data["phase"]])[0],
        encoders["therapeutic_area"].transform([site_data["therapeutic_area"]])[0],
        encoders["country"].transform([site_data["country"]])[0],
        encoders["site_type"].transform([site_data["site_type"]])[0],
        encoders["population_density"].transform([site_data["population_density"]])[0],
        site_data["investigator_experience_years"],
        site_data["site_staff_count"],
        site_data["prior_trials_completed"],
        site_data["patient_database_size"],
        site_data["distance_to_metro_km"],
        site_data["inclusion_criteria_count"],
        site_data["exclusion_criteria_count"],
        site_data["protocol_amendments"],
        site_data["target_per_site"]
    ]

    return np.array(features).reshape(1, -1)


def get_feature_importance_names():
    """
    Get human-readable feature names for interpretation.

    Returns:
        list: Feature names
    """
    return [
        "Trial Phase",
        "Therapeutic Area",
        "Country",
        "Site Type",
        "Population Density",
        "Investigator Experience (years)",
        "Site Staff Count",
        "Prior Trials Completed",
        "Patient Database Size",
        "Distance to Metro (km)",
        "Inclusion Criteria Count",
        "Exclusion Criteria Count",
        "Protocol Amendments",
        "Target Enrollment per Site"
    ]
