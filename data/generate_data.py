"""
Generate synthetic clinical trial enrollment data for VitalGen Therapeutics.

This script creates artificial data for demonstration purposes only.
All data is AI-generated and does not represent real clinical trials.
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_trial_data(n_trials=50):
    """Generate synthetic clinical trial information."""

    phases = ["Phase I", "Phase II", "Phase III"]
    therapeutic_areas = [
        "Oncology", "Cardiovascular", "Neurology",
        "Immunology", "Metabolic", "Respiratory"
    ]
    indications = {
        "Oncology": ["Lung Cancer", "Breast Cancer", "Melanoma"],
        "Cardiovascular": ["Hypertension", "Heart Failure", "Atrial Fibrillation"],
        "Neurology": ["Alzheimer's", "Parkinson's", "Multiple Sclerosis"],
        "Immunology": ["Rheumatoid Arthritis", "Lupus", "Crohn's Disease"],
        "Metabolic": ["Type 2 Diabetes", "Obesity", "Hyperlipidemia"],
        "Respiratory": ["Asthma", "COPD", "Pulmonary Fibrosis"]
    }

    trial_data = []

    for i in range(n_trials):
        trial_id = f"VGT-{2020+i//10}-{str(i%10).zfill(3)}"
        phase = np.random.choice(phases, p=[0.2, 0.4, 0.4])
        therapeutic_area = np.random.choice(therapeutic_areas)
        indication = np.random.choice(indications[therapeutic_area])

        # Target enrollment varies by phase
        if phase == "Phase I":
            target_enrollment = np.random.randint(20, 80)
        elif phase == "Phase II":
            target_enrollment = np.random.randint(80, 300)
        else:
            target_enrollment = np.random.randint(300, 1500)

        # Number of sites varies by phase
        if phase == "Phase I":
            n_sites = np.random.randint(1, 5)
        elif phase == "Phase II":
            n_sites = np.random.randint(5, 20)
        else:
            n_sites = np.random.randint(20, 80)

        # Trial characteristics
        inclusion_criteria_count = np.random.randint(5, 20)
        exclusion_criteria_count = np.random.randint(3, 15)
        protocol_amendments = np.random.randint(0, 5)

        # Start date in the past 3 years
        days_ago = np.random.randint(30, 1095)
        start_date = datetime.now() - timedelta(days=days_ago)

        # Duration in months
        planned_duration = np.random.randint(12, 48)

        trial_data.append({
            "trial_id": trial_id,
            "phase": phase,
            "therapeutic_area": therapeutic_area,
            "indication": indication,
            "target_enrollment": target_enrollment,
            "n_sites": n_sites,
            "inclusion_criteria_count": inclusion_criteria_count,
            "exclusion_criteria_count": exclusion_criteria_count,
            "protocol_amendments": protocol_amendments,
            "start_date": start_date.date(),
            "planned_duration_months": planned_duration
        })

    return pl.DataFrame(trial_data)


def generate_site_data(trial_df):
    """Generate synthetic site information for each trial."""

    countries = ["USA", "Canada", "UK", "Germany", "France", "Spain", "Australia"]
    site_types = ["Academic Medical Center", "Community Hospital", "Private Practice"]

    site_data = []
    site_counter = 0

    for trial in trial_df.iter_rows(named=True):
        for _ in range(trial["n_sites"]):
            site_counter += 1
            site_id = f"SITE-{str(site_counter).zfill(4)}"

            country = np.random.choice(countries, p=[0.4, 0.1, 0.15, 0.1, 0.1, 0.05, 0.1])
            site_type = np.random.choice(site_types)

            # Site characteristics that affect enrollment
            investigator_experience = np.random.randint(1, 25)  # years
            site_staff_count = np.random.randint(3, 30)
            prior_trials_completed = np.random.randint(0, 50)
            patient_database_size = np.random.randint(500, 50000)

            # Geographic factors
            population_density = np.random.choice(["Urban", "Suburban", "Rural"], p=[0.5, 0.3, 0.2])
            distance_to_metro = np.random.randint(0, 200)  # km

            site_data.append({
                "trial_id": trial["trial_id"],
                "site_id": site_id,
                "country": country,
                "site_type": site_type,
                "investigator_experience_years": investigator_experience,
                "site_staff_count": site_staff_count,
                "prior_trials_completed": prior_trials_completed,
                "patient_database_size": patient_database_size,
                "population_density": population_density,
                "distance_to_metro_km": distance_to_metro
            })

    return pl.DataFrame(site_data)


def generate_enrollment_data(site_df, trial_df):
    """Generate synthetic enrollment time-series data."""

    enrollment_data = []

    # Join to get trial information
    combined = site_df.join(trial_df, on="trial_id")

    for row in combined.iter_rows(named=True):
        trial_start = row["start_date"]

        # Calculate site-specific enrollment rate factors
        experience_factor = min(row["investigator_experience_years"] / 10, 2.0)
        staff_factor = min(row["site_staff_count"] / 10, 2.0)
        database_factor = min(row["patient_database_size"] / 10000, 2.0)

        # Base enrollment rate (patients per month)
        if row["phase"] == "Phase I":
            base_rate = 0.5
        elif row["phase"] == "Phase II":
            base_rate = 1.5
        else:
            base_rate = 3.0

        # Adjust by site characteristics
        site_rate = base_rate * (experience_factor + staff_factor + database_factor) / 3

        # Add randomness
        site_rate *= np.random.uniform(0.5, 1.5)

        # Generate monthly enrollment over trial duration
        target_per_site = row["target_enrollment"] / row["n_sites"]
        months = min(row["planned_duration_months"], 36)  # Cap at 3 years

        cumulative_enrolled = 0

        for month in range(months):
            if cumulative_enrolled >= target_per_site:
                break

            # Enrollment ramp-up in first few months
            ramp_factor = min(month / 3, 1.0)

            # Monthly enrollment with variability
            monthly_enrolled = max(0, int(np.random.poisson(site_rate * ramp_factor)))
            cumulative_enrolled += monthly_enrolled

            # Don't exceed target
            if cumulative_enrolled > target_per_site:
                monthly_enrolled -= (cumulative_enrolled - target_per_site)
                cumulative_enrolled = target_per_site

            enrollment_date = trial_start + timedelta(days=30*month)

            enrollment_data.append({
                "trial_id": row["trial_id"],
                "site_id": row["site_id"],
                "month": month + 1,
                "enrollment_date": enrollment_date,
                "monthly_enrolled": monthly_enrolled,
                "cumulative_enrolled": cumulative_enrolled,
                "target_per_site": int(target_per_site)
            })

    return pl.DataFrame(enrollment_data)


def generate_patient_demographics(enrollment_df):
    """Generate synthetic patient demographic data."""

    demographics_data = []
    patient_counter = 0

    for row in enrollment_df.iter_rows(named=True):
        for _ in range(int(row["monthly_enrolled"])):
            patient_counter += 1
            patient_id = f"PT-{str(patient_counter).zfill(6)}"

            # Demographics
            age = max(18, min(85, int(np.random.normal(55, 15))))
            gender = np.random.choice(["Male", "Female"], p=[0.48, 0.52])
            ethnicity = np.random.choice(
                ["Caucasian", "African American", "Hispanic", "Asian", "Other"],
                p=[0.6, 0.13, 0.18, 0.06, 0.03]
            )

            # Clinical characteristics
            bmi = max(15, min(45, np.random.normal(27, 5)))

            demographics_data.append({
                "patient_id": patient_id,
                "trial_id": row["trial_id"],
                "site_id": row["site_id"],
                "enrollment_date": row["enrollment_date"],
                "age": age,
                "gender": gender,
                "ethnicity": ethnicity,
                "bmi": round(bmi, 1)
            })

    return pl.DataFrame(demographics_data)


def main():
    """Generate all synthetic datasets."""

    print("Generating synthetic clinical trial enrollment data...")
    print("=" * 60)

    # Generate data
    print("\n1. Generating trial data...")
    trial_df = generate_trial_data(n_trials=50)
    print(f"   Created {len(trial_df)} trials")

    print("\n2. Generating site data...")
    site_df = generate_site_data(trial_df)
    print(f"   Created {len(site_df)} sites")

    print("\n3. Generating enrollment time-series data...")
    enrollment_df = generate_enrollment_data(site_df, trial_df)
    print(f"   Created {len(enrollment_df)} enrollment records")

    print("\n4. Generating patient demographics...")
    demographics_df = generate_patient_demographics(enrollment_df)
    print(f"   Created {len(demographics_df)} patient records")

    # Save to CSV with synthetic prefix
    print("\n5. Saving data files...")

    # Add header comments to indicate synthetic data
    trial_df.write_csv("data/synthetic-trials.csv")
    print("   ✓ data/synthetic-trials.csv")

    site_df.write_csv("data/synthetic-sites.csv")
    print("   ✓ data/synthetic-sites.csv")

    enrollment_df.write_csv("data/synthetic-enrollment.csv")
    print("   ✓ data/synthetic-enrollment.csv")

    demographics_df.write_csv("data/synthetic-patient-demographics.csv")
    print("   ✓ data/synthetic-patient-demographics.csv")

    # Create summary statistics
    summary_stats = {
        "Total Trials": len(trial_df),
        "Total Sites": len(site_df),
        "Total Patients": len(demographics_df),
        "Phase I Trials": len(trial_df.filter(pl.col("phase") == "Phase I")),
        "Phase II Trials": len(trial_df.filter(pl.col("phase") == "Phase II")),
        "Phase III Trials": len(trial_df.filter(pl.col("phase") == "Phase III")),
    }

    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    for key, value in summary_stats.items():
        print(f"{key:.<40} {value:>10}")

    print("\n✓ Data generation complete!")
    print("\nNOTE: All data is synthetic and AI-generated for demonstration purposes only.")


if __name__ == "__main__":
    main()
