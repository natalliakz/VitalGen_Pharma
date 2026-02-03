"""
Clinical Trial Enrollment Prediction API

FastAPI web service for predicting clinical trial site enrollment success.

This project contains synthetic data and analysis created for demonstration purposes only.

Endpoints:
- GET /health - API health check
- GET /data - Sample clinical trial data
- POST /predict - Predict enrollment success for a site
- GET /model-info - Model metadata and performance metrics
- GET /docs - Interactive API documentation
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import polars as pl
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

# Initialize FastAPI app
app = FastAPI(
    title="VitalGen Clinical Trial Enrollment API",
    description="API for predicting clinical trial site enrollment success. This project contains synthetic data and analysis created for demonstration purposes only.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Load model artifacts
try:
    model = joblib.load("ml/enrollment_model.joblib")
    scaler = joblib.load("ml/scaler.joblib")
    encoders = joblib.load("ml/encoders.joblib")

    with open("ml/model_metadata.json", "r") as f:
        model_metadata = json.load(f)
except Exception as e:
    print(f"Warning: Could not load model artifacts: {e}")
    model = None
    scaler = None
    encoders = None
    model_metadata = {}


# Pydantic models for request/response validation
class SiteCharacteristics(BaseModel):
    """Site and trial characteristics for enrollment prediction."""

    phase: str = Field(..., description="Trial phase (Phase I, Phase II, Phase III)")
    therapeutic_area: str = Field(..., description="Therapeutic area")
    country: str = Field(..., description="Country where site is located")
    site_type: str = Field(..., description="Type of site (Academic Medical Center, Community Hospital, Private Practice)")
    population_density: str = Field(..., description="Population density (Urban, Suburban, Rural)")
    investigator_experience_years: int = Field(..., ge=1, le=50, description="Years of investigator experience")
    site_staff_count: int = Field(..., ge=1, le=100, description="Number of staff at site")
    prior_trials_completed: int = Field(..., ge=0, le=200, description="Number of prior trials completed")
    patient_database_size: int = Field(..., ge=100, le=100000, description="Size of patient database")
    distance_to_metro_km: int = Field(..., ge=0, le=500, description="Distance to nearest metro area in km")
    inclusion_criteria_count: int = Field(..., ge=1, le=50, description="Number of inclusion criteria")
    exclusion_criteria_count: int = Field(..., ge=1, le=50, description="Number of exclusion criteria")
    protocol_amendments: int = Field(..., ge=0, le=20, description="Number of protocol amendments")
    target_per_site: int = Field(..., ge=1, le=1000, description="Target enrollment per site")

    class Config:
        json_schema_extra = {
            "example": {
                "phase": "Phase II",
                "therapeutic_area": "Oncology",
                "country": "USA",
                "site_type": "Academic Medical Center",
                "population_density": "Urban",
                "investigator_experience_years": 15,
                "site_staff_count": 20,
                "prior_trials_completed": 25,
                "patient_database_size": 15000,
                "distance_to_metro_km": 10,
                "inclusion_criteria_count": 12,
                "exclusion_criteria_count": 8,
                "protocol_amendments": 2,
                "target_per_site": 75
            }
        }


class PredictionResponse(BaseModel):
    """Response model for enrollment success prediction."""

    success_probability: float = Field(..., description="Probability of meeting enrollment target (0-1)")
    prediction: str = Field(..., description="Prediction label (Success or At Risk)")
    risk_level: str = Field(..., description="Risk level (Low, Moderate, High)")
    recommendation: str = Field(..., description="Recommendation based on prediction")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str


# API Endpoints

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "VitalGen Clinical Trial Enrollment Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "data": "/data",
            "predict": "/predict",
            "model_info": "/model-info",
            "docs": "/docs"
        },
        "disclaimer": "This project contains synthetic data and analysis created for demonstration purposes only."
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint to verify API is running and model is loaded.

    Returns:
        HealthResponse: API health status
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.get("/data", tags=["Data"])
async def get_sample_data(limit: int = 10):
    """
    Get sample clinical trial data.

    Args:
        limit: Number of records to return (default: 10, max: 100)

    Returns:
        JSON with sample trial data
    """
    # Check if data exists, generate if not
    data_path = Path("data/synthetic-trials.csv")
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Data not found. Please run data generation script.")

    # Limit to reasonable size
    limit = min(limit, 100)

    try:
        trials = pl.read_csv("data/synthetic-trials.csv").head(limit)

        return {
            "count": len(trials),
            "data": trials.to_dicts(),
            "note": "All data is synthetic and AI-generated for demonstration purposes only."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_enrollment(site: SiteCharacteristics):
    """
    Predict the probability of successful enrollment for a clinical trial site.

    Args:
        site: Site and trial characteristics

    Returns:
        PredictionResponse: Enrollment success prediction
    """
    if model is None or scaler is None or encoders is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model artifacts exist in ml/ directory."
        )

    try:
        # Validate categorical values
        valid_phases = list(encoders["phase"].classes_)
        valid_tas = list(encoders["therapeutic_area"].classes_)
        valid_countries = list(encoders["country"].classes_)
        valid_site_types = list(encoders["site_type"].classes_)
        valid_densities = list(encoders["population_density"].classes_)

        if site.phase not in valid_phases:
            raise HTTPException(status_code=400, detail=f"Invalid phase. Must be one of: {valid_phases}")
        if site.therapeutic_area not in valid_tas:
            raise HTTPException(status_code=400, detail=f"Invalid therapeutic area. Must be one of: {valid_tas}")
        if site.country not in valid_countries:
            raise HTTPException(status_code=400, detail=f"Invalid country. Must be one of: {valid_countries}")
        if site.site_type not in valid_site_types:
            raise HTTPException(status_code=400, detail=f"Invalid site type. Must be one of: {valid_site_types}")
        if site.population_density not in valid_densities:
            raise HTTPException(status_code=400, detail=f"Invalid population density. Must be one of: {valid_densities}")

        # Prepare features
        features = [
            encoders["phase"].transform([site.phase])[0],
            encoders["therapeutic_area"].transform([site.therapeutic_area])[0],
            encoders["country"].transform([site.country])[0],
            encoders["site_type"].transform([site.site_type])[0],
            encoders["population_density"].transform([site.population_density])[0],
            site.investigator_experience_years,
            site.site_staff_count,
            site.prior_trials_completed,
            site.patient_database_size,
            site.distance_to_metro_km,
            site.inclusion_criteria_count,
            site.exclusion_criteria_count,
            site.protocol_amendments,
            site.target_per_site
        ]

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        # Make prediction
        probability = float(model.predict_proba(X_scaled)[0, 1])
        prediction = "Success" if probability >= 0.5 else "At Risk"

        # Determine risk level and recommendation
        if probability >= 0.7:
            risk_level = "Low"
            recommendation = "This site has excellent characteristics for meeting enrollment targets. Proceed with confidence."
        elif probability >= 0.4:
            risk_level = "Moderate"
            recommendation = "This site has reasonable potential but may benefit from additional support or optimization strategies."
        else:
            risk_level = "High"
            recommendation = "This site faces significant challenges in meeting enrollment targets. Consider additional sites, protocol modifications, or enhanced support."

        return PredictionResponse(
            success_probability=probability,
            prediction=prediction,
            risk_level=risk_level,
            recommendation=recommendation
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model-info", tags=["Model"])
async def get_model_info():
    """
    Get information about the trained model including performance metrics.

    Returns:
        JSON with model metadata
    """
    if not model_metadata:
        raise HTTPException(status_code=404, detail="Model metadata not found.")

    return {
        "model_info": model_metadata,
        "note": "Model trained on synthetic data for demonstration purposes only."
    }


@app.get("/valid-values", tags=["Model"])
async def get_valid_values():
    """
    Get valid categorical values for prediction requests.

    Returns:
        JSON with valid values for each categorical field
    """
    if encoders is None:
        raise HTTPException(status_code=503, detail="Encoders not loaded.")

    return {
        "phases": list(encoders["phase"].classes_),
        "therapeutic_areas": list(encoders["therapeutic_area"].classes_),
        "countries": list(encoders["country"].classes_),
        "site_types": list(encoders["site_type"].classes_),
        "population_densities": list(encoders["population_density"].classes_)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
