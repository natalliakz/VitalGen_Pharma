# VitalGen Therapeutics - Clinical Trial Enrollment Analytics

A comprehensive demonstration project showcasing data analytics, machine learning, and interactive applications for clinical trial enrollment optimization.

## Overview

This project demonstrates advanced analytics capabilities for clinical trial enrollment using Python (with polars for ETL), R examples, machine learning models, interactive dashboards (Shiny and Streamlit), and REST APIs. The project is designed to showcase how modern data science tools can optimize clinical trial site selection and enrollment strategies.

**Key Features:**
- **Python ETL with Polars**: Fast, modern data manipulation (alternative to R's tidyverse/dplyr)
- **Exploratory Data Analysis**: Comprehensive Quarto report with visualizations and insights
- **Machine Learning**: Predictive models for enrollment success
- **Interactive Dashboards**: Both Shiny for Python and Streamlit applications
- **REST API**: FastAPI endpoints for model predictions
- **Reproducible**: Fully documented and version-controlled with environment management

## Project Structure

```
VitalGen_Pharmaceuticals/
├── data/
│   ├── generate_data.py              # Synthetic data generation script
│   ├── synthetic-trials.csv          # Trial information (generated)
│   ├── synthetic-sites.csv           # Site characteristics (generated)
│   ├── synthetic-enrollment.csv      # Enrollment time-series (generated)
│   └── synthetic-patient-demographics.csv  # Patient demographics (generated)
├── ml/
│   ├── model_utils.py                # Model utility functions
│   ├── train_model.py                # Model training script
│   ├── enrollment_model.joblib       # Trained model (generated)
│   ├── scaler.joblib                 # Feature scaler (generated)
│   ├── encoders.joblib               # Categorical encoders (generated)
│   └── model_metadata.json           # Model performance metrics (generated)
├── eda.qmd                           # Exploratory data analysis (Quarto)
├── eda.html                          # Rendered analysis report
├── app.py                            # Shiny for Python dashboard
├── streamlit_app.py                  # Streamlit dashboard
├── api.py                            # FastAPI web service
├── _brand.yml                        # VitalGen branding configuration
├── pyproject.toml                    # Python dependencies
├── uv.lock                           # Locked dependency versions
└── README.md                         # This file
```

## Installation and Setup

### Prerequisites

- **Python 3.13+** (managed by `uv`)
- **Quarto** for rendering analysis reports
- **Git** for version control

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd VitalGen_Pharmaceuticals
```

### Step 2: Install Dependencies

This project uses `uv` for fast, reliable Python dependency management:

```bash
# Install dependencies (uv will create a virtual environment automatically)
uv sync
```

### Step 3: Generate Synthetic Data

All data files are generated programmatically and are not checked into version control:

```bash
uv run python data/generate_data.py
```

This creates:
- 50 clinical trials across 6 therapeutic areas
- 1,250 sites in 7 countries
- ~21,000 patient enrollment records
- Demographic information

### Step 4: Train Machine Learning Model

```bash
uv run python ml/train_model.py
```

This trains a Random Forest classifier to predict enrollment success and saves model artifacts.

## Usage

### Exploratory Data Analysis

Render the comprehensive EDA report:

```bash
uv run quarto render eda.qmd
```

Open `eda.html` in your browser to view the analysis.

**Key Insights in the Report:**
- Portfolio overview by phase and therapeutic area
- Site performance analysis by country and type
- Investigator experience impact on enrollment
- Patient demographics and diversity metrics
- Enrollment timeline trends

### Interactive Dashboards

#### Shiny for Python Dashboard

Launch the Shiny application:

```bash
uv run shiny run app.py --port 8501
```

Navigate to `http://localhost:8501` in your browser.

**Features:**
- Portfolio overview with key metrics
- Site performance analytics with filters
- ML-powered enrollment predictor

#### Streamlit Dashboard

Launch the Streamlit application:

```bash
uv run streamlit run streamlit_app.py
```

The app will open automatically in your browser.

**Features:**
- Interactive portfolio visualization
- Site performance comparison
- Real-time enrollment predictions

**Comparison:** Both dashboards provide similar functionality, demonstrating Shiny (Posit-integrated) vs. Streamlit (popular alternative framework).

### REST API

Start the FastAPI server:

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

Navigate to `http://localhost:8000/docs` for interactive API documentation.

**Available Endpoints:**

- `GET /health` - API health check
- `GET /data?limit=10` - Sample clinical trial data
- `POST /predict` - Predict enrollment success probability
- `GET /model-info` - Model metadata and performance
- `GET /valid-values` - Valid categorical values for predictions

**Example API Request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response:**

```json
{
  "success_probability": 0.939,
  "prediction": "Success",
  "risk_level": "Low",
  "recommendation": "This site has excellent characteristics for meeting enrollment targets."
}
```

## Python ETL with Polars

This project demonstrates **polars**, a modern, fast alternative to pandas and R's tidyverse/dplyr.

### Why Polars?

- **10-100x faster** than pandas on large datasets
- **Similar API** to tidyverse/dplyr
- **Lazy evaluation** for query optimization
- **Better memory efficiency**

### Polars vs. dplyr Comparison

| Operation | dplyr (R) | polars (Python) |
|-----------|-----------|-----------------|
| Filter rows | `filter(age > 18)` | `.filter(pl.col("age") > 18)` |
| Select columns | `select(name, age)` | `.select(["name", "age"])` |
| Group & summarize | `group_by(country) %>% summarize(avg = mean(value))` | `.group_by("country").agg(pl.col("value").mean())` |
| Join | `left_join(df1, df2, by="id")` | `.join(df2, on="id", how="left")` |
| Create column | `mutate(new_col = old_col * 2)` | `.with_columns((pl.col("old_col") * 2).alias("new_col"))` |

### Example ETL Workflow

```python
import polars as pl

# Read data
trials = pl.read_csv("data/synthetic-trials.csv")

# Filter, group, and aggregate
results = (
    trials
    .filter(pl.col("phase") == "Phase III")
    .group_by("therapeutic_area")
    .agg([
        pl.count("trial_id").alias("n_trials"),
        pl.col("target_enrollment").mean().alias("avg_target")
    ])
    .sort("n_trials", descending=True)
)
```

## Custom Branding with `_brand.yml`

The project uses `_brand.yml` for consistent branding across Quarto documents and Shiny applications.

**To customize for your organization:**

1. Edit `_brand.yml` with your brand colors, fonts, and logos
2. Quarto documents automatically apply the theme
3. Shiny apps use: `theme=ui.Theme.from_brand(__file__)`
4. Streamlit apps can be customized via CSS

## Key Technical Demonstrations

### 1. Data Engineering
- Synthetic data generation with realistic relationships
- Efficient data manipulation with polars
- Time-series aggregations and transformations

### 2. Machine Learning
- Feature engineering from site and trial characteristics
- Random Forest classification for enrollment prediction
- Model evaluation with proper train/validation/test splits
- Feature importance analysis

### 3. Visualization
- Portfolio analytics dashboards
- Site performance comparisons
- Time-series enrollment trends
- Interactive filtering and exploration

### 4. Application Development
- **Shiny for Python**: Reactive web applications with Posit integration
- **Streamlit**: Rapid prototyping and deployment
- **FastAPI**: Production-ready REST APIs with automatic documentation

### 5. Reproducibility
- Environment management with `uv`
- Version-locked dependencies (`uv.lock`)
- Programmatic data generation
- Comprehensive documentation

## Use Cases

This project can be adapted for:

1. **Site Selection**: Identify high-performing sites for new trials
2. **Enrollment Forecasting**: Predict timeline to enrollment targets
3. **Resource Optimization**: Allocate support to at-risk sites
4. **Portfolio Planning**: Optimize trial design based on historical data
5. **Feasibility Assessment**: Evaluate site capacity for new protocols

## Technical Stack

- **Python 3.13** - Core language
- **polars** - High-performance data manipulation
- **scikit-learn** - Machine learning
- **FastAPI** - Web API framework
- **Shiny for Python** - Interactive dashboards (Posit)
- **Streamlit** - Interactive dashboards (alternative)
- **Quarto** - Reproducible reporting
- **matplotlib/seaborn** - Visualization
- **uv** - Fast dependency management

## Next Steps

### For Data Scientists
- Explore the EDA report to understand enrollment patterns
- Experiment with different ML models in `ml/train_model.py`
- Extend the feature set with additional trial characteristics

### For Engineers
- Deploy the API to a production environment
- Integrate with existing trial management systems
- Build automated reporting pipelines

### For Product Teams
- Test the interactive dashboards with stakeholders
- Gather feedback on key metrics and visualizations
- Identify additional use cases for enrollment analytics

## Support and Questions

For questions about adapting this project to your organization's needs, please contact your Posit representative.

---

## Important Disclaimer

**This project contains synthetic data and analysis created for demonstration purposes only.**

All data, insights, business scenarios, and analytics presented in this demonstration project have been artificially generated using AI. The data does not represent actual business information, performance metrics, customer data, or operational statistics.

### Key Points:

- **Synthetic Data**: All datasets are computer-generated and designed to illustrate analytical capabilities
- **Illustrative Analysis**: Insights and recommendations are examples of the types of analysis possible with Posit tools
- **No Actual Business Data**: No real business information or data was used or accessed in creating this demonstration
- **Educational Purpose**: This project serves as a technical demonstration of data science workflows and reporting capabilities
- **AI-Generated Content**: Analysis, commentary, and business scenarios were created by AI for illustration purposes
- **No Real-World Implications**: The scenarios and insights presented should not be interpreted as actual business advice or strategies

This demonstration showcases how Posit's data science platform and open-source tools can be applied to the pharmaceutical industry. The synthetic data and analysis provide a foundation for understanding the potential value of implementing similar analytical workflows with actual business data.

For questions about adapting these techniques to your real business scenarios, please contact your Posit representative.

---

*This demonstration was created using Posit's commercial data science tools and open-source packages. All synthetic data and analysis are provided for evaluation purposes only.*
