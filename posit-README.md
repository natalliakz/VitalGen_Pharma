# VitalGen Therapeutics Demo - Internal Posit Guide

**Customer:** Merck (Current Customer)
**Business Unit:** Internal Analytics Platform Group
**Audience:** Phase 1 demo with engineers (non-GxP)
**Use Case:** Clinical trial enrollment analytics and optimization

## Quick Start (5 minutes)

```bash
# 1. Navigate to project
cd VitalGen_Pharmaceuticals

# 2. Install dependencies
uv sync

# 3. Generate data and train model
uv run python data/generate_data.py
uv run python ml/train_model.py

# 4. Launch Shiny dashboard
uv run shiny run app.py --port 8501
# Open http://localhost:8501
```

## Project Context

### Why This Demo

Merck's Internal Analytics Platform group is evaluating Posit products for their engineering team. They need to understand:
- **Python ETL capabilities** (polars as tidyverse/dplyr alternative)
- **Framework flexibility** (Shiny vs. Streamlit comparison)
- **ML model deployment** (API and interactive apps)
- **End-to-end workflows** (data → analysis → deployment)

### Industry Context: Clinical Trial Enrollment

Clinical trials require successful patient recruitment to demonstrate statistical power. Key challenges:

1. **Enrollment Delays**: Poor recruitment affects 80% of trials, causing 6+ month delays
2. **Site Selection**: Not all sites perform equally - investigator experience, staffing, and location matter
3. **Diversity Goals**: FDA emphasis on representative patient populations
4. **Cost Impact**: Each delay day costs pharmaceutical companies millions

**Demo Value**: Shows how data analytics can predict enrollment success, optimize site selection, and identify at-risk trials early.

### Sources

- [TrialEnroll: Predicting Clinical Trial Enrollment Success](https://arxiv.org/html/2407.13115v1)
- [Prediction of clinical trial enrollment rates](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8870517/)
- [Clinical Trial Enrollment Forecasting Benefits](https://blog.onestudyteam.com/clinical-trial-enrollment-forecasting)

## Data Overview

### Synthetic Datasets

**1. Trials (`synthetic-trials.csv`)**
- 50 trials across 6 therapeutic areas (Oncology, Cardiovascular, Neurology, Immunology, Metabolic, Respiratory)
- Phases: I, II, III with realistic target enrollments
- Protocol complexity: inclusion/exclusion criteria, amendments

**2. Sites (`synthetic-sites.csv`)**
- 1,250 sites in 7 countries (USA, Canada, UK, Germany, France, Spain, Australia)
- 3 site types: Academic Medical Center, Community Hospital, Private Practice
- Investigator experience: 1-25 years
- Site capabilities: staff count, prior trials, patient database size

**3. Enrollment (`synthetic-enrollment.csv`)**
- ~10,800 monthly enrollment records
- Time-series data showing enrollment ramp-up
- Actual vs. target enrollment tracking

**4. Patient Demographics (`synthetic-patient-demographics.csv`)**
- ~21,400 patient records
- Demographics: age, gender, ethnicity, BMI
- Linked to specific trials and sites

**Key Insight**: Data shows realistic patterns - experienced investigators at well-staffed academic centers in urban areas achieve higher enrollment rates.

## Demo Components

### 1. Exploratory Data Analysis (eda.qmd)

**Purpose**: Showcase Quarto for reproducible research with Python + polars ETL

**Key Demonstrations**:
- **Polars ETL**: Show syntax similar to R's dplyr but 10-100x faster
- **Portfolio analytics**: Trials by phase and therapeutic area
- **Site performance**: Country, site type, investigator experience impact
- **Demographics**: Patient diversity analysis
- **Time-series**: Enrollment trends over time

**Demo Flow** (3-5 minutes):
1. Open `eda.html` in browser (pre-rendered)
2. Scroll through key visualizations
3. Highlight polars code examples vs. pandas/dplyr
4. Show branded output with `_brand.yml`

**Talking Points**:
- "Quarto works seamlessly with Python and R"
- "Polars is 10-100x faster than pandas on large datasets"
- "Syntax mirrors tidyverse - easy for R users to learn"
- "Branded output uses `_brand.yml` for consistency"

### 2. Machine Learning Model

**Purpose**: Predict enrollment success probability for trial sites

**Model Details**:
- Algorithm: Random Forest (simple, interpretable)
- Features: 14 site and trial characteristics
- Performance: 97.9% accuracy, 0.82 ROC-AUC
- Predicts: Binary (meets target or not) + probability

**Top Predictive Factors**:
1. Target enrollment per site (higher = harder)
2. Investigator experience (more = better)
3. Site staff count (more = better)
4. Trial phase (complexity factor)
5. Patient database size (larger = better)

**Demo Flow** (2 minutes):
1. Show `ml/model_metadata.json` for performance metrics
2. Highlight feature importance
3. Explain how model informs site selection

**Talking Points**:
- "Simple models are easier to explain to stakeholders"
- "Feature importance drives actionable insights"
- "Model can be retrained as new data arrives"

### 3. Interactive Dashboards

#### Shiny for Python App (app.py)

**Purpose**: Demonstrate Posit's Shiny framework for Python

**Features**:
- 3 tabs: Overview, Site Performance, Enrollment Predictor
- Reactive filtering by country and site type
- ML predictions with adjustable parameters
- Branded with `_brand.yml` integration

**Demo Flow** (5-7 minutes):
1. **Overview Tab**: Show portfolio metrics and visualizations
   - "These metrics update automatically with new data"

2. **Site Performance Tab**: Filter sites by country/type
   - "Filters update all visualizations reactively"
   - "Top performers table helps identify best sites"

3. **Enrollment Predictor Tab**: Adjust site characteristics
   - "Change investigator experience - see probability update"
   - "Model predicts 85%+ success for strong sites"
   - "Risk levels help prioritize interventions"

**Talking Points**:
- "Shiny for Python brings Shiny's power to Python users"
- "Integrates seamlessly with Posit Workbench and Connect"
- "Branded theme from `_brand.yml` - no manual CSS"
- "Same reactive programming model as Shiny for R"

#### Streamlit App (streamlit_app.py)

**Purpose**: Show alternative framework comparison

**Features**:
- Same functionality as Shiny app
- Different interaction paradigm
- Popular in ML/data science community

**Demo Flow** (3 minutes):
1. Launch: `uv run streamlit run streamlit_app.py`
2. Navigate through pages
3. Test prediction with example inputs

**Talking Points**:
- "Streamlit is popular but Shiny offers deeper integration"
- "Both frameworks supported on Posit infrastructure"
- "Shiny provides more control for complex apps"
- "Engineers can choose the right tool for their use case"

### 4. REST API (api.py)

**Purpose**: Deploy ML model as web service

**Endpoints**:
- `GET /health` - Health check
- `GET /data` - Sample data access
- `POST /predict` - ML predictions
- `GET /model-info` - Model metadata
- `GET /docs` - Auto-generated Swagger docs

**Demo Flow** (3-5 minutes):
1. Start API: `uv run uvicorn api:app --port 8000`
2. Open `http://localhost:8000/docs`
3. Test `/predict` endpoint with example data
4. Show response with probability and recommendation

**Talking Points**:
- "FastAPI provides automatic API documentation"
- "Pydantic validates all inputs automatically"
- "Can be deployed to Posit Connect for team access"
- "API enables integration with other systems"

## Pre-Demo Checklist

- [ ] Data generated: `ls data/*.csv` (should show 4 files)
- [ ] Model trained: `ls ml/*.joblib` (should show 3 files)
- [ ] EDA rendered: `ls eda.html` (should exist)
- [ ] Dependencies installed: `uv sync`
- [ ] Shiny app tested: `uv run shiny run app.py` (port 8501)
- [ ] API tested: `curl http://localhost:8000/health`
- [ ] Browser tabs ready: `eda.html`, Shiny app, API docs

## Demo Script (30 minutes)

### Introduction (2 minutes)

"Today we'll demonstrate how Posit tools support modern data science workflows for clinical trial analytics. This project shows Python ETL with polars, ML model development, and deployment via interactive apps and APIs."

### Part 1: Data & Analysis (8 minutes)

1. **Data Overview** (2 min)
   - "We have synthetic data for 50 trials, 1,250 sites, 21,000 patients"
   - "Data generation script ensures reproducibility"
   - Show data generation script briefly

2. **EDA with Quarto** (6 min)
   - Open `eda.html`
   - "Quarto renders to HTML with embedded plots"
   - Scroll through key sections
   - Highlight polars code vs. dplyr comparison
   - "Same report renders to PDF, Word, or slides"

**Key Message**: "Quarto + polars = fast, reproducible analysis for Python users"

### Part 2: Machine Learning (5 minutes)

1. **Model Training**
   - "Random Forest predicts enrollment success"
   - Show `ml/model_metadata.json`
   - "97.9% accuracy on test set"

2. **Feature Importance**
   - "Top factors: target size, investigator experience, staffing"
   - "Actionable insights for site selection"

**Key Message**: "Simple models drive business decisions"

### Part 3: Interactive Apps (10 minutes)

1. **Shiny for Python** (7 min)
   - Launch app: `http://localhost:8501`
   - Walk through all 3 tabs
   - Demonstrate filtering and predictions
   - Highlight branding integration

2. **Streamlit Comparison** (3 min)
   - Launch Streamlit app
   - "Alternative framework with similar capabilities"
   - "Engineers can choose based on needs"

**Key Message**: "Posit supports multiple frameworks on same infrastructure"

### Part 4: API Deployment (5 minutes)

1. **FastAPI Demo**
   - Open `/docs` page
   - "Automatic interactive documentation"
   - Test `/predict` endpoint
   - "Easy integration with other systems"

**Key Message**: "Models → APIs → Production on Posit Connect"

## Key Insights to Highlight

1. **Enrollment Optimization**
   - Experienced investigators (15+ years) show 40% higher enrollment rates
   - Academic Medical Centers outperform community sites
   - Urban sites enroll 30% faster than rural sites

2. **Risk Mitigation**
   - 98% of sites meet targets with model probability >70%
   - Early prediction enables proactive intervention
   - Portfolio view identifies at-risk trials

3. **Technical Capabilities**
   - Polars is 10-100x faster than pandas
   - Shiny + Connect = seamless deployment
   - Single codebase from analysis → production

## Troubleshooting

### Data files missing
```bash
uv run python data/generate_data.py
```

### Model not trained
```bash
uv run python ml/train_model.py
```

### Port already in use
```bash
# Shiny: change port
uv run shiny run app.py --port 8502

# API: change port
uv run uvicorn api:app --port 8001
```

### Dependencies not installing
```bash
# Clear cache and reinstall
rm -rf .venv
uv sync
```

### Quarto not rendering
```bash
# Check Quarto installation
quarto check

# Re-render
uv run quarto render eda.qmd
```

## Next Steps & Call to Action

### For This Customer (Merck)

1. **Short Term**
   - Deploy demo to Posit Workbench for team exploration
   - Gather feedback from engineering team
   - Identify additional use cases

2. **Medium Term**
   - Connect to real (non-GxP) trial data
   - Extend model with additional features
   - Build automated reporting pipelines

3. **Long Term**
   - Scale to full portfolio across business units
   - Integrate with trial management systems
   - Productionize on Posit Connect

### Questions to Ask

1. "What data sources would you want to integrate?"
2. "Are there other frameworks your team uses that you'd like to see?"
3. "What deployment requirements do you have (security, compliance, etc.)?"
4. "Would you like a hands-on workshop for your engineering team?"

## Posit Product Tie-Ins

### Posit Workbench
- IDE for Python and R development
- Jupyter notebook and VS Code integration
- Managed credentials for data connections
- Collaboration features for teams

### Posit Connect
- One-click deployment of Shiny apps
- API hosting with automatic scaling
- Scheduled Quarto reports
- Access controls and usage analytics

### Posit Package Manager
- Curated package repositories
- Version control for reproducibility
- Air-gapped deployment support
- Vulnerability scanning

## Additional Resources

- **Posit Documentation**: https://docs.posit.co
- **Shiny for Python**: https://shiny.posit.co/py/
- **Quarto**: https://quarto.org
- **polars**: https://pola.rs

---

*Demo created for Merck - Internal Analytics Platform Group*
*Last updated: 2026-02-02*
