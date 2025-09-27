# Heart Disease Classification Project

## Description
A heart disease classifier built on the **CDC dataset (319,795 samples)**.
Implements a full **MLOps pipeline** for predicting heart disease occurrence based on medical and lifestyle indicators.

### Goal
Develop a reliable, reproducible model to predict heart disease based on medical and demographic data.

## Key Performance Indicators (KPIs)
- **F1-score (weighted) ≥ 0.75** (logged in MLflow)
- **LogLoss** as a measure of probabilistic prediction quality
- **Cross-validation stability (std ≤ 0.05)**
- Automated hyperparameter tuning with **Optuna**
- Experiment and model tracking via **MLflow + Git**

## Risks
- **Data imbalance** – 91.4% No, 8.6% Yes → handled with `class_weights` + stratified sampling
- **Framework dependency** – CatBoost
- **Overfitting** – controlled with early stopping + CV
- **Reproducibility** – ensured via MLflow, joblib, git hash, random_state

## Dataset
- **Samples:** 319,795
- **Features:** 17 predictors + 1 target (`HeartDisease`)
- **Target distribution:**
  - No → 91.4%
  - Yes → 8.6%

### Features
| Feature          | Description                          |
|------------------|--------------------------------------|
| HeartDisease     | Diagnosis (Yes/No) – **Target**      |
| BMI              | Body Mass Index                     |
| Smoking          | Smoking status (Yes/No)             |
| AlcoholDrinking  | Alcohol consumption (Yes/No)        |
| Stroke           | Stroke history (Yes/No)             |
| PhysicalHealth   | Days of poor physical health (0–30) |
| MentalHealth     | Days of poor mental health (0–30)   |
| DiffWalking      | Difficulty walking (Yes/No)         |
| Sex              | Gender                              |
| AgeCategory      | Age group                           |
| Race             | Race/ethnicity                      |
| Diabetic         | Diabetes status                     |
| PhysicalActivity | Activity level                      |
| GenHealth        | General health rating               |
| SleepTime        | Hours of sleep                      |
| Asthma           | Asthma (Yes/No)                     |
| KidneyDisease    | Kidney disease (Yes/No)             |
| SkinCancer       | Skin cancer (Yes/No)                |

---

## Model
- **Algorithm:** CatBoostClassifier (gradient boosting)
- **Strengths:**
  - Native categorical support
  - Ordered boosting → less overfitting
  - Built-in handling of class imbalance

---

## Hyperparameter Tuning
- **Optuna** + MLflow
- Best params stored in `best_params.pkl`
- Optimized: `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, `class_weights`

---

## Cross-Validation
- **5-fold stratified CV** with CatBoost
- Metrics: F1, LogLoss, AUC-ROC
- Results visualized with **Plotly**

---

## Project Structure
```
PZ_ARISA_MLOps_Final/
├── .devcontainer/           # Codespaces / Docker configuration
├── .github/workflows/       # CI/CD pipeline
│   ├── ci.yml              # code linting and formatting
│   └── lint-code.yml       # additional PR tests
├── ARISA_DSML/             # main source code
│   ├── config.py           # project configuration
│   ├── preproc.py          # data processing (Kaggle API)
│   ├── train.py            # model training with Optuna
│   ├── predict.py          # predictions and drift monitoring
│   ├── resolve.py          # model management (champion/challenger)
│   └── helpers.py          # utility functions
├── data/                   # project data (excluded from git)
│   ├── raw/               # raw data from Kaggle
│   ├── interim/           # intermediate data
│   ├── processed/         # processed data
│   └── external/          # external data
├── models/                 # saved models (excluded from git)
├── reports/                # reports and visualizations (excluded from git)
│   └── figures/           # SHAP plots, CV charts
├── results/                # experiment outputs
├── mlruns/                 # MLflow tracking (excluded from git)
├── mlartifacts/            # MLflow artifacts (excluded from git)
├── notebooks/              # Jupyter notebooks
│   ├── 01-Before_MLOps.ipynb
│   └── 02-Model_version.ipynb
├── tests/                  # unit tests
├── docs/                   # documentation
├── Makefile               # task automation
├── README.md              # project description
├── pyproject.toml         # package configuration
├── setup.cfg              # tool configuration
├── requirements.txt       # Python dependencies
└── .gitignore             # git exclusions (data/, models/, mlruns/)
```

---

## Prerequisites
- Python 3.11+
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Plotly
- Jupyter Notebook
- MLflow
- Git & GitHub
- Kaggle API credentials

---

## Installation & Setup

1. **Clone repository**
```bash
git clone https://github.com/Pawel20240101/PZ_ARISA_MLOps_Final.git
cd PZ_ARISA_MLOps_Final
```

2. **Create virtual environment**
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup Kaggle API**
- Place `kaggle.json` in project root or configure Kaggle credentials
- Data will be automatically downloaded via `preproc.py`

5. **Start MLflow UI**
```bash
mlflow ui --host 127.0.0.1 --port 5000
```

---

## MLOps Pipeline

### Local Development (Makefile)
```bash
# Full pipeline (development & debugging)
make test                  # Complete: linting + preprocess + train + predict

# Individual stages
make preprocess           # Download and process data from Kaggle
make train               # Hyperparameter tuning + model training
make predict             # Predictions + drift monitoring + SHAP

# Code quality
make lint                # Check formatting (black, flake8, isort)
make format              # Auto-format code
```

### CI/CD (GitHub Actions)
- **ci.yml** (push/PR to main/test): linting, formatting, unit tests
- **lint-code.yml** (PR to main): additional code quality checks
- **No ML pipeline in CI** - only code quality (fast, lightweight)

---

## Data Processing
- **Automatic download** from Kaggle via API
- **Target conversion:** Yes/No → 1/0
- **Train/test split:** stratified (80/20)
- **Class balancing:** class_weights + stratified sampling
- **Feature validation:** categorical column checking

---

## Evaluation Metrics
- **F1-score (weighted)** - primary metric
- Precision/Recall for positive class
- AUC-ROC
- Confusion Matrix
- LogLoss

---

## Monitoring & Support
- **MLflow:** experiment tracking, model registry, champion/challenger pattern
- **NannyML:** data drift detection, performance estimation
- **SHAP:** model interpretability and feature importance
- **Git integration:** commit hash tracking for reproducibility

---

## Git Management
Large files excluded via `.gitignore`:
- `data/` - downloaded via Kaggle API
- `models/` - generated during training
- `mlruns/`, `mlartifacts/` - MLflow artifacts
- `reports/` - generated plots and charts

**Benefits:** Lightweight repository (~MB vs hundreds of MB)

---

## CI/CD Pipeline
- **GitHub Actions:** code quality only (linting, formatting, tests)
- **Branch protection:** main branch requires PR + reviews
- **Automated checks:** black, flake8, isort, pytest
- **No ML training in CI** - keeps pipelines fast and focused

---

## Reproducibility
- **Git version control** with commit hash tracking
- **Random state seeding** across all components
- **MLflow tracking** for parameters, metrics, and artifacts
- **Dependency management** via requirements.txt
- **Automated data download** ensures consistent datasets

---

## Current Experiment Results
**Cross-Validation (N=5)**
- **Mean F1 Score:** ~0.15 (below target - requires improvement)
- **Mean LogLoss:** ~0.49 (after convergence)
- **Standard deviation:** <0.05 (stable, no overfitting)

**SHAP Analysis**
- **Key predictors:** AgeCategory, GenHealth, Stroke, BMI
- **Low impact:** Race (model avoids discrimination)

---

## Performance Issues & Recommendations
**Current challenges:**
- F1-score significantly below target KPI (0.15 vs ≥0.75)
- Strong class imbalance affects model performance

**Improvement strategies:**
- **SMOTE/ADASYN** for synthetic sample generation
- **Threshold optimization** for classification
- **Ensemble methods** (voting, stacking)
- **Feature engineering** (interactions, derived features)
- **Advanced balancing** techniques

---

## Medical Insights
**Key risk factors identified:**
- Advanced age (AgeCategory)
- Poor general health status (GenHealth)
- Stroke history
- High BMI

**Clinical relevance:**
- Model provides interpretable predictions via SHAP
- Focus areas for preventive medicine
- Requires optimization before clinical deployment

---

## Development Status
- ✅ Complete MLOps pipeline (preprocess → train → predict)
- ✅ Experiment tracking and model registry
- ✅ Data drift monitoring
- ✅ CI/CD for code quality
- ⚠️ Model performance requires improvement
- 🔄 Active development for better F1-score

---