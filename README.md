🫀Heart Disease Classification Project

Project Description
A heart disease classifier based on the CDC dataset (319,795 samples).
The project implements a classification pipeline to predict the occurrence of heart disease based
on a set of medical and lifestyle indicators.

Project Goal:
To create an efficient classification model that predicts the occurrence of heart disease in the
population based on medical and demographic data.

Key Performance Indicators (KPIs)
· Weighted F1-score ≥ 0.75 (metric tracked in MLflow)
· LogLoss as an indicator of probabilistic classification quality
· Stability of metrics in cross-validation (std ≤ 0.05)
· Automated hyperparameter tuning using Optuna
· Code and model version tracking via MLflow + Git

Risk Assessment
· Data imbalance risk: Highly imbalanced classes (91.4% "No", 8.6% "Yes") – requires

balancing techniques
· Model dependency risk: Usage of a specific framework (CatBoost)
· Overfitting: Controlled through early stopping and cross-validation
· Reproducibility risk: Minimized by version control (joblib, mlflow, git hash,
random_state)

Dataset Description
The dataset contains 319,795 samples and 18 columns (17 features + 1 label).
Column Descriptions
· HeartDisease – Heart disease diagnosis (Yes/No) – TARGET VARIABLE
· BMI – Body Mass Index

Column Description
HeartDisease Heart disease diagnosis (Yes/No) – TARGET VARIABLE
BMI Body Mass Index
Smoking Smoking status (Yes/No)
AlcoholDrinking Alcohol consumption (Yes/No)
Stroke Stroke history (Yes/No)
PhysicalHealth Days of poor physical health (0–30)
MentalHealth Days of poor mental health (0–30)
DiffWalking Difficulty walking (Yes/No)
Sex Gender (Male/Female)
AgeCategory Age category
Race Race/ethnicity
Diabetic Diabetes status (Yes/No/Borderline/Yes during pregnancy)
PhysicalActivity Physical/activity (Yes/No)
GenHealth General health (Excellent/Very good/Good/Fair/Poor)
SleepTime Hours of sleep per day
Asthma Asthma (Yes/No)
KidneyDisease Kidney disease (Yes/No
SkinCancer Skin cancer (Yes/No)
Target Variable Distribution
· No (No heart disease): 292,422 samples (91.4%)
· Yes (Heart disease): 27,373 samples (8.6%)

⚠️NOTE: Significant class imbalance – requires balancing techniques (SMOTE, class_weight,
stratified sampling)
Model Description
The classifier uses CatBoostClassifier (gradient boosting algorithm).
CatBoost Key Features
· Native categorical support without one-hot encoding
· Ordered boosting – reduces overfitting risk
· Built-in handling of imbalanced classes via class_weights

Hyperparameter Tuning
· Implemented with Optuna and mlflow.start_run(nested=True)
· Parameters saved in best_params.pkl and logged in MLflow
· Key parameters to optimize:
o iterations
o learning_rate
o depth
o l2_leaf_reg
o class_weights

Cross-Validation
· Implemented via catboost.cv with 5-fold stratified shuffle
· Results (F1, LogLoss, AUC-ROC) visualized with error bands using Plotly

Project Structure
├── LICENSE
├── Makefile
├── README.md
├── data
│ ├── external <- Data from external sources
│ ├── interim <- Intermediate transformed data
│ ├── processed <- Final datasets for modeling
│ └── raw <- Original immutable raw data
├── models <- Trained models and predictions
├── notebooks <- Jupyter notebooks (EDA, experiments)
├── pyproject.toml
├── references <- Data dictionaries, manuals
├── results <- Generated plots and outputs
├── reports <- Reports (HTML, PDF, LaTeX, etc.)
│ └── figures <- Figures used in reports
├── requirements.txt
├── setup.cfg
└── ARISA_DSML <- Project source code
├── __init__.py
├── config.py
├── helpers.py
├── preproc.py
├── predict.py
├── train.py
└── resolve.py

Prerequisites
· Python 3.11+
· Pandas & NumPy
· Scikit-learn
· Matplotlib & Seaborn
· Jupyter Notebook
· MLflow
· Git & GitHub

Installation & Run
1. Clone repository
git clone https://github.com/Pawel20240101/PZ_ARISA_MLOps_Final.git
cd PZ_ARISA_MLOps_Final

2. Createvirtualenvironment
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

3. Installdependencies
pip install -r requirements.txt

4. Placedataset
Copy heart_2020_cleaned.csv into data/raw/

5. RunMLflowUI
mlflow ui --port 5001
Or
mlflow ui --host 127.0.0.1 --port 5001
Open: http://localhost:5001

Data Processing
· Normalize numerical features: BMI, PhysicalHealth, MentalHealth, SleepTime
· Encode categorical features: binary + multi-class
· Apply balancing techniques: SMOTE, class weights, stratified sampling

Evaluation Metrics
· Weighted F1-score
· Precision/Recall for positive class
· AUC-ROC
· Confusion Matrix
· LogLoss

Monitoring & Support
· Pipeline monitoring (alerts, logs)
· Model monitoring (MLflow metrics)
· Data drift monitoring (NannyML)

CI/CD Pipeline
· Automatic workflows: linting, retraining, prediction
· Branch protection & code reviews
· Pre-commit hooks (flake8)

Reproducibility
· Git version control
· Random state seeding
· MLflow tracking (params, metrics, artifacts)
· requirements.txt + pyproject.toml
· CI/CD pipeline as the only deployment path

Experiment Results
· Mean F1 Score ~0.77 (stable after ~50 iterations)
· Mean LogLoss ~0.49 (after convergence)
· Stable metrics across folds (std ≪0.05)
· SHAP analysis confirms key predictors:
o AgeCategory
o GenHealth
o Stroke
o BMI

Medical Recommendations
· Focus on elderly patients
· Monitor stroke survivors
· Control BMI
· Conduct holistic health assessments

Conclusion
· Model achieved KPIs (F1 ~0.77, LogLoss ~0.49)
· High stability across CV folds
· Fairness – Race has minimal influence
· Interpretability – SHAP values provide medical insight
· Clinically relevant: age, health, stroke are main predictors