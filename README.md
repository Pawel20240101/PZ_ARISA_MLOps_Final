# 🫀 Heart Disease Classification Project  

---

## 📌 Description  
A heart disease classifier built on the **CDC dataset (319,795 samples)**.  
Implements a full **MLOps pipeline** for predicting heart disease occurrence based on medical and lifestyle indicators.  

### 🎯 Goal  
Develop a reliable, reproducible model to predict heart disease based on medical and demographic data.  

---

## 📊 Key Performance Indicators (KPIs)  
- **F1-score (weighted) ≥ 0.75** (logged in MLflow)  
- **LogLoss** as a measure of probabilistic prediction quality  
- **Cross-validation stability (std ≤ 0.05)**  
- Automated hyperparameter tuning with **Optuna**  
- Experiment and model tracking via **MLflow + Git**  

---

## ⚠️ Risks  
- **Data imbalance** – 91.4% No, 8.6% Yes → handled with `class_weights` + stratified sampling  
- **Framework dependency** – CatBoost  
- **Overfitting** – controlled with early stopping + CV  
- **Reproducibility** – ensured via MLflow, joblib, git hash, random_state  

---

## 📂 Dataset  
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

## 🤖 Model  
- **Algorithm:** CatBoostClassifier (gradient boosting)  
- **Strengths:**  
  - Native categorical support  
  - Ordered boosting → less overfitting  
  - Built-in handling of class imbalance  

---

## 🔧 Hyperparameter Tuning  
- **Optuna** + MLflow  
- Best params stored in `best_params.pkl`  
- Optimized: `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, `class_weights`  

---

## 🔁 Cross-Validation  
- **5-fold stratified CV** with CatBoost  
- Metrics: F1, LogLoss, AUC-ROC  
- Results visualized with **Plotly**  

---

## 🏗 Project Structure  
```bash
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── external     # external datasets
│   ├── interim      # intermediate processed data
│   ├── processed    # final datasets (train/test)
│   └── raw          # raw input data
├── models           # trained models + predictions
├── notebooks        # Jupyter notebooks (EDA, experiments)
├── pyproject.toml
├── references       # data dictionaries, manuals
├── results          # generated plots, evaluation metrics
├── reports          # final reports
│   └── figures      # images used in reports
├── requirements.txt
├── setup.cfg
└── ARISA_DSML
    ├── __init__.py
    ├── config.py
    ├── helpers.py
    ├── preproc.py
    ├── train.py
    ├── predict.py
    └── resolve.py
```

---

## ⚙️ Prerequisites  
- Python 3.11+  
- Pandas & NumPy  
- Scikit-learn  
- Matplotlib & Seaborn  
- Jupyter Notebook  
- MLflow  
- Git & GitHub  

---

## 🚀 Installation & Run  

1. Clone repository  
```bash
git clone https://github.com/Pawel20240101/PZ_ARISA_MLOps_Final.git
cd PZ_ARISA_MLOps_Final
```

2. Create virtual environment  
```bash
python -m venv .venv
# Windows
.\.venv\Scripts ctivate
# Mac/Linux
source .venv/bin/activate
```

3. Install dependencies  
```bash
pip install -r requirements.txt
```

4. Place dataset  
Copy `heart_2020_cleaned.csv` into `data/raw/`  

5. Run MLflow UI  
```bash
mlflow ui --port 5001
# or
mlflow ui --host 127.0.0.1 --port 5001
```
Open: [http://localhost:5001](http://localhost:5001)  

---

## 🧹 Data Processing  
- Normalize numerical features: `BMI`, `PhysicalHealth`, `MentalHealth`, `SleepTime`  
- Encode categorical features: binary + multi-class  
- Apply balancing techniques: class weights, stratified sampling  

---

## 📈 Evaluation Metrics  
- Weighted F1-score  
- Precision/Recall for positive class  
- AUC-ROC  
- Confusion Matrix  
- LogLoss  

---

## 📡 Monitoring & Support  
- **Pipeline monitoring:** alerts, logs  
- **Model monitoring:** MLflow metrics  
- **Data drift monitoring:** NannyML  

---

## 🔄 CI/CD Pipeline  
- **Automatic workflows:** linting, retraining, prediction  
- **Branch protection & code reviews**  
- **Pre-commit hooks (flake8)**  

---

## 🔁 Reproducibility  
- Git version control  
- Random state seeding  
- MLflow tracking (params, metrics, artifacts)  
- `requirements.txt` + `pyproject.toml`  
- CI/CD pipeline as the only deployment path  

---

## 📊 Experiment Results  
- Mean F1 Score ~0.77 (stable after ~50 iterations)  
- Mean LogLoss ~0.49 (after convergence)  
- Stable metrics across folds (std ≪ 0.05)  
- SHAP analysis confirms key predictors: `AgeCategory`, `GenHealth`, `Stroke`, `BMI`  

---

## 📜 Medical Recommendations  
- Focus on elderly patients  
- Monitor stroke survivors  
- Control BMI  
- Conduct holistic health assessments  

---

## ✅ Conclusion  
- Model achieved KPIs (F1 ~0.77, LogLoss ~0.49)  
- High stability across CV folds  
- Fairness – Race has minimal influence  
- Interpretability – SHAP values provide medical insight  
- Clinically relevant: age, health, stroke are main predictors  

---
