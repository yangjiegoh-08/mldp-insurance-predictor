# MediCost AI — Medical Insurance Charges Predictor

A machine learning web application that predicts medical insurance charges for individuals based on personal and health-related attributes. Built as part of the CAI2C08 Machine Learning for Developers module at Temasek Polytechnic.

## Business Understanding
Insurance companies need to accurately price premiums for new policyholders. Inaccurate pricing can lead to financial losses or overcharging customers. This application uses machine learning to predict annual medical insurance charges based on an individual's personal and health profile, helping insurers make data-driven pricing decisions.

## Dataset
- **Source:** [Kaggle Medical Insurance Dataset](https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction)
- **Size:** 1,338 rows × 7 columns
- **Target:** `expenses` — annual medical insurance charges (USD)
- **Features:** age, sex, BMI, children, smoker status, region

## What I Did

### 1. Exploratory Data Analysis (EDA)
- Analysed distribution of target variable and features
- Found that `expenses` is right-skewed
- Identified smoker status and BMI as the strongest predictors of insurance charges
- Visualised relationships using correlation heatmap, boxplots and scatter plots

### 2. Data Preparation
- Removed duplicate rows (1 duplicate found)
- Encoded categorical columns (sex, smoker, region)
- Feature engineering:
    - `bmi_smoker` — interaction term (BMI × smoker flag) to capture combined effect of high BMI and smoking
    - `obese` — binary flag for BMI ≥ 30 to capture threshold effect
    - `log_expenses` — log-transform on target to handle right skew
- Train-test split: 70% train, 30% test

### 3. Modelling
Trained and compared 3 baseline models:

| Model | RMSE | R2 |
|---|---|---|
| Linear Regression | 0.4227 | 0.7969 |
| Random Forest | 0.3958 | 0.8220 |
| Gradient Boosting | 0.3840 | 0.8323 |

Gradient Boosting achieved the best performance and was selected for hyperparameter tuning.

### 4. Hyperparameter Tuning
Used `RandomizedSearchCV` to tune Gradient Boosting with the following parameters:
- `n_estimators`: [100, 200, 300]
- `max_depth`: [3, 5, 10]
- `min_samples_split`: [2, 5, 10]

Best parameters found: `n_estimators=100`, `max_depth=3`, `min_samples_split=2`

### 5. Final Model
**Tuned Gradient Boosting** selected as final model:
- RMSE: 0.3840
- R2: 0.8323

Model saved as `insurance_model.pkl` using joblib.

## Tech Stack
- **Language:** Python 3.13
- **ML Library:** scikit-learn
- **Data Processing:** pandas, numpy
- **Visualisation:** matplotlib, seaborn
- **Web App:** Streamlit
- **Model Saving:** joblib

## Project Structure
mldp-insurance-predictor/
├── MLDP Program Codes.ipynb # Jupyter notebook with full ML pipeline
├── streamlit_app.py # Streamlit web application
├── insurance_model.pkl # Trained Gradient Boosting model
├── insurance.csv # Dataset
└── requirements.txt # Required libraries

## How to Run the App Locally
1. Clone the repository:

git clone https://github.com/yangjiegoh-08/mldp-insurance-predictor

2. Install required libraries:

pip install -r requirements.txt

3. Run the Streamlit app:

streamlit run streamlit_app.py

4. Open your browser and go to `http://localhost:8501`

## Live Demo
https://mldp-insurance-predictor-5ybzpxmnw2e5skjjnhh2kz.streamlit.app/

## Author
Goh Yang Jie(2501307I) | Temasek Polytechnic | School of Informatics & IT | AY2026/2027