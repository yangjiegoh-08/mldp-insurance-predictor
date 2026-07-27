import joblib
import streamlit as st
import numpy as np
import pandas as pd

## Load trained model
model = joblib.load("insurance_model.pkl")

## Page config
st.set_page_config(
    page_title="Medical Insurance Charges Predictor",
    page_icon="🏥",
    layout="centered"
)

## CSS
st.markdown("""
<style>

.stApp {
    background: linear-gradient(160deg, #1a1f2e, #1e2a3a, #1a1f2e);
}

.main-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}
.main-header h1 {
    color: #1a365d;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.main-header p {
    color: #4a5568;
    font-size: 0.95rem;
}

.section-title {
    color: #2b6cb0;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    margin-top: 1rem;
}

.result-box {
    background: linear-gradient(135deg, #2b6cb0, #1a365d);
    color: white;
    padding: 1.8rem;
    border-radius: 12px;
    text-align: center;
    margin: 1rem 0;
}
.result-label {
    font-size: 0.85rem;
    opacity: 0.8;
    margin-bottom: 0.3rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.result-amount {
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0.2rem 0;
}
.result-note {
    font-size: 0.78rem;
    opacity: 0.6;
    margin-top: 0.5rem;
}

.stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #1a365d);
    color: white;
    border: none;
    padding: 0.7rem 2rem;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    margin-top: 0.5rem;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1a365d, #2b6cb0);
    color: white;
}

.summary-card {
    background: white;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #e2e8f0;
    margin-top: 1rem;
}
.summary-row {
    display: flex;
    justify-content: space-between;
    padding: 0.45rem 0;
    border-bottom: 1px solid #f0f0f0;
    font-size: 0.9rem;
    color: #2d3748;
}
.summary-key { color: #718096; }
.summary-val { font-weight: 600; color: #1a365d; }

.risk-low {
    display: inline-block;
    background: #c6f6d5;
    color: #276749;
    padding: 0.25rem 0.9rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    margin-top: 0.6rem;
}
.risk-medium {
    display: inline-block;
    background: #fefcbf;
    color: #744210;
    padding: 0.25rem 0.9rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    margin-top: 0.6rem;
}
.risk-high {
    display: inline-block;
    background: #fed7d7;
    color: #822727;
    padding: 0.25rem 0.9rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    margin-top: 0.6rem;
}

</style>
""", unsafe_allow_html=True)

## Header
st.markdown("""
<div class="main-header">
    <h1>🏥 Medical Insurance Charges Predictor</h1>
    <p>Enter your details below to get an estimated annual insurance charge.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

## Personal Info
st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age_selected      = st.slider("Age", min_value=18, max_value=64, value=30)
    children_selected = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
with col2:
    sex_selected    = st.selectbox("Sex", ["Male", "Female"])
    region_selected = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

## Health Info
st.markdown('<div class="section-title">💉 Health Information</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    height_cm = st.slider("Height (cm)", min_value=140, max_value=210, value=170)
with col4:
    weight_kg = st.slider("Weight (kg)", min_value=30, max_value=150, value=70)

## Auto-calculate BMI
bmi_selected = weight_kg / ((height_cm / 100) ** 2)

## BMI category
if bmi_selected < 18.5:
    bmi_cat = "⚠️ Underweight"
elif bmi_selected < 25:
    bmi_cat = "✅ Normal weight"
elif bmi_selected < 30:
    bmi_cat = "⚠️ Overweight"
else:
    bmi_cat = "🔴 Obese"

st.caption(f"Your calculated BMI: **{bmi_selected:.1f}** — {bmi_cat}")

smoker_selected = st.selectbox("Smoking Status", ["No", "Yes"])

st.divider()

## Predict button
if st.button("🔍 Predict Insurance Charges"):

    ## Input validation
    if age_selected < 18 or age_selected > 64:
        st.error("⚠️ Age must be between 18 and 64.")
    elif bmi_selected < 10.0 or bmi_selected > 55.0:
        st.error("⚠️ Please enter a valid height and weight.")
    else:
        ## Convert inputs
        sex_val        = 1 if sex_selected == "Male" else 0
        smoker_val     = 1 if smoker_selected == "Yes" else 0
        bmi_smoker_val = bmi_selected * smoker_val
        obese_val      = 1 if bmi_selected >= 30 else 0

        region_northwest = 1 if region_selected == "Northwest" else 0
        region_southeast = 1 if region_selected == "Southeast" else 0
        region_southwest = 1 if region_selected == "Southwest" else 0

        ## Build DataFrame
        df_input = pd.DataFrame({
            "age":              [age_selected],
            "sex":              [sex_val],
            "bmi":              [bmi_selected],
            "children":         [children_selected],
            "smoker":           [smoker_val],
            "region_northwest": [region_northwest],
            "region_southeast": [region_southeast],
            "region_southwest": [region_southwest],
            "bmi_smoker":       [bmi_smoker_val],
            "obese":            [obese_val]
        })

        df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

        ## Predict
        log_pred          = model.predict(df_input)[0]
        predicted_charges = np.exp(log_pred)

        ## Risk badge
        if predicted_charges < 8000:
            risk_html = '<span class="risk-low">🟢 Low Risk</span>'
        elif predicted_charges < 20000:
            risk_html = '<span class="risk-medium">🟡 Medium Risk</span>'
        else:
            risk_html = '<span class="risk-high">🔴 High Risk</span>'

        ## Result
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Estimated Annual Insurance Charges</div>
            <div class="result-amount">${predicted_charges:,.2f}</div>
            {risk_html}
            <div class="result-note">Based on the details you provided</div>
        </div>
        """, unsafe_allow_html=True)

        ## Summary
        st.markdown("""
        <div class="summary-card">
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">📋 Input Summary</div>', unsafe_allow_html=True)

        for key, val in [
            ("Age",        f"{age_selected} years"),
            ("Sex",        sex_selected),
            ("Height",     f"{height_cm} cm"),
            ("Weight",     f"{weight_kg} kg"),
            ("BMI",        f"{bmi_selected:.1f} — {bmi_cat}"),
            ("Children",   str(children_selected)),
            ("Smoker",     smoker_selected),
            ("Region",     region_selected),
        ]:
            st.markdown(f"""
            <div class="summary-row">
                <span class="summary-key">{key}</span>
                <span class="summary-val">{val}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)