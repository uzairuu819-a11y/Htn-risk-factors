import streamlit as st
import pandas as pd
import xgboost as xgb



st.title("Hypertension Risk CDSS")
st.write("Enter patient vitals and history in the sidebar to calculate hypertension risk.")

# Load dataset and train model
@st.cache_data
def load_and_train():
    data = pd.read_csv('hypertension_risk_dataset.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X = pd.get_dummies(X)
    if y.dtype == 'object':
        y = y.astype('category').cat.codes
    
    model = xgb.XGBClassifier(eval_metric='logloss')
        # Ensure all features in X are numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Ensure target variable y is numeric
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = y.astype('category').cat.codes
model.fit(X, y)
    return model, X.columns, X

model, feature_names, X_data = load_and_train()

# Build Sidebar Controls
st.sidebar.header("Patient Vitals")
patient_data = {}

for col in feature_names:
    if X_data[col].nunique() <= 2:
        patient_data[col] = st.sidebar.selectbox(f"{col}", [0, 1])
    else:
        min_val = float(X_data[col].min())
        max_val = float(X_data[col].max())
        mean_val = float(X_data[col].mean())
        patient_data[col] = st.sidebar.slider(f"{col}", min_val, max_val, mean_val)

# Predict Outcome
if st.sidebar.button("Calculate Risk"):
    patient_df = pd.DataFrame([patient_data])
    risk_probability = model.predict_proba(patient_df)[0][1] * 100
    
    st.header(f"Risk Score: {risk_probability:.1f}%")
    
    if risk_probability > 50:
        st.error("⚠️ High Risk of Hypertension. Early intervention recommended.")
    else:
        st.success("✅ Low Risk. Maintain current lifestyle.")
