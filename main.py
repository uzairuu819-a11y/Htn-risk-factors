import glob
import pandas as pd
import streamlit as st
import xgboost as xgb

st.set_page_config(
    page_title="Hypertension Risk CDSS", page_icon="🩺", layout="wide"
)


@st.cache_resource
def load_and_train():
  csv_files = glob.glob("*.csv")
  if not csv_files:
    st.error(
        "No CSV dataset found in your GitHub repository! Please ensure your"
        " dataset file is uploaded."
    )
    st.stop()

  df = pd.read_csv(csv_files[0])

  target_col = df.columns[-1]
  X = df.drop(columns=[target_col])
  y = df[target_col]

  X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

  if y.dtype == "object" or y.dtype.name == "category":
    y = y.astype("category").cat.codes
  else:
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

  model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
  model.fit(X, y)

  return model, X.columns, X


st.title("Hypertension Risk CDSS")
st.write(
    "Enter patient vitals and history in the sidebar to calculate hypertension"
    " risk."
)

try:
  model, feature_names, X_data = load_and_train()
except Exception as e:
  st.error(f"Error loading model or data: {e}")
  st.stop()

st.sidebar.header("Patient Vitals & History")
input_data = {}

for col in feature_names:
  default_val = float(X_data[col].mean()) if len(X_data) > 0 else 0.0
  input_data[col] = st.sidebar.number_input(label=str(col), value=default_val)

if st.sidebar.button("Predict Risk", type="primary"):
  input_df = pd.DataFrame([input_data])
  prediction = model.predict(input_df)[0]
  probability = model.predict_proba(input_df)[0][1]

  st.subheader("Assessment Result")
  if prediction == 1:
    st.error(
        f"**High Risk of Hypertension** (Probability: {probability * 100:.2f}%)"
    )
  else:
    st.success(
        f"**Low Risk of Hypertension** (Probability: {probability * 100:.2f}%)"
    )
