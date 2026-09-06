import glob
import pandas as pd
import streamlit as st
import xgboost as xgb

# 1. Page Configuration & Professional Medical UI Styling
st.set_page_config(
    page_title="CardioLens - Hypertension Risk CDSS",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stSidebar {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    .stButton>button {
        width: 100%;
        background-color: #0284c7;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #0369a1;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper function to assign standard medical units and physiological bounds
def get_feature_unit_and_bounds(col_name, series):
  col_lower = col_name.lower()
  unit = ""
  min_val = float(series.min()) if not series.empty else 0.0
  max_val = float(series.max()) if not series.empty else 100.0
  default_val = float(series.median()) if not series.empty else 0.0

  if "age" in col_lower:
    unit = "Years"
    min_val, max_val = 1.0, 120.0
  elif (
      "bp" in col_lower
      or "systolic" in col_lower
      or "sbp" in col_lower
      or "blood pressure" in col_lower
  ):
    unit = "mmHg"
    min_val, max_val = 60.0, 250.0
  elif "diastolic" in col_lower or "dbp" in col_lower:
    unit = "mmHg"
    min_val, max_val = 40.0, 150.0
  elif "bmi" in col_lower:
    unit = "kg/m²"
    min_val, max_val = 10.0, 60.0
  elif "glucose" in col_lower or "sugar" in col_lower:
    unit = "mg/dL"
    min_val, max_val = 50.0, 500.0
  elif "cholesterol" in col_lower or "chol" in col_lower:
    unit = "mg/dL"
    min_val, max_val = 100.0, 450.0
  elif "heart" in col_lower or "pulse" in col_lower or "hr" in col_lower:
    unit = "bpm"
    min_val, max_val = 40.0, 200.0
  elif "weight" in col_lower:
    unit = "kg"
    min_val, max_val = 20.0, 300.0
  elif "height" in col_lower:
    unit = "cm"
    min_val, max_val = 50.0, 250.0

  return unit, min_val, max_val, default_val


@st.cache_resource
def load_and_train():
  csv_files = glob.glob("*.csv")
  if not csv_files:
    st.error("⚠️ No CSV dataset found in your GitHub repository.")
    st.stop()

  df = pd.read_csv(csv_files[0])

  # Automatically detect target column if named explicitly, else use the last column
  target_candidates = [
      col
      for col in df.columns
      if col.lower() in ["target", "output", "class", "htn", "hypertension", "risk"]
  ]
  if target_candidates:
    target_col = target_candidates[0]
  else:
    target_col = df.columns[-1]

  X = df.drop(columns=[target_col])
  y = df[target_col]

  X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

  if y.dtype == "object" or y.dtype.name == "category":
    y = y.astype("category").cat.codes
  else:
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

  model = xgb.XGBClassifier(
      eval_metric="logloss", random_state=42, n_estimators=100
  )
  model.fit(X, y)

  return model, X.columns, X, target_col


# App Header
st.title("🩺 CardioLens: Hypertension Risk CDSS")
st.markdown(
    "Clinical Decision Support System for evaluating patient cardiovascular risk"
    " profiles."
)
st.markdown("---")

try:
  model, feature_names, X_data, target_col = load_and_train()
except Exception as e:
  st.error(f"Error loading model or data: {e}")
  st.stop()

# Interactive Sidebar with Medical Units & Bounds
st.sidebar.header("🫀 Patient Vitals & History")
st.sidebar.markdown(
    "Adjust parameters below to simulate patient risk assessment."
)

input_data = {}
for col in feature_names:
  unit, min_v, max_v, default_v = get_feature_unit_and_bounds(col, X_data[col])
  label_text = f"{col} ({unit})" if unit else f"{col}"

  if min_v >= max_v:
    min_v, max_v = 0.0, 100.0

  input_data[col] = st.sidebar.number_input(
      label=label_text,
      value=float(default_v),
      min_value=float(min_v),
      max_value=float(max_v),
      step=1.0,
  )

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("Calculate Hypertension Risk")

# Main Content Layout
col1, col2 = st.columns([2, 1])

with col1:
  st.subheader("📋 Patient Clinical Summary")
  input_df = pd.DataFrame([input_data])
  st.dataframe(input_df, use_container_width=True)

  if predict_btn:
    probabilities = model.predict_proba(input_df)[0]

    # Dynamically map probability to the positive (high risk) class
    if len(probabilities) > 1:
      # If class 1 is high risk, take index 1. Ensure robustness against class label order.
      classes = list(model.classes_)
      if 1 in classes:
        high_risk_idx = classes.index(1)
        probability = probabilities[high_risk_idx]
      else:
        probability = probabilities[1]
    else:
      probability = probabilities[0]

    st.markdown("### 📊 Assessment Report")

    if probability > 0.5:
      st.error(
          f"### 🔴 High Risk of Hypertension Detected\n"
          f"**Confidence Probability:** `{probability * 100:.1f}%`\n\n"
          "*Clinical Recommendation:* Immediate lifestyle intervention, ECG"
          " evaluation, and secondary screening recommended."
      )
    else:
      st.success(
          f"### 🟢 Low Risk of Hypertension\n"
          f"**Confidence Probability:** `{(1 - probability) * 100:.1f}%` (Normal"
          " Range)\n\n"
          "*Clinical Recommendation:* Maintain standard healthy lifestyle"
          " habits and annual check-ups."
      )

    st.progress(float(probability))

with col2:
  st.markdown("### ℹ️ CDSS Guide")
  st.info(
      "**How to use:**\n"
      "1. Enter or modify patient metrics in the sidebar panel with standard"
      " units.\n"
      "2. Ensure values fall within standard physiological ranges.\n"
      "3. Click **Calculate Risk** to execute real-time XGBoost"
      " classification."
  )
  st.markdown("---")
  st.markdown(
      "*Note: This tool is intended to assist clinical decision-making and"
      " should be validated by professional diagnostics.*"
  )
