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


# Helper function to assign standard clinical units, ranges, and types based on features
def get_feature_meta(col_name, series):
  col_lower = col_name.lower()
  unit = ""
  min_val = float(series.min()) if not series.empty else 0.0
  max_val = float(series.max()) if not series.empty else 100.0
  default_val = float(series.median()) if not series.empty else 0.0
  input_type = "number"
  options = []

  if "gender" in col_lower or "sex" in col_lower:
    input_type = "selectbox"
    options = ["Male", "Female"]
    default_val = "Male"
  elif "age" in col_lower:
    unit = "Years"
    min_val, max_val = 1.0, 120.0
  elif "salt" in col_lower or "sodium" in col_lower:
    unit = "g/day"
    min_val, max_val = 0.0, 30.0
  elif "physical" in col_lower or "activity" in col_lower or "mvpa" in col_lower:
    unit = "min/week"
    min_val, max_val = 0.0, 1000.0
  elif "sleep" in col_lower:
    unit = "h/night"
    min_val, max_val = 1.0, 16.0
  elif "stress" in col_lower:
    unit = "Scale (1-10)"
    min_val, max_val = 1.0, 10.0
  elif "family" in col_lower or "history" in col_lower or "fh" in col_lower:
    input_type = "selectbox"
    options = ["No (0)", "Yes (1)"]
    default_val = "No (0)"
  elif "bp" in col_lower or "systolic" in col_lower or "sbp" in col_lower:
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
  elif "smoke" in col_lower or "smoking" in col_lower:
    input_type = "selectbox"
    options = ["Never (0)", "Former (1)", "Current (2)"]
    default_val = "Never (0)"
  elif "alcohol" in col_lower:
    unit = "g/day"
    min_val, max_val = 0.0, 200.0

  return unit, min_val, max_val, default_val, input_type, options


@st.cache_resource
def load_and_train():
  csv_files = glob.glob("*.csv")
  if not csv_files:
    st.error("⚠️ No CSV dataset found in your GitHub repository.")
    st.stop()

  df = pd.read_csv(csv_files[0])

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

  # Preprocess columns for training
  for col in X.columns:
    if X[col].dtype == "object":
      X[col] = X[col].astype("category").cat.codes
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

# Interactive Sidebar with Medical Units, Gender, and Standardized Standards
st.sidebar.header("🫀 Patient Vitals & History")
st.sidebar.markdown(
    "Adjust parameters below to simulate patient risk assessment."
)

input_data = {}
raw_input_for_model = {}

for col in feature_names:
  unit, min_v, max_v, default_v, input_type, options = get_feature_meta(
      col, X_data[col]
  )
  label_text = f"{col} ({unit})" if unit else f"{col}"

  if input_type == "selectbox":
    selected_val = st.sidebar.selectbox(label=label_text, options=options)
    input_data[col] = selected_val
    # Map back to numeric for model prediction
    if "Male" in options or "Female" in options:
      raw_input_for_model[col] = 1 if selected_val == "Male" else 0
    elif "Never" in options:
      raw_input_for_model[col] = (
          0 if "Never" in selected_val else (1 if "Former" in selected_val else 2)
      )
    else:
      raw_input_for_model[col] = 1 if "Yes" in selected_val else 0
  else:
    if min_v >= max_v:
      min_v, max_v = 0.0, 100.0

    val = st.sidebar.number_input(
        label=label_text,
        value=float(default_v),
        min_value=float(min_v),
        max_value=float(max_v),
        step=1.0,
    )
    input_data[col] = val
    raw_input_for_model[col] = val

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("Calculate Hypertension Risk")

# Main Content Layout
col1, col2 = st.columns([2, 1])

with col1:
  st.subheader("📋 Patient Clinical Summary")
  input_df = pd.DataFrame([input_data])
  st.dataframe(input_df, use_container_width=True)

  if predict_btn:
    model_input_df = pd.DataFrame([raw_input_for_model])
    probabilities = model.predict_proba(model_input_df)[0]

    if len(probabilities) > 1:
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
      "1. Configure gender, lifestyle metrics, and vitals with standard clinical"
      " units in the sidebar.\n"
      "2. Ensure parameters comply with standard physiological ranges.\n"
      "3. Click **Calculate Risk** to execute real-time XGBoost"
      " classification."
  )
  st.markdown("---")
  st.markdown(
      "*Note: This tool is intended to assist clinical decision-making and"
      " should be validated by professional diagnostics.*"
  )
