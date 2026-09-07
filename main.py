import glob
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# ---------------------------------------------------------------------------
# 1. Page Configuration & Professional Medical UI Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioLens - Hypertension Risk CDSS",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stSidebar { background-color: #ffffff; border-right: 1px solid #e5e7eb; }
    .stButton>button {
        width: 100%; background-color: #0284c7; color: white;
        font-weight: bold; border-radius: 8px; padding: 10px;
    }
    .stButton>button:hover { background-color: #0369a1; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 2. Categorical value mappers
# ---------------------------------------------------------------------------
def _map_gender_value(v):
  s = str(v).strip().lower()
  if s in ("f", "female", "woman"): return "Female"
  if s in ("m", "male", "man"): return "Male"
  if "female" in s: return "Female"
  if "male" in s: return "Male"
  return None

def _map_smoking_status_value(v):
  s = str(v).strip().lower()
  if s in ("never", "non-smoker", "nonsmoker", "non smoker"): return "Never"
  if s in ("former", "ex-smoker", "ex smoker", "past smoker", "quit"): return "Former"
  if s in ("current", "smoker", "active smoker"): return "Current"
  if "never" in s: return "Never"
  if "quit" in s or "ex-smoker" in s or "former" in s: return "Former"
  if "current" in s or "active" in s: return "Current"
  return None

def _map_yesno_value(v):
  s = str(v).strip().lower()
  if s in ("yes", "y", "true", "positive", "1"): return "Yes"
  if s in ("no", "n", "false", "negative", "0"): return "No"
  return None

def _map_bp_history_value(v):
  s = str(v).strip().lower()
  if s in ("normotensive", "normal"): return "Normotensive"
  if s in ("prehypertensive", "pre-hypertensive", "elevated", "borderline"): return "Prehypertensive"
  if s in ("hypertensive", "high"): return "Hypertensive"
  if "normo" in s: return "Normotensive"
  if "prehyp" in s or "elevated" in s or "borderline" in s: return "Prehypertensive"
  if "hyper" in s: return "Hypertensive"
  yn = _map_yesno_value(v)
  if yn == "Yes": return "Hypertensive"
  if yn == "No": return "Normotensive"
  return None

def _map_low_mod_high_value(v):
  s = str(v).strip().lower()
  if s in ("low", "mild"): return "Low"
  if s in ("moderate", "medium", "mod", "average"): return "Moderate"
  if s in ("high", "elevated", "severe"): return "High"
  return None

def _map_frequency_value(v):
  s = str(v).strip().lower()
  table = {
      "none": "None", "never": "None",
      "occasional": "Occasional", "occasionally": "Occasional",
      "rare": "Occasional", "rarely": "Occasional",
      "regular": "Regular", "regularly": "Regular",
      "frequent": "Frequent", "frequently": "Frequent",
      "daily": "Frequent", "heavy": "Frequent",
  }
  return table.get(s)

def _classify_column(col_name):
  name = col_name.lower()
  if "gender" in name or "sex" in name: return "gender"
  if "blood_pressure_history" in name or "bp_history" in name: return "bp_history"
  if "smok" in name: return "smoking"
  if "famil" in name or "history" in name or name == "fh": return "binary_yesno"
  return None

def _try_clean_mapping(col_name, uniques, mapper, order):
  mapped_pairs = {u: mapper(u) for u in uniques}
  values = list(mapped_pairs.values())
  if all(v is not None for v in values) and len(set(values)) == len(uniques) and set(values) <= set(order):
    label_to_raw = {label: u for u, label in mapped_pairs.items()}
    options = [lbl for lbl in order if lbl in label_to_raw]
    label_to_raw = {lbl: label_to_raw[lbl] for lbl in options}
    encode_map = {label_to_raw[lbl]: order.index(lbl) for lbl in options}
    return options, label_to_raw, encode_map
  return None

def _raw_value_fallback(col_name, uniques):
  sorted_u = sorted(uniques, key=str)
  label_to_raw = {f"{col_name} = {u}": u for u in sorted_u}
  encode_map = {u: i for i, u in enumerate(sorted_u)}
  return list(label_to_raw.keys()), label_to_raw, encode_map

def _best_categorical_mapping(col_name, uniques, candidates):
  for mapper, order in candidates:
    result = _try_clean_mapping(col_name, uniques, mapper, order)
    if result is not None: return result
  return _raw_value_fallback(col_name, uniques)

def _cat_result(options, label_to_raw, encode_map):
  return {"kind": "categorical", "options": options, "label_to_raw": label_to_raw, "encode_map": encode_map, "unit": ""}

def build_feature_encoder(col_name, series):
  tag = _classify_column(col_name)
  clean = series.dropna()
  is_object = not pd.api.types.is_numeric_dtype(clean)

  if is_object and tag is None and not clean.empty:
    numeric_try = pd.to_numeric(clean, errors="coerce")
    if numeric_try.notna().mean() > 0.9:
      is_object = False
      clean = numeric_try.dropna()

  if tag == "gender":
    uniques = list(pd.unique(clean))
    return _cat_result(*_best_categorical_mapping(col_name, uniques, [(_map_gender_value, ["Female", "Male"])]))

  if tag == "smoking" and (is_object or len(pd.unique(clean)) <= 5):
    uniques = list(pd.unique(clean))
    return _cat_result(*_best_categorical_mapping(col_name, uniques, [(_map_smoking_status_value, ["Never", "Former", "Current"]), (_map_yesno_value, ["No", "Yes"])]))

  if tag == "bp_history":
    uniques = list(pd.unique(clean))
    return _cat_result(*_best_categorical_mapping(col_name, uniques, [(_map_bp_history_value, ["Normotensive", "Prehypertensive", "Hypertensive"])]))

  if tag == "binary_yesno":
    uniques = list(pd.unique(clean))
    return _cat_result(*_best_categorical_mapping(col_name, uniques, [(_map_yesno_value, ["No", "Yes"])]))

  if is_object:
    uniques = list(pd.unique(clean))
    return _cat_result(*_best_categorical_mapping(col_name, uniques, [(_map_low_mod_high_value, ["Low", "Moderate", "High"]), (_map_frequency_value, ["None", "Occasional", "Regular", "Frequent"]), (_map_yesno_value, ["No", "Yes"])]))

  unit, step = "", 1.0
  name = col_name.lower()
  min_val = float(clean.min()) if not clean.empty else 0.0
  max_val = float(clean.max()) if not clean.empty else 100.0
  default_val = float(clean.median()) if not clean.empty else 0.0

  if "waist" in name: unit, min_val, max_val, step = "cm", 50.0, 170.0, 0.5
  elif "exercise" in name and "day" in name: unit, min_val, max_val, step = "days/week", 0.0, 7.0, 1.0
  elif "age" in name: unit, min_val, max_val, step = "Years", 1.0, 120.0, 1.0
  elif "salt" in name or "sodium" in name: unit, min_val, max_val, step = "g/day", 0.0, 30.0, 0.5
  elif "physical" in name or "activity" in name or "mvpa" in name: unit, min_val, max_val, step = "min/week", 0.0, 1000.0, 5.0
  elif "sleep" in name: unit, min_val, max_val, step = "h/night", 1.0, 16.0, 0.1
  elif "stress" in name: unit, min_val, max_val, step = "Scale (1-10)", 1.0, 10.0, 1.0
  elif "diastolic" in name or "dbp" in name: unit, min_val, max_val, step = "mmHg", 40.0, 150.0, 1.0
  elif "bp" in name or "systolic" in name or "sbp" in name: unit, min_val, max_val, step = "mmHg", 60.0, 250.0, 1.0
  elif "bmi" in name: unit, min_val, max_val, step = "kg/m²", 10.0, 60.0, 0.1
  elif "glucose" in name or "sugar" in name: unit, min_val, max_val, step = "mg/dL", 50.0, 500.0, 1.0
  elif "chol" in name: unit, min_val, max_val, step = "mg/dL", 100.0, 450.0, 1.0
  elif "heart" in name or "pulse" in name or "hr" in name: unit, min_val, max_val, step = "bpm", 40.0, 200.0, 1.0
  elif "alcohol" in name: unit, min_val, max_val, step = "g/day", 0.0, 200.0, 1.0
  elif "smok" in name: unit, min_val, max_val, step = "pack-years", 0.0, 100.0, 1.0

  if min_val >= max_val: min_val, max_val = 0.0, max(100.0, default_val + 1.0)
  default_val = max(min_val, min(max_val, default_val))
  return {"kind": "numeric", "unit": unit, "min_val": min_val, "max_val": max_val, "default_val": default_val, "step": step}

# ---------------------------------------------------------------------------
# 3. Target column detection
# ---------------------------------------------------------------------------
STRONG_TARGET_CANDIDATES = ["target", "output", "label", "diagnosis", "outcome"]
WEAK_TARGET_CANDIDATES = ["class", "htn", "hypertension", "risk", "disease", "y"]
FEATURE_DISQUALIFIERS = ["history", "family", "smoking", "activity", "salt", "sodium", "sleep", "stress", "alcohol", "age", "gender", "sex", "bmi", "glucose", "cholesterol", "heart_rate", "pulse", "waist", "exercise", "food"]

def detect_target_column(df):
  cols_lower = {c.lower(): c for c in df.columns}
  for cand in STRONG_TARGET_CANDIDATES + WEAK_TARGET_CANDIDATES:
    if cand in cols_lower: return cols_lower[cand]
  for cand in STRONG_TARGET_CANDIDATES:
    for lower_name, orig in cols_lower.items():
      if cand in lower_name: return orig
  for cand in WEAK_TARGET_CANDIDATES:
    for lower_name, orig in cols_lower.items():
      if cand in lower_name and not any(d in lower_name for d in FEATURE_DISQUALIFIERS): return orig
  return df.columns[-1]

# ---------------------------------------------------------------------------
# 4. Target encoding 
# ---------------------------------------------------------------------------
RISK_KEYWORD_ORDER = ["low", "moderate", "medium", "normal", "mild", "elevated", "high", "severe"]
POSITIVE_HINTS = ["yes", "high", "positive", "hypertensive", "present", "disease"]
NEGATIVE_HINTS = ["no", "low", "negative", "normal", "absent", "healthy"]

def _risk_rank(v):
  s = str(v).strip().lower()
  for i, word in enumerate(RISK_KEYWORD_ORDER):
    if word in s: return i
  return None

def _guess_binary_label(value, positive_hints, negative_hints):
  v = str(value).strip().lower()
  for hint in positive_hints:
    if hint in v: return "positive"
  for hint in negative_hints:
    if hint in v: return "negative"
  return None

def _score_unknown_class(u):
  g = _guess_binary_label(u, POSITIVE_HINTS, NEGATIVE_HINTS)
  return {"negative": 0, None: 1, "positive": 2}[g]

def encode_target(y_raw):
  uniques = list(pd.unique(y_raw.dropna()))
  if pd.api.types.is_numeric_dtype(y_raw):
    y_int = y_raw.fillna(0).astype(int)
    ordered_vals = sorted(pd.unique(y_int))
    remap = {v: i for i, v in enumerate(ordered_vals)}
    y_enc = y_int.map(remap)
    class_labels = [str(v) for v in ordered_vals]
  else:
    ranks = {u: _risk_rank(u) for u in uniques}
    if all(r is not None for r in ranks.values()) and len(set(ranks.values())) == len(uniques):
      ordered = sorted(uniques, key=lambda u: ranks[u])
    else:
      ordered = sorted(uniques, key=_score_unknown_class)
    remap = {v: i for i, v in enumerate(ordered)}
    y_enc = y_raw.map(remap)
    class_labels = [str(v) for v in ordered]

  y_enc = y_enc.fillna(0).astype(int)
  counts = y_enc.value_counts()
  class_balance = {class_labels[k]: int(counts.get(k, 0)) for k in range(len(class_labels))}
  return y_enc, class_labels, class_balance

def _ensure_all_numeric(X):
  X = X.copy()
  forced_cols = []
  for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
      forced_cols.append(col)
      codes, _ = pd.factorize(X[col])
      X[col] = codes
  X = X.fillna(0)
  return X, forced_cols

# ---------------------------------------------------------------------------
# 5. Load data & train model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_and_train():
  csv_files = glob.glob("*.csv")
  if not csv_files:
    st.error("⚠️ No CSV dataset found in your GitHub repository.")
    st.stop()

  df = pd.read_csv(csv_files[0], keep_default_na=False, na_values=[""])
  target_col = detect_target_column(df)

  X_raw = df.drop(columns=[target_col])
  y_raw = df[target_col]

  encoders = {col: build_feature_encoder(col, X_raw[col]) for col in X_raw.columns}

  X = pd.DataFrame(index=X_raw.index)
  for col in X_raw.columns:
    enc = encoders[col]
    if enc["kind"] == "categorical":
      X[col] = X_raw[col].map(enc["encode_map"])
    else:
      X[col] = pd.to_numeric(X_raw[col], errors="coerce")
  X = X.fillna(X.median(numeric_only=True)).fillna(0)
  X, forced_cols = _ensure_all_numeric(X)

  y, class_labels, class_balance = encode_target(y_raw)
  n_classes = len(class_labels)

  counts_by_code = y.value_counts().to_dict()
  total = len(y)
  class_weight = {code: total / (n_classes * cnt) for code, cnt in counts_by_code.items()}
  sample_weight = y.map(class_weight)

  eval_metric = "mlogloss" if n_classes > 2 else "logloss"
  model = xgb.XGBClassifier(
      eval_metric=eval_metric, random_state=42, n_estimators=300, max_depth=5, learning_rate=0.1,
  )
  try:
    model.fit(X, y, sample_weight=sample_weight)
  except ValueError as e:
    bad = {c: str(X[c].dtype) for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])}
    raise ValueError(f"{e} | Non-numeric columns still present: {bad}") from e

  train_proba_all = model.predict_proba(X)
  high_risk_idx = n_classes - 1
  train_risk_proba = train_proba_all[:, high_risk_idx]

  diagnostics = {
      "target_col": target_col, "class_labels": class_labels, "class_balance": class_balance,
      "n_classes": n_classes, "forced_cols": forced_cols, "train_proba_min": float(train_risk_proba.min()),
      "train_proba_max": float(train_risk_proba.max()), "train_proba_mean": float(train_risk_proba.mean()),
      "schema": {c: (encoders[c]["options"] if encoders[c]["kind"] == "categorical" else f"numeric ({encoders[c]['unit'] or 'no unit'}, {encoders[c]['min_val']:.0f}-{encoders[c]['max_val']:.0f})") for c in X.columns},
  }
  return model, X.columns, encoders, diagnostics

# ---------------------------------------------------------------------------
# 6. App layout with SI Unit Translation
# ---------------------------------------------------------------------------
st.title("🩺 CardioLens: Hypertension Risk CDSS")
st.markdown("Clinical Decision Support System for evaluating patient cardiovascular risk profiles.")
st.markdown("---")

try:
  model, feature_names, encoders, diagnostics = load_and_train()
except Exception as e:
  st.error(f"Error loading model or data: {e}")
  st.stop()

class_labels = diagnostics["class_labels"]
n_classes = diagnostics["n_classes"]

st.sidebar.header("🫀 Patient Vitals & History")
st.sidebar.markdown("Adjust parameters below to simulate patient risk assessment.")

input_display = {}
model_input = {}

for col in feature_names:
  enc = encoders[col]
  name_lower = col.lower()

  if enc["kind"] == "categorical":
    # ---------------------------------------------------------
    # UI Overrides: Display SI Units, output hidden Categories
    # ---------------------------------------------------------
    if "salt" in name_lower:
      val = st.sidebar.number_input(f"{col.replace('_', ' ')} (g/day)", 0.0, 30.0, 5.0, 1.0)
      mapped = "Low" if val < 5 else ("Moderate" if val <= 10 else "High")
      input_display[col] = f"{val} g/day ({mapped})"
      model_input[col] = enc["encode_map"].get(mapped, 0)
      
    elif "physical" in name_lower or "activity" in name_lower:
      val = st.sidebar.number_input(f"{col.replace('_', ' ')} (min/week)", 0.0, 1000.0, 150.0, 10.0)
      mapped = "Low" if val < 90 else ("Moderate" if val <= 150 else "High")
      input_display[col] = f"{val} min/week ({mapped})"
      model_input[col] = enc["encode_map"].get(mapped, 0)
      
    elif "stress" in name_lower:
      val = st.sidebar.number_input(f"{col.replace('_', ' ')} (Scale 1-10)", 1.0, 10.0, 5.0, 1.0)
      mapped = "Low" if val <= 3 else ("Moderate" if val <= 7 else "High")
      input_display[col] = f"{val} ({mapped})"
      model_input[col] = enc["encode_map"].get(mapped, 0)
      
    elif "alcohol" in name_lower:
      val = st.sidebar.number_input(f"{col.replace('_', ' ')} (g/day)", 0.0, 200.0, 0.0, 5.0)
      mapped = "None" if val == 0 else ("Occasional" if val <= 20 else "Frequent")
      val_enc = enc["encode_map"].get(mapped)
      model_input[col] = val_enc if val_enc is not None else 0
      input_display[col] = f"{val} g/day ({mapped})"
      
    elif "smok" in name_lower:
      label = st.sidebar.selectbox(f"{col.replace('_', ' ')} (Status)", ["Never (0 pack-years)", "Former", "Current"])
      mapped = "No" if "Never" in label else "Yes"
      input_display[col] = label
      model_input[col] = enc["encode_map"].get(mapped, 0)
      
    else:
      label = st.sidebar.selectbox(col.replace("_", " "), options=enc["options"])
      input_display[col] = label
      model_input[col] = enc["encode_map"][enc["label_to_raw"][label]]
  else:
    label_text = f"{col.replace('_', ' ')} ({enc['unit']})" if enc["unit"] else col.replace("_", " ")
    val = st.sidebar.number_input(
        label=label_text, value=float(enc["default_val"]),
        min_value=float(enc["min_val"]), max_value=float(enc["max_val"]),
        step=float(enc.get("step", 1.0)),
    )
    input_display[col] = val
    model_input[col] = val

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("Calculate Hypertension Risk")

col1, col2 = st.columns([2, 1])

with col1:
  st.subheader("📋 Patient Clinical Summary")
  st.dataframe(pd.DataFrame([input_display]), use_container_width=True)

  if predict_btn:
    model_df = pd.DataFrame([model_input])[feature_names]
    proba = model.predict_proba(model_df)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = class_labels[pred_idx]
    pred_prob = float(proba[pred_idx])

    st.markdown("### 📊 Assessment Report")
    if pred_idx == 0:
      st.success(
          f"### 🟢 {pred_label}\n"
          f"**Model confidence:** `{pred_prob * 100:.1f}%`\n\n"
          "*Clinical Recommendation:* Maintain standard healthy lifestyle habits and annual check-ups."
      )
    elif pred_idx == n_classes - 1:
      st.error(
          f"### 🔴 {pred_label}\n"
          f"**Model confidence:** `{pred_prob * 100:.1f}%`\n\n"
          "*Clinical Recommendation:* Immediate lifestyle intervention, ECG evaluation, and secondary screening recommended."
      )
    else:
      st.warning(
          f"### 🟠 {pred_label}\n"
          f"**Model confidence:** `{pred_prob * 100:.1f}%`\n\n"
          "*Clinical Recommendation:* Closer monitoring and preventive lifestyle changes recommended."
      )

    st.markdown("**Full probability breakdown:**")
    prob_series = pd.Series({lbl: p for lbl, p in zip(class_labels, proba)})
    st.bar_chart(prob_series)

with col2:
  st.markdown("### ℹ️ CDSS Guide")
  st.info(
      "**How to use:**\n"
      "1. Configure lifestyle metrics and history using standard SI units.\n"
      "2. The app translates SI values to match the dataset format automatically.\n"
      "3. Click **Calculate Risk** to run the XGBoost classifier."
  )
