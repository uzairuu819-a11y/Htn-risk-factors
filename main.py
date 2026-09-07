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
# 2. Encoding helpers
# ---------------------------------------------------------------------------
# WHY THIS SECTION EXISTS:
# The previous version encoded categorical columns during training using
# pandas' `.cat.codes` (alphabetical order), then re-encoded UI selections
# using HARD-CODED numbers (e.g. Male=1, Female=0) that assumed a specific
# order. If the CSV's actual values didn't match that assumption, the model
# was fed different numbers at prediction time than it saw in training -
# which is the most likely reason the app always reported "Low risk"
# regardless of input.
#
# FIX: build ONE explicit mapping per categorical column directly from the
# real values in the training data, and reuse that exact same mapping both
# when training and when turning a UI selection back into a model input.


def _classify_column(col_name):
  """Tag a column by name so we know which semantic encoder to use."""
  name = col_name.lower()
  if "gender" in name or "sex" in name:
    return "gender"
  # Check this BEFORE the generic "history" catch-all below - a column like
  # "Blood_Pressure_History" contains "history" but needs its own 2/3-level
  # handling, not the generic Yes/No one.
  if "blood_pressure_history" in name or "bp_history" in name:
    return "bp_history"
  if "smok" in name:
    return "smoking"
  if "famil" in name or "history" in name or name == "fh":
    return "binary_yesno"
  return None


def _map_gender_value(v):
  s = str(v).strip().lower()
  if s in ("f", "female", "woman"):
    return "Female"
  if s in ("m", "male", "man"):
    return "Male"
  if "female" in s:
    return "Female"
  if "male" in s:
    return "Male"
  return None


def _map_smoking_value(v):
  s = str(v).strip().lower()
  if s in ("never", "non-smoker", "nonsmoker", "non smoker"):
    return "Never"
  if s in ("former", "ex-smoker", "ex smoker", "past smoker", "quit"):
    return "Former"
  if s in ("current", "smoker", "active smoker"):
    return "Current"
  if "never" in s:
    return "Never"
  if "quit" in s or "ex-smoker" in s or "former" in s:
    return "Former"
  if "current" in s or "active" in s:
    return "Current"
  return None


def _map_yesno_value(v):
  s = str(v).strip().lower()
  if s in ("yes", "y", "true", "positive"):
    return "Yes"
  if s in ("no", "n", "false", "negative"):
    return "No"
  return None


def _map_bp_history_value(v):
  s = str(v).strip().lower()
  if s in ("normotensive", "normal", "0"):
    return "Normotensive"
  if s in ("prehypertensive", "pre-hypertensive", "elevated", "borderline"):
    return "Prehypertensive"
  if s in ("hypertensive", "high", "1"):
    return "Hypertensive"
  if "normo" in s:
    return "Normotensive"
  if "prehyp" in s or "elevated" in s or "borderline" in s:
    return "Prehypertensive"
  if "hyper" in s:
    return "Hypertensive"
  yn = _map_yesno_value(v)
  if yn == "Yes":
    return "Hypertensive"
  if yn == "No":
    return "Normotensive"
  return None


def _categorical_from_mapper(col_name, uniques, mapper, order):
  """
  Try to map every unique raw value to a known label via `mapper`. Only use
  that semantic mapping if EVERY value maps unambiguously to a DISTINCT
  label in `order` - otherwise fall back to showing the raw values verbatim,
  so we never silently guess wrong on an unfamiliar coding scheme.
  """
  mapped_pairs = {u: mapper(u) for u in uniques}
  values = list(mapped_pairs.values())
  if (
      all(v is not None for v in values)
      and len(set(values)) == len(uniques)
      and set(values) <= set(order)
  ):
    label_to_raw = {label: u for u, label in mapped_pairs.items()}
    options = [lbl for lbl in order if lbl in label_to_raw]
    label_to_raw = {lbl: label_to_raw[lbl] for lbl in options}
    encode_map = {label_to_raw[lbl]: order.index(lbl) for lbl in options}
    return options, label_to_raw, encode_map

  # Fallback: show the literal raw values so nothing is guessed incorrectly.
  sorted_u = sorted(uniques, key=str)
  label_to_raw = {f"{col_name} = {u}": u for u in sorted_u}
  encode_map = {u: i for i, u in enumerate(sorted_u)}
  return list(label_to_raw.keys()), label_to_raw, encode_map


def build_feature_encoder(col_name, series):
  """
  Inspect a raw training column and return everything needed to (a) turn it
  into a numeric feature for XGBoost and (b) render a matching UI widget
  whose selections map back to EXACTLY the same numbers.
  """
  tag = _classify_column(col_name)
  clean = series.dropna()
  # Anything that isn't a numeric dtype is treated as text/categorical.
  # (pandas 3.x uses its own StringDtype - shown as "str" - for CSV text
  # columns instead of the classic "object" dtype, so we can't just check
  # for `== object` anymore; is_numeric_dtype is the robust check.)
  is_object = not pd.api.types.is_numeric_dtype(clean)

  # A numeric clinical value stored as text (e.g. "120" as a string) should
  # be treated as numeric, not as an arbitrary category.
  if is_object and tag is None and not clean.empty:
    numeric_try = pd.to_numeric(clean, errors="coerce")
    if numeric_try.notna().mean() > 0.9:
      is_object = False
      clean = numeric_try.dropna()

  if tag == "gender":
    uniques = list(pd.unique(clean))
    options, label_to_raw, encode_map = _categorical_from_mapper(
        col_name, uniques, _map_gender_value, ["Female", "Male"])
    return {"kind": "categorical", "options": options,
            "label_to_raw": label_to_raw, "encode_map": encode_map, "unit": ""}

  if tag == "smoking":
    uniques = list(pd.unique(clean))
    # Only treat as Never/Former/Current if it's genuinely low-cardinality;
    # a numeric column with many distinct values is more likely pack-years.
    if is_object or len(uniques) <= 5:
      options, label_to_raw, encode_map = _categorical_from_mapper(
          col_name, uniques, _map_smoking_value, ["Never", "Former", "Current"])
      return {"kind": "categorical", "options": options,
              "label_to_raw": label_to_raw, "encode_map": encode_map, "unit": ""}
    # else: fall through to numeric handling below (pack-years).

  if tag == "bp_history":
    uniques = list(pd.unique(clean))
    order = ["Normotensive", "Prehypertensive", "Hypertensive"]
    options, label_to_raw, encode_map = _categorical_from_mapper(
        col_name, uniques, _map_bp_history_value, order)
    return {"kind": "categorical", "options": options,
            "label_to_raw": label_to_raw, "encode_map": encode_map, "unit": ""}

  if tag == "binary_yesno":
    uniques = list(pd.unique(clean))
    options, label_to_raw, encode_map = _categorical_from_mapper(
        col_name, uniques, _map_yesno_value, ["No", "Yes"])
    return {"kind": "categorical", "options": options,
            "label_to_raw": label_to_raw, "encode_map": encode_map, "unit": ""}

  if is_object:
    uniques = sorted(pd.unique(clean), key=str)
    label_to_raw = {str(u): u for u in uniques}
    encode_map = {u: i for i, u in enumerate(uniques)}
    return {"kind": "categorical", "options": list(label_to_raw.keys()),
            "label_to_raw": label_to_raw, "encode_map": encode_map, "unit": ""}

  # --- Numeric column ------------------------------------------------------
  unit = ""
  name = col_name.lower()
  min_val = float(clean.min()) if not clean.empty else 0.0
  max_val = float(clean.max()) if not clean.empty else 100.0
  default_val = float(clean.median()) if not clean.empty else 0.0

  if "age" in name:
    unit, min_val, max_val = "Years", 1.0, 120.0
  elif "salt" in name or "sodium" in name:
    unit, min_val, max_val = "g/day", 0.0, 30.0
  elif "physical" in name or "activity" in name or "mvpa" in name:
    unit, min_val, max_val = "min/week", 0.0, 1000.0
  elif "sleep" in name:
    unit, min_val, max_val = "h/night", 1.0, 16.0
  elif "stress" in name:
    unit, min_val, max_val = "Scale (1-10)", 1.0, 10.0
  elif "diastolic" in name or "dbp" in name:
    unit, min_val, max_val = "mmHg", 40.0, 150.0
  elif "bp" in name or "systolic" in name or "sbp" in name:
    unit, min_val, max_val = "mmHg", 60.0, 250.0
  elif "bmi" in name:
    unit, min_val, max_val = "kg/m²", 10.0, 60.0
  elif "glucose" in name or "sugar" in name:
    unit, min_val, max_val = "mg/dL", 50.0, 500.0
  elif "chol" in name:
    unit, min_val, max_val = "mg/dL", 100.0, 450.0
  elif "heart" in name or "pulse" in name or "hr" in name:
    unit, min_val, max_val = "bpm", 40.0, 200.0
  elif "alcohol" in name:
    unit, min_val, max_val = "g/day", 0.0, 200.0
  elif "smok" in name:
    unit, min_val, max_val = "pack-years", 0.0, 100.0

  if min_val >= max_val:
    min_val, max_val = 0.0, max(100.0, default_val + 1.0)
  default_val = max(min_val, min(max_val, default_val))

  return {"kind": "numeric", "unit": unit, "min_val": min_val,
          "max_val": max_val, "default_val": default_val}


# ---------------------------------------------------------------------------
# 3. Target column + "which class means high risk" detection
# ---------------------------------------------------------------------------
# Unambiguous - these words essentially only ever appear on an outcome
# column, so a substring match is safe.
STRONG_TARGET_CANDIDATES = ["target", "output", "label", "diagnosis", "outcome"]
# Ambiguous - these words can also appear inside risk-FACTOR column names
# (e.g. "Family_History_Hypertension", "Blood_Pressure_History"), so they
# are only substring-matched on columns that don't look like a risk-factor
# column (see FEATURE_DISQUALIFIERS below).
WEAK_TARGET_CANDIDATES = ["class", "htn", "hypertension", "risk", "disease", "y"]
FEATURE_DISQUALIFIERS = [
    "history", "family", "smoking", "activity", "salt", "sodium", "sleep",
    "stress", "alcohol", "age", "gender", "sex", "bmi", "glucose",
    "cholesterol", "heart_rate", "pulse",
]
POSITIVE_HINTS = ["yes", "high", "positive", "hypertensive", "present", "disease"]
NEGATIVE_HINTS = ["no", "low", "negative", "normal", "absent", "healthy"]


def detect_target_column(df):
  cols_lower = {c.lower(): c for c in df.columns}

  # 1) exact column-name match against any candidate.
  for cand in STRONG_TARGET_CANDIDATES + WEAK_TARGET_CANDIDATES:
    if cand in cols_lower:
      return cols_lower[cand]

  # 2) substring match for unambiguous candidates.
  for cand in STRONG_TARGET_CANDIDATES:
    for lower_name, orig in cols_lower.items():
      if cand in lower_name:
        return orig

  # 3) substring match for ambiguous candidates, but skip columns that look
  #    like a risk-FACTOR (history/lifestyle) column rather than the outcome.
  for cand in WEAK_TARGET_CANDIDATES:
    for lower_name, orig in cols_lower.items():
      if cand in lower_name and not any(d in lower_name for d in FEATURE_DISQUALIFIERS):
        return orig

  # 4) fallback: last column, as before.
  return df.columns[-1]


def _guess_binary_label(value, positive_hints, negative_hints):
  v = str(value).strip().lower()
  for hint in positive_hints:
    if hint in v:
      return "positive"
  for hint in negative_hints:
    if hint in v:
      return "negative"
  return None


def encode_target(y_raw):
  """
  Returns (y_encoded, positive_label, class_balance). `positive_label` is the
  RAW value in y_raw that represents "high risk" (encoded as 1), so it can be
  shown to the clinician for verification.
  """
  uniques = list(pd.unique(y_raw.dropna()))

  if pd.api.types.is_numeric_dtype(y_raw) and set(uniques) <= {0, 1}:
    y_enc = y_raw.fillna(0).astype(int)
    return y_enc, 1, y_enc.value_counts().to_dict()

  pos_val = None
  for u in uniques:
    if _guess_binary_label(u, POSITIVE_HINTS, NEGATIVE_HINTS) == "positive":
      pos_val = u
      break

  if pos_val is not None:
    y_enc = y_raw.apply(lambda v: 1 if v == pos_val else 0)
    return y_enc, pos_val, y_enc.value_counts().to_dict()

  if pd.api.types.is_numeric_dtype(y_raw):
    y_enc = y_raw.fillna(0).astype(int)
    return y_enc, y_enc.max(), y_enc.value_counts().to_dict()

  # Two unrecognised string categories: assume the minority class is
  # "high risk" (disease is usually the minority in screening data).
  # Verify this against the Diagnostics panel.
  counts = y_raw.value_counts()
  pos_val = counts.idxmin()
  y_enc = y_raw.apply(lambda v: 1 if v == pos_val else 0)
  return y_enc, pos_val, y_enc.value_counts().to_dict()


# ---------------------------------------------------------------------------
# 4. Load data & train model (cached)
# ---------------------------------------------------------------------------
def _ensure_all_numeric(X):
  """
  Belt-and-braces safety net: guarantee every column reaching XGBoost is
  numeric. If a column still contains an unmapped/unexpected value (any
  value our encoders didn't recognise) this force-encodes it instead of
  crashing, and reports which column(s) needed it so you can double-check
  that field's encoding.
  """
  X = X.copy()
  forced_cols = []
  for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
      forced_cols.append(col)
      codes, _ = pd.factorize(X[col])
      X[col] = codes
  X = X.fillna(0)
  return X, forced_cols


@st.cache_resource
def load_and_train():
  csv_files = glob.glob("*.csv")
  if not csv_files:
    st.error("⚠️ No CSV dataset found in your GitHub repository.")
    st.stop()

  df = pd.read_csv(csv_files[0])
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

  # Guaranteed fallback - training can never crash on a stray string value.
  X, forced_cols = _ensure_all_numeric(X)

  y, positive_label, class_balance = encode_target(y_raw)

  # Hypertension datasets are usually imbalanced (more "no HTN" than "HTN").
  # Without correcting for that, XGBoost tends to just always predict the
  # majority class - which looks exactly like "always Low risk" regardless
  # of input. scale_pos_weight is the standard fix.
  neg_count = int((y == 0).sum())
  pos_count = int((y == 1).sum())
  scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

  model = xgb.XGBClassifier(
      eval_metric="logloss", random_state=42, n_estimators=200, max_depth=4,
      scale_pos_weight=scale_pos_weight,
  )
  try:
    model.fit(X, y)
  except ValueError as e:
    bad = {c: str(X[c].dtype) for c in X.columns
           if not pd.api.types.is_numeric_dtype(X[c])}
    raise ValueError(f"{e} | Non-numeric columns still present: {bad}") from e

  # Sanity check: what's the actual spread of predicted risk across the
  # model's own training data? If even the highest-risk-looking real patient
  # in your training set gets a low probability, the target/label direction
  # is almost certainly wrong rather than anything in the sidebar inputs.
  train_proba_all = model.predict_proba(X)
  classes_ = list(model.classes_)
  risk_col = classes_.index(1) if 1 in classes_ else int(np.argmax(classes_))
  train_risk_proba = train_proba_all[:, risk_col]

  diagnostics = {
      "target_col": target_col,
      "positive_label": positive_label,
      "class_balance": class_balance,
      "n_classes": int(y.nunique()),
      "forced_cols": forced_cols,
      "scale_pos_weight": round(scale_pos_weight, 2),
      "train_proba_min": float(train_risk_proba.min()),
      "train_proba_max": float(train_risk_proba.max()),
      "train_proba_mean": float(train_risk_proba.mean()),
      "schema": {c: (encoders[c]["options"] if encoders[c]["kind"] == "categorical"
                      else f"numeric ({encoders[c]['unit'] or 'no unit'})")
                 for c in X.columns},
  }
  return model, X.columns, encoders, diagnostics


# ---------------------------------------------------------------------------
# 5. App layout
# ---------------------------------------------------------------------------
st.title("🩺 CardioLens: Hypertension Risk CDSS")
st.markdown(
    "Clinical Decision Support System for evaluating patient cardiovascular risk profiles."
)
st.markdown("---")

try:
  model, feature_names, encoders, diagnostics = load_and_train()
except Exception as e:
  st.error(f"Error loading model or data: {e}")
  st.stop()

st.sidebar.header("🫀 Patient Vitals & History")
st.sidebar.markdown("Adjust parameters below to simulate patient risk assessment.")

input_display = {}
model_input = {}

for col in feature_names:
  enc = encoders[col]
  if enc["kind"] == "categorical":
    label = st.sidebar.selectbox(col, options=enc["options"])
    input_display[col] = label
    model_input[col] = enc["encode_map"][enc["label_to_raw"][label]]
  else:
    label_text = f"{col} ({enc['unit']})" if enc["unit"] else col
    val = st.sidebar.number_input(
        label=label_text,
        value=float(enc["default_val"]),
        min_value=float(enc["min_val"]),
        max_value=float(enc["max_val"]),
        step=1.0,
    )
    input_display[col] = val
    model_input[col] = val

st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "High-risk decision threshold",
    min_value=0.05, max_value=0.95, value=0.5, step=0.05,
    help="Probability above which a patient is flagged high risk. Lower it "
         "if the model seems to under-flag risk on your dataset.",
)
predict_btn = st.sidebar.button("Calculate Hypertension Risk")

col1, col2 = st.columns([2, 1])

with col1:
  st.subheader("📋 Patient Clinical Summary")
  st.dataframe(pd.DataFrame([input_display]), use_container_width=True)

  if predict_btn:
    model_df = pd.DataFrame([model_input])[feature_names]
    proba = model.predict_proba(model_df)[0]
    classes = list(model.classes_)

    if 1 in classes:
      risk_idx = classes.index(1)
    elif len(classes) == len(proba):
      risk_idx = int(np.argmax(classes))
    else:
      risk_idx = len(proba) - 1
    probability = float(proba[risk_idx])

    st.markdown("### 📊 Assessment Report")
    if probability >= threshold:
      st.error(
          f"### 🔴 High Risk of Hypertension Detected\n"
          f"**Model probability:** `{probability * 100:.1f}%` "
          f"(threshold {threshold * 100:.0f}%)\n\n"
          "*Clinical Recommendation:* Immediate lifestyle intervention, ECG "
          "evaluation, and secondary screening recommended."
      )
    else:
      st.success(
          f"### 🟢 Low Risk of Hypertension\n"
          f"**Model probability:** `{probability * 100:.1f}%` "
          f"(threshold {threshold * 100:.0f}%)\n\n"
          "*Clinical Recommendation:* Maintain standard healthy lifestyle "
          "habits and annual check-ups."
      )
    st.progress(probability)

    with st.expander("🔍 Debug: exact values sent to the model"):
      st.write(model_df)

with col2:
  st.markdown("### ℹ️ CDSS Guide")
  st.info(
      "**How to use:**\n"
      "1. Configure gender, lifestyle metrics, and vitals in the sidebar.\n"
      "2. Ensure parameters comply with standard physiological ranges.\n"
      "3. Click **Calculate Risk** to run the XGBoost classifier.\n"
      "4. If risk seems stuck on one result, open **Model diagnostics** "
      "below first."
  )
  with st.expander("🛠️ Model diagnostics", expanded=True):
    st.write(f"**Target column detected:** `{diagnostics['target_col']}`")
    st.write(
        f"**Raw value treated as 'high risk' (encoded 1):** "
        f"`{diagnostics['positive_label']}`"
    )
    st.write("**Class balance in training data (0 = low risk, 1 = high risk):**")
    st.write(diagnostics["class_balance"])
    st.write(
        f"**Imbalance correction (scale_pos_weight):** "
        f"`{diagnostics['scale_pos_weight']}`"
    )
    st.write(
        f"**Predicted risk probability across your OWN training data:** "
        f"min `{diagnostics['train_proba_min']*100:.1f}%`, "
        f"max `{diagnostics['train_proba_max']*100:.1f}%`, "
        f"mean `{diagnostics['train_proba_mean']*100:.1f}%`"
    )
    if diagnostics["train_proba_max"] < 0.5:
      st.error(
          "⚠️ Even the highest-risk patient in your OWN training data never "
          "crosses 50%. This means the target column or the 'high risk' "
          "label direction is very likely wrong - it is not something you "
          "can fix from the sidebar. Share your target column's exact name "
          "and its distinct values and I'll fix the mapping directly."
      )
    st.write("**Detected column types/units:**")
    st.write(diagnostics["schema"])
    if diagnostics.get("forced_cols"):
      st.warning(
          "These columns had values that didn't match the expected category "
          f"mapping and were auto-encoded as a fallback: {diagnostics['forced_cols']}. "
          "Their sidebar widget may not reflect true clinical meaning - tell me "
          "the column name and its raw values if you want this cleaned up properly."
      )
    if diagnostics["n_classes"] < 2:
      st.warning(
          "Only one class was found in the target column - the model "
          "cannot distinguish risk levels. Check that the correct target "
          "column was detected above."
      )
    importances = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    st.write("**Feature importance:**")
    st.bar_chart(importances)
  st.markdown("---")
  st.markdown(
      "*Note: This tool is intended to support clinical decision-making and "
      "must be validated against professional diagnosis before use in "
      "patient care.*"
  )
