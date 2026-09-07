import glob
import math
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# 1. Page Configuration & Light Blue Theme UI/UX Styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CardioLens - Risk CDSS", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
    /* Light blue color theme background */
    .stApp {
        background-color: #EBF8FF;
        background-image: url("data:image/svg+xml,%3Csvg width='400' height='400' viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M200 250c-50-50-80-70-80-100 0-30 20-50 50-50 20 0 30 15 30 15s10-15 30-15c30 0 50 20 50 50 0 30-30 50-80 100z' stroke='%230284C7' stroke-width='2' fill='none' opacity='0.05'/%3E%3Cpath d='M0 200h100l20-40 40 100 30-80 20 20h190' stroke='%230369A1' stroke-width='2' fill='none' opacity='0.06'/%3E%3C/svg%3E");
        background-attachment: fixed;
        color: #1E293B;
    }
    
    /* Hide Streamlit default top toolbar, hamburger menu, footer, and manage app badge */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stToolbar"] {display: none !important;}
    .viewerBadge_container__1QSob {display: none !important;}
    
    div[data-baseweb="tab"] { font-weight: 600; }
    .stButton>button {
        width: 100%; background-color: #0284C7; color: white;
        font-weight: 600; border-radius: 8px; padding: 12px; border: none;
    }
    .stButton>button:hover { background-color: #0369A1; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. ASCVD Calculation & Visualization
# ---------------------------------------------------------------------------
def calculate_ascvd(age, sex, race, tc, hdl, sbp, meds, smoker, diab):
    try:
        ln_age, ln_tc, ln_hdl, ln_sbp = math.log(age), math.log(tc), math.log(hdl), math.log(sbp)
        if sex == 'Male' and race == 'White/Other':
            c = {'a': 12.344, 'a2': 0, 'tc': 11.853, 'atc': -2.664, 'h': -7.990, 'ah': 1.769, 'st': 1.797, 'ast': 0, 'su': 1.764, 'asu': 0, 'sm': 7.837, 'asm': -1.795, 'd': 0.658, 's10': 0.9144, 'mean': 61.18}
        elif sex == 'Male' and race == 'African American':
            c = {'a': 2.469, 'a2': 0, 'tc': 0.302, 'atc': 0, 'h': -0.307, 'ah': 0, 'st': 1.916, 'ast': 0, 'su': 1.809, 'asu': 0, 'sm': 0.549, 'asm': 0, 'd': 0.645, 's10': 0.8954, 'mean': 19.54}
        elif sex == 'Female' and race == 'White/Other':
            c = {'a': -29.799, 'a2': 4.884, 'tc': 13.540, 'atc': -3.114, 'h': -13.578, 'ah': 3.149, 'st': 2.019, 'ast': 0, 'su': 1.957, 'asu': 0, 'sm': 7.574, 'asm': -1.665, 'd': 0.661, 's10': 0.9665, 'mean': -29.18}
        else:
            c = {'a': 17.114, 'a2': 0, 'tc': 0.940, 'atc': 0, 'h': -18.920, 'ah': 4.475, 'st': 29.291, 'ast': -6.432, 'su': 27.820, 'asu': -6.087, 'sm': 0.691, 'asm': 0, 'd': 0.874, 's10': 0.9533, 'mean': 86.61}

        sm = (c['a']*ln_age) + (c['a2']*(ln_age**2)) + (c['tc']*ln_tc) + (c['atc']*ln_age*ln_tc) + (c['h']*ln_hdl) + (c['ah']*ln_age*ln_hdl)
        sm += (c['st']*ln_sbp + c['ast']*ln_age*ln_sbp) if meds else (c['su']*ln_sbp + c['asu']*ln_age*ln_sbp)
        sm += (c['sm'] + c['asm']*ln_age) if smoker else 0
        sm += c['d'] if diab else 0

        risk = (1 - (c['s10'] ** math.exp(sm - c['mean']))) * 100
        return min(max(round(risk, 1), 0), 100)
    except:
        return 0.0

def make_gauge(val, title, thresholds, colors):
    is_percentage = "ASCVD" in title or "Cardiovascular" in title
    max_val = 100 if is_percentage else 1.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val, title={'text': title, 'font': {'size': 18}},
        number={'suffix': "%" if is_percentage else ""},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1},
            'bar': {'color': "#1E293B"},
            'steps': [
                {'range': [0, thresholds[0]], 'color': colors[0]},
                {'range': [thresholds[0], thresholds[1]], 'color': colors[1]},
                {'range': [thresholds[1], max_val], 'color': colors[2]}
            ]
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#1E293B"})
    return fig

# ---------------------------------------------------------------------------
# 3. Dynamic Machine Learning Backend
# ---------------------------------------------------------------------------
def _classify_column(c): 
    n = c.lower()
    if "gender" in n or "sex" in n: return "gender"
    if "alcohol" in n or "alc" in n: return "alcohol"
    return None

def build_feature_encoder(col_name, series):
    clean = series.dropna()
    is_object = not pd.api.types.is_numeric_dtype(clean)
    tag = _classify_column(col_name)
    
    if tag == "gender":
        return {"kind": "categorical", "options": ["Female", "Male"], "encode_map": {"Female":0, "Male":1}}
    if tag == "alcohol":
        opts = ["non alcoholic", "former", "current"]
        return {"kind": "categorical", "options": opts, "encode_map": {o:i for i,o in enumerate(opts)}}
    if is_object:
        opts = sorted(list(pd.unique(clean)))
        return {"kind": "categorical", "options": opts, "encode_map": {o:i for i,o in enumerate(opts)}}
    
    return {"kind": "numeric", "min_val": float(clean.min()), "max_val": float(clean.max()), "default_val": float(clean.median())}

def detect_target_column(df):
    for c in ["target", "outcome", "risk", "htn"]:
        for col in df.columns:
            if c in col.lower(): return col
    return df.columns[-1]

@st.cache_resource
def load_and_train():
    files = glob.glob("*.csv")
    if not files: return None, None, None, None
    df = pd.read_csv(files[0]).fillna(0)
    target = detect_target_column(df)
    X_raw, y_raw = df.drop(columns=[target]), df[target]
    
    encoders = {c: build_feature_encoder(c, X_raw[c]) for c in X_raw.columns}
    X = pd.DataFrame(index=X_raw.index)
    for c in X_raw.columns:
        if encoders[c]["kind"] == "categorical":
            X[c] = X_raw[c].astype(str).str.lower().map(encoders[c]["encode_map"]).fillna(0)
        else:
            X[c] = pd.to_numeric(X_raw[c], errors="coerce").fillna(0)
    
    y = y_raw.astype('category').cat.codes
    model = xgb.XGBClassifier(random_state=42, n_estimators=100, max_depth=3)
    model.fit(X, y)
    return model, X.columns, encoders, list(y_raw.astype('category').cat.categories)

model, feature_names, encoders, class_labels = load_and_train()

# ---------------------------------------------------------------------------
# 4. Frontend Application
# ---------------------------------------------------------------------------
st.title("CardioLens Risk Assessment")
st.markdown("Clinical Decision Support System integrating 10-Year ASCVD Risk and Hypertension Probability.")

if not model:
    st.warning("⚠️ No CSV dataset found in the directory. Please upload a dataset to train the ML model.")
    st.stop()

with st.form("risk_form"):
    t1, t2, t3, t4 = st.tabs(["Demographics", "Vitals & Labs", "History & Lifestyle", "ML Dataset Extras"])
    ui = {}
    
    with t1:
        c1, c2, c3 = st.columns(3)
        ui['age'] = c1.slider("Age (Years)", 20, 100, 50)
        ui['sex'] = c2.selectbox("Biological Sex", ["Male", "Female"])
        ui['race'] = c3.selectbox("Ethnicity (ASCVD)", ["White/Other", "African American"])
        
    with t2:
        c1, c2 = st.columns(2)
        ui['sbp'] = c1.slider("Systolic BP (mmHg)", 90, 200, 120)
        ui['dbp'] = c2.slider("Diastolic BP (mmHg)", 60, 130, 80)
        
        c3, c4 = st.columns(2)
        ui['tc'] = c3.slider("Total Cholesterol (mg/dL)", 100, 350, 180)
        ui['hdl'] = c4.slider("HDL Cholesterol (mg/dL)", 20, 100, 50)
        
        st.markdown("##### Anthropometrics & Nutrition")
        wc_col1, wc_col2 = st.columns([1, 2])
        ui['waist_unit'] = wc_col1.radio("Waist Unit", ["cm", "inches"], horizontal=True)
        default_waist = 90.0 if ui['waist_unit'] == 'cm' else 35.0
        ui['waist_val'] = wc_col2.number_input("Waist Circumference", value=default_waist, min_value=10.0, max_value=300.0)
        ui['waist_cm'] = ui['waist_val'] if ui['waist_unit'] == 'cm' else ui['waist_val'] * 2.54
        
        ui['salt'] = st.slider("Salt Intake (g/day)", 0.0, 30.0, 6.0, step=0.5)
        
    with t3:
        c1, c2 = st.columns(2)
        ui['smoker'] = c1.toggle("Current Smoker")
        ui['diab'] = c2.toggle("Has Diabetes")
        ui['meds'] = c1.toggle("On BP Medication")
        ui['alcohol'] = c2.selectbox("Alcohol Consumption", ["non alcoholic", "former", "current"])
        ui['stress'] = st.slider("Stress Level (1-10 Scale)", 1, 10, 3)
        
    with t4:
        st.caption("Fields automatically detected from your CSV dataset.")
        ml_only_inputs = {}
        standard_keywords = ['age', 'sex', 'gender', 'bp', 'systolic', 'diastolic', 'chol', 'hdl', 'smok', 'diab', 'waist', 'alcohol', 'alc', 'salt', 'sodium', 'stress']
        
        for col in feature_names:
            if not any(k in col.lower() for k in standard_keywords):
                enc = encoders[col]
                if enc['kind'] == 'categorical':
                    ml_only_inputs[col] = st.selectbox(col, enc['options'])
                else:
                    ml_only_inputs[col] = st.number_input(col, value=float(enc['default_val']))
                    
    submit = st.form_submit_button("Assess Risk Profile")

# ---------------------------------------------------------------------------
# 5. Model Execution, Results & Clinical Reference Guides
# ---------------------------------------------------------------------------
if submit:
    st.markdown("---")
    res_c1, res_c2 = st.columns(2)
    
    # 1. ASCVD Output
    ascvd_score = calculate_ascvd(
        ui['age'], ui['sex'], ui['race'], ui['tc'], ui['hdl'], ui['sbp'], ui['meds'], ui['smoker'], ui['diab']
    )
    
    with res_c1:
        st.subheader("10-Year ASCVD Risk")
        fig_ascvd = make_gauge(ascvd_score, "Cardiovascular Event Risk", [5.0, 7.5], ["#22C55E", "#F59E0B", "#EF4444"])
        st.plotly_chart(fig_ascvd, use_container_width=True)

    # 2. XGBoost ML Output
    ml_input_dict = {}
    for col in feature_names:
        cl = col.lower()
        if 'age' in cl: val = ui['age']
        elif 'sex' in cl or 'gender' in cl: val = encoders[col]['encode_map'].get(ui['sex'], 0)
        elif 'sys' in cl or 'sbp' in cl: val = ui['sbp']
        elif 'dia' in cl or 'dbp' in cl: val = ui['dbp']
        elif 'chol' in cl and 'hdl' not in cl: val = ui['tc']
        elif 'hdl' in cl: val = ui['hdl']
        elif 'smok' in cl: val = 1 if ui['smoker'] else 0
        elif 'diab' in cl: val = 1 if ui['diab'] else 0
        elif 'waist' in cl: val = ui['waist_cm']
        elif 'alcohol' in cl or 'alc' in cl: val = encoders[col]['encode_map'].get(ui['alcohol'], 0)
        elif 'salt' in cl or 'sodium' in cl: val = ui['salt']
        elif 'stress' in cl: val = ui['stress']
        else:
            val = ml_only_inputs[col]
            if encoders[col]['kind'] == 'categorical':
                val = encoders[col]['encode_map'].get(val, 0)
                
        ml_input_dict[col] = [val]
        
    df_pred = pd.DataFrame(ml_input_dict)
    proba = model.predict_proba(df_pred)[0]
    high_risk_prob = float(proba[-1]) 
    
    with res_c2:
        st.subheader("Current Hypertension Risk")
        fig_htn = make_gauge(high_risk_prob, "ML Hypertension Probability", [0.4, 0.7], ["#22C55E", "#F59E0B", "#EF4444"])
        st.plotly_chart(fig_htn, use_container_width=True)
        pred_label = class_labels[np.argmax(proba)]
        st.info(f"**ML Classification:** {pred_label}")

    st.markdown("---")
    st.subheader("📖 Clinical Guidelines Reference & Interpretation Guide")
    
    ref_col1, ref_col2 = st.columns(2)
    
    with ref_col1:
        st.markdown("#### 10-Year ASCVD Risk Categories")
        st.markdown("""
        | Risk Category | 10-Year Risk % | Clinical Interpretation & Action |
        | :--- | :--- | :--- |
        | **Low Risk** | `< 5.0%` | Reinforce heart-healthy lifestyle; reassess in 4-6 years. |
        | **Borderline Risk** | `5.0% to < 7.5%` | Consider moderate-intensity statin if risk enhancers are present. |
        | **Intermediate Risk** | `7.5% to < 20.0%` | Moderate-intensity statin recommended after clinician-patient discussion. |
        | **High Risk** | `≥ 20.0%` | High-intensity statin recommended to achieve ≥50% LDL reduction. |
        """)
        
    with ref_col2:
        st.markdown("#### Hypertension / Blood Pressure Tiers")
        st.markdown("""
        | BP Category | Systolic / Diastolic (mmHg) | Recommended Clinical Intervention |
        | :--- | :--- | :--- |
        | **Normal** | `< 120` **and** `< 80` | Healthy lifestyle choices; annual check-ups. |
        | **Elevated** | `120–129` **and** `< 80` | Non-pharmacological lifestyle changes; reassess in 3-6 months. |
        | **Stage 1 Hypertension** | `130–139` **or** `80–89` | Lifestyle changes + 10-year risk assessment (if ≥10%, initiate meds). |
        | **Stage 2 Hypertension** | `≥ 140` **or** `≥ 90` | Combination of lifestyle changes and dual-agent antihypertensive therapy. |
        """)

    st.markdown("---")
    st.markdown("#### 💡 Tailored Clinical Intervention Plan")
    
    interventions = []
    if ascvd_score >= 7.5:
        interventions.append("- **Statin Therapy Consideration:** ACC/AHA guidelines suggest initiating a discussion regarding moderate-to-high intensity statin therapy.")
    if ui['sbp'] >= 130 or ui['dbp'] >= 80:
        interventions.append("- **Blood Pressure Management:** Implement the DASH diet, rich in fruits, vegetables, and low-fat dairy, alongside sodium restriction.")
    if ui['salt'] > 5.0:
        interventions.append(f"- **Sodium Reduction:** Current salt intake ({ui['salt']} g/day) exceeds recommended dietary targets. Gradually taper sodium to ease vascular resistance.")
    if ui['smoker']:
        interventions.append("- **Tobacco Cessation:** Strongly advise smoking cessation and provide support resources or pharmacotherapy.")
    if ui['stress'] >= 7:
        interventions.append(f"- **Stress Management:** High stress level reported ({ui['stress']}/10). Recommend mindfulness, cognitive behavioral therapy, or structured physical activity.")
    if not interventions:
        interventions.append("- **Maintenance:** Patient profile is stable. Maintain routine physical activity (≥150 min/week moderate intensity) and balanced nutrition.")
        
    for item in interventions:
        st.markdown(item)
