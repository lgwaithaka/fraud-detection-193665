"""
DSA 8502 — Motor Insurance Fraud Detection
Streamlit Data Product
Author  : Lawrence Gacheru Waithaka | 193665
Course  : Strathmore University — MSc Data Science
Version : 1.0  |  March 2026

Run locally:
    pip install streamlit shap matplotlib seaborn pandas numpy joblib xgboost scikit-learn
    streamlit run streamlit_app.py

Deploy:
    Push to GitHub → connect to share.streamlit.io
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── 2. Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Motor Insurance Fraud Detector — East Africa",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 3. Colour palette & CSS ───────────────────────────────────────────────────
NAVY   = "#1F4E79"
STEEL  = "#2E75B6"
RED    = "#C62828"
GREEN  = "#1B5E20"
AMBER  = "#E65100"

st.markdown(f"""
<style>
    .main-header {{
        background: linear-gradient(135deg, {NAVY} 0%, {STEEL} 100%);
        padding: 1.8rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
    }}
    .metric-card {{
        background: #F8FAFC;
        border: 1px solid #E0E8F0;
        border-left: 5px solid {STEEL};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.4rem 0;
    }}
    .fraud-alert {{
        background: #FFEBEE;
        border: 2px solid {RED};
        border-radius: 8px;
        padding: 1.2rem;
        color: {RED};
        font-weight: bold;
    }}
    .safe-alert {{
        background: #E8F5E9;
        border: 2px solid {GREEN};
        border-radius: 8px;
        padding: 1.2rem;
        color: {GREEN};
        font-weight: bold;
    }}
    .review-alert {{
        background: #FFF3E0;
        border: 2px solid {AMBER};
        border-radius: 8px;
        padding: 1.2rem;
        color: {AMBER};
        font-weight: bold;
    }}
    .sidebar-info {{
        background: #EEF4FB;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.85rem;
        margin-top: 1rem;
    }}
    .section-header {{
        border-bottom: 2px solid {STEEL};
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        color: {NAVY};
        font-size: 1.1rem;
        font-weight: 600;
    }}
</style>
""", unsafe_allow_html=True)

# ── 4. Load artefacts ─────────────────────────────────────────────────────────
ARTEFACT_DIR = Path("artefacts")

@st.cache_resource(show_spinner="Loading fraud detection model…")
def load_artefacts():
    """Load all model artefacts once and cache them."""
    artefacts = {}
    try:
        artefacts["model"]         = joblib.load(ARTEFACT_DIR / "fraud_model.pkl")
        artefacts["scaler"]        = joblib.load(ARTEFACT_DIR / "scaler.pkl")
        artefacts["encoders"]      = joblib.load(ARTEFACT_DIR / "encoders.pkl")
        artefacts["feature_names"] = joblib.load(ARTEFACT_DIR / "feature_names.pkl")
        artefacts["shap_model"]    = joblib.load(ARTEFACT_DIR / "xgb_for_shap.pkl")
        artefacts["explainer"]     = joblib.load(ARTEFACT_DIR / "shap_explainer.pkl")
        with open(ARTEFACT_DIR / "config.json") as f:
            artefacts["config"]    = json.load(f)
        artefacts["loaded"] = True
    except FileNotFoundError as e:
        artefacts["loaded"] = False
        artefacts["error"]  = str(e)
    return artefacts

artefacts = load_artefacts()

# ── 5. Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:1.8rem;">🛡️ Motor Insurance Fraud Detector</h1>
    <p style="margin:0.3rem 0 0; opacity:0.85; font-size:0.95rem;">
        · A Case for East Africa · <br> Kenya DPA 2019 Compliant ·  <br> · Powered by XGBoost + Stacking Ensemble + SHAP/LIME XAI · <br>
        · DSA 8502 | Lawrence Gacheru Waithaka (193665) <br>· Strathmore University · 
    </p>
</div>
""", unsafe_allow_html=True)

# ── 6. Sidebar — model info & navigation ──────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 System Info")
    if artefacts.get("loaded"):
        cfg = artefacts["config"]
        st.markdown(f"""
        <div class="sidebar-info">
        <b>Model:</b> {cfg.get('best_model', 'N/A')}<br>
        <b>Test AUC-ROC:</b> {cfg.get('test_auc', 0):.4f}<br>
        <b>Test Recall:</b> {cfg.get('test_recall', 0):.4f}<br>
        <b>Test F1:</b> {cfg.get('test_f1', 0):.4f}<br>
        <b>Threshold:</b> {cfg.get('business_threshold', 0.5):.2f}<br>
        <b>Features:</b> {cfg.get('n_features', 0)}<br>
        <b>Training records:</b> {cfg.get('training_records', 0):,}<br>
        <b>Seed:</b> {cfg.get('seed', 42)}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"⚠️ Model not loaded.\n\n{artefacts.get('error', '')}\n\nRun the notebook first to generate artefacts/.")

    st.markdown("---")
    page = st.radio(
        "Navigate to:",
        ["🎯 Score a Claim:", "📊 Batch Upload:", "📈 Model Performance:", "📖 About the Model:"],
        index=0,
    )

# ── 7. Utility functions ──────────────────────────────────────────────────────
BUSINESS_LABELS = {
    "days_into_policy"           : "Days from policy start to claim",
    "loss_ratio_band"            : "Loss ratio band",
    "claims_same_policy_num"     : "Claims on same policy",
    "claim_category"             : "Claim type",
    "days_loss_to_notify"        : "Days to notify after accident",
    "channel_type"               : "Distribution channel",
    "decline_reason_category"    : "EXPAQ decline typology",
    "early_claim_flag"           : "Early claim (≤30 days)",
    "multi_claim_flag"           : "Multi-claim policy",
    "TotalClaim"                 : "Total claim (KES)",
    "claim_to_deductible_ratio"  : "Claim/deductible ratio",
    "injury_claim_fraction"      : "Injury fraction",
    "DriverRating"               : "Driver risk rating",
    "WitnessPresent"             : "Witness present",
    "PoliceReportFiled"          : "Police abstract filed",
    "rapid_notify_flag"          : "Same-day notification",
    "external_channel_flag"      : "External channel",
}


def preprocess_input(data: dict, artefacts: dict) -> np.ndarray:
    """Convert a raw claim dict into a scaled feature vector."""
    feature_names = artefacts["feature_names"]
    scaler        = artefacts["scaler"]
    encoders      = artefacts["encoders"]

    row = {}
    for feat in feature_names:
        val = data.get(feat, 0)
        if feat in encoders:
            le = encoders[feat]
            val_str = str(val)
            if val_str in le.classes_:
                val = le.transform([val_str])[0]
            else:
                val = 0  # unseen category → 0
        row[feat] = val

    X_raw = np.array([[row[f] for f in feature_names]], dtype=np.float32)
    X_sc  = scaler.transform(X_raw)
    return X_sc


def predict_claim(X_sc: np.ndarray, artefacts: dict) -> tuple:
    """
    Return (fraud_prob, decision, colour_class).
    decision: 'FRAUD ALERT', 'NEEDS REVIEW', 'LIKELY LEGITIMATE'
    """
    model  = artefacts["model"]
    thresh = artefacts["config"].get("business_threshold", 0.5)
    prob   = model.predict_proba(X_sc)[0, 1]

    if prob >= thresh:
        return prob, "FRAUD ALERT", "fraud"
    elif prob >= thresh * 0.6:
        return prob, "NEEDS REVIEW", "review"
    else:
        return prob, "LIKELY LEGITIMATE", "safe"


def shap_waterfall(X_sc: np.ndarray, artefacts: dict, feature_names: list) -> plt.Figure:
    """Compute SHAP values and return a waterfall matplotlib figure."""
    explainer = artefacts["explainer"]
    sv        = explainer.shap_values(X_sc)[0]
    display   = [BUSINESS_LABELS.get(f, f) for f in feature_names]

    exp = shap.Explanation(
        values=sv,
        base_values=explainer.expected_value,
        data=X_sc[0],
        feature_names=display,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(exp, max_display=15, show=False)
    plt.title("Why was this claim flagged?", fontsize=11)
    plt.tight_layout()
    return fig


# ── 8. PAGE: Score a Claim ────────────────────────────────────────────────────
if page == "🎯 Score a Claim":
    st.markdown('<div class="section-header">Enter Claim Details</div>', unsafe_allow_html=True)

    if not artefacts.get("loaded"):
        st.error("Model artefacts not found. Run the notebook first to generate artefacts/.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Policy & Claimant**")
        age         = st.slider("Claimant Age", 18, 80, 35)
        driver_rat  = st.selectbox("Driver Risk Rating (1=High Risk)", [1, 2, 3, 4], index=1)
        marital     = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        sex         = st.selectbox("Sex", ["Male", "Female"])
        num_cars    = st.selectbox("Number of Cars on Policy", ["1 vehicle", "2 vehicles", "3 to 4", "more than 4"])
        client_seg  = st.selectbox("Client Segment", ["Retail", "Corporate"])

    with col2:
        st.markdown("**Claim Details**")
        claim_cat   = st.selectbox("Claim Type", [
            "RTA_GENERAL", "WINDSCREEN", "THEFT", "THIRD_PARTY_INJURY",
            "THIRD_PARTY_PROPERTY", "SELF_INVOLVED", "FIRE", "FLOOD",
            "MALICIOUS_DAMAGE", "OTHER"
        ])
        injury_claim   = st.number_input("Injury Claim Amount (KES)", 0, 10_000_000, 0, 10_000)
        property_claim = st.number_input("Property Claim Amount (KES)", 0, 10_000_000, 150_000, 10_000)
        vehicle_claim  = st.number_input("Vehicle Claim Amount (KES)", 0, 10_000_000, 350_000, 10_000)
        deductible     = st.selectbox("Policy Excess/Deductible (KES)", [300, 400, 500, 700])
        fault          = st.selectbox("Fault Attribution", ["Policy Holder", "Third Party"])

    with col3:
        st.markdown("**Timing & Channel**")
        days_into_pol    = st.slider("Days into policy at claim (IRA signal)", 0, 365, 45)
        days_notify      = st.slider("Days from accident to notification", 0, 180, 5)
        claims_same_pol  = st.slider("Prior claims on same policy", 1, 15, 1)
        channel          = st.selectbox("Distribution Channel", [
            "BROKER", "AGENT", "BANCASSURANCE", "DIRECT", "OTHER"
        ])
        police_rep       = st.selectbox("Police Report Filed?", ["No", "Yes"])
        witness          = st.selectbox("Witness Present?", ["No (0)", "Yes (1)"])
        vehicle_cat      = st.selectbox("Vehicle Category", ["Sedan", "Sport", "Utility"])
        base_pol         = st.selectbox("Base Policy", ["Collision", "Liability", "All Perils"])
        product_type     = st.selectbox("Product Type", ["MOTOR PRIVATE", "MOTOR COMMERCIAL"])

    # ── Build input dict ──────────────────────────────────────────────────────
    total_claim  = injury_claim + property_claim + vehicle_claim
    early_flag   = 1 if days_into_pol < 30 else 0
    rapid_notify = 1 if days_notify <= 1 else 0
    multi_claim  = 1 if claims_same_pol >= 3 else 0
    ext_channel  = 1 if channel in ["BROKER", "BANCASSURANCE", "AGENT"] else 0

    # Categorical features that need LR band derivation
    if total_claim == 0:
        lr_band = "low"
    elif total_claim > 10_000_000:
        lr_band = "extreme"
    elif total_claim > 5_000_000:
        lr_band = "high"
    elif total_claim > 1_000_000:
        lr_band = "elevated"
    else:
        lr_band = "normal"

    claim_input = {
        "Month"                   : "Jan",
        "WeekOfMonth"             : 2,
        "DayOfWeek"               : "Monday",
        "Make"                    : "Toyota",
        "AccidentArea"            : "Urban",
        "DayOfWeekClaimed"        : "Wednesday",
        "MonthClaimed"            : "Jan",
        "WeekOfMonthClaimed"      : 2,
        "Sex"                     : sex,
        "MaritalStatus"           : marital,
        "Age"                     : age,
        "Fault"                   : fault,
        "PolicyType"              : f"{vehicle_cat} - {base_pol}",
        "VehicleCategory"         : vehicle_cat,
        "VehiclePrice"            : "20000 to 29000",
        "PolicyNumber"            : 999999,
        "RepNumber"               : 5,
        "Deductible"              : deductible,
        "DriverRating"            : driver_rat,
        "Days_Policy_Accident"    : "more than 30" if days_into_pol > 30 else "1 to 7",
        "Days_Policy_Claim"       : "1 to 7" if days_notify <= 7 else "8 to 15",
        "PastNumberOfClaims"      : "none" if claims_same_pol <= 1 else "1" if claims_same_pol == 2 else "2 to 4",
        "AgeOfVehicle"            : "more than 7",
        "AgeOfPolicyHolder"       : "36 to 40",
        "PoliceReportFiled"       : police_rep,
        "WitnessPresent"          : 1 if "Yes" in witness else 0,
        "AgentType"               : "External" if ext_channel else "Internal",
        "NumberOfSuppliments"     : "none",
        "AddressChange_Claim"     : "no change",
        "NumberOfCars"            : num_cars,
        "Year"                    : 2024,
        "BasePolicy"              : base_pol,
        "InjuryClaim"             : injury_claim,
        "PropertyClaim"           : property_claim,
        "VehicleClaim"            : vehicle_claim,
        "claim_category"          : claim_cat,
        "channel_type"            : channel,
        "product_type"            : product_type,
        "days_loss_to_notify"     : days_notify,
        "days_into_policy"        : days_into_pol,
        "loss_ratio_band"         : lr_band,
        "claims_same_policy_num"  : claims_same_pol,
        "client_segment"          : client_seg,
        "decline_reason_category" : "NOT_DECLINED",
        # Engineered features
        "TotalClaim"                 : total_claim,
        "claim_to_deductible_ratio"  : min(total_claim / max(deductible, 1), 1000),
        "injury_claim_fraction"      : injury_claim / max(total_claim, 1),
        "early_claim_flag"           : early_flag,
        "rapid_notify_flag"          : rapid_notify,
        "multi_claim_flag"           : multi_claim,
        "external_channel_flag"      : ext_channel,
        "expaq_confirmed_flag"       : 0,
    }

    # ── Score button ──────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔍  Score This Claim", type="primary", use_container_width=True):

        X_sc = preprocess_input(claim_input, artefacts)
        prob, decision, colour = predict_claim(X_sc, artefacts)
        thresh = artefacts["config"].get("business_threshold", 0.5)

        # Result alert
        css_class = f"{colour}-alert"
        icon = "🚨" if colour == "fraud" else ("⚠️" if colour == "review" else "✅")
        st.markdown(f"""
        <div class="{css_class}">
            {icon} &nbsp;<b>{decision}</b><br>
            Fraud Probability: <b>{prob*100:.1f}%</b> &nbsp;|&nbsp;
            Decision threshold: {thresh:.2f} &nbsp;|&nbsp;
            {"Refer for investigation" if colour != "safe" else "Proceed to payment"}
        </div>
        """, unsafe_allow_html=True)

        # Gauge meter
        st.markdown('<div class="section-header">Fraud Risk Score</div>', unsafe_allow_html=True)
        fig_gauge, ax = plt.subplots(figsize=(8, 1.5))
        cmap = plt.cm.RdYlGn_r
        ax.barh(0, prob, color=cmap(prob), height=0.5, edgecolor="black", linewidth=0.5)
        ax.barh(0, 1 - prob, left=prob, color="#E0E0E0", height=0.5)
        ax.axvline(thresh, color="black", linewidth=2, linestyle="--", label=f"Threshold={thresh:.2f}")
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Fraud Probability", fontsize=10)
        ax.text(prob, 0, f" {prob*100:.1f}%", va="center", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_facecolor("#FAFAFA")
        fig_gauge.patch.set_facecolor("#FAFAFA")
        plt.tight_layout()
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close(fig_gauge)

        # Key risk flags
        st.markdown('<div class="section-header">Key Risk Signals</div>', unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Early Claim Flag", "⚠️ YES" if early_flag else "✅ No", delta=None)
        r2.metric("Multi-Claim Flag", "⚠️ YES" if multi_claim else "✅ No", delta=None)
        r3.metric("Rapid Notification", "⚠️ YES" if rapid_notify else "✅ No", delta=None)
        r4.metric("Total Claim (KES)", f"{total_claim:,.0f}", delta=None)

        # SHAP explanation
        st.markdown('<div class="section-header">Why was this score assigned? (SHAP)</div>', unsafe_allow_html=True)
        with st.spinner("Computing SHAP explanation…"):
            try:
                shap_fig = shap_waterfall(X_sc, artefacts, artefacts["feature_names"])
                st.pyplot(shap_fig, use_container_width=True)
                plt.close(shap_fig)
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

        # Adjuster summary
        st.markdown('<div class="section-header">📋 Claims Adjuster Summary</div>', unsafe_allow_html=True)
        reasons = []
        if early_flag:
            reasons.append(f"Claim filed {days_into_pol} days after policy inception (IRA Kenya: ≤30 days is primary fraud signal)")
        if multi_claim:
            reasons.append(f"{claims_same_pol} claims on the same policy (fraud ring indicator)")
        if rapid_notify:
            reasons.append(f"Claim notified {days_notify} day(s) after accident (suspiciously fast for complex incident)")
        if lr_band in ["high", "extreme"]:
            reasons.append(f"Claim amount (KES {total_claim:,}) is disproportionate relative to policy value")
        if not reasons:
            reasons.append("No major IRA Kenya fraud signals detected in the input fields.")

        summary = f"**Claim Fraud Probability: {prob*100:.1f}%** (threshold: {thresh:.0%})\n\n"
        summary += "**Key factors driving this score:**\n"
        for i, r in enumerate(reasons, 1):
            summary += f"\n{i}. {r}"
        st.info(summary)

# ── 9. PAGE: Batch Upload ─────────────────────────────────────────────────────
elif page == "📊 Batch Upload":
    st.markdown('<div class="section-header">📊 Batch Claim Scoring</div>', unsafe_allow_html=True)
    st.markdown("""
    Upload a CSV file with claim data. The system will score each claim and return a
    risk-ranked report you can download.

    **Required columns:** Same as the `insurance_fraud_v3_final.csv` feature schema (46 columns).
    A sample file is available in `artefacts/sample_claims.csv`.
    """)

    uploaded = st.file_uploader("Upload claims CSV", type=["csv"])
    if uploaded is not None:
        df_upload = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_upload):,} claims × {df_upload.shape[1]} columns")

        if not artefacts.get("loaded"):
            st.error("Model artefacts not loaded.")
        else:
            with st.spinner(f"Scoring {len(df_upload):,} claims…"):
                feature_names = artefacts["feature_names"]
                scaler        = artefacts["scaler"]
                encoders      = artefacts["encoders"]
                model         = artefacts["model"]
                thresh        = artefacts["config"].get("business_threshold", 0.5)

                df_score = df_upload.copy()
                for feat in feature_names:
                    if feat not in df_score.columns:
                        df_score[feat] = 0
                    if feat in encoders:
                        le = encoders[feat]
                        df_score[feat] = df_score[feat].astype(str).apply(
                            lambda v: le.transform([v])[0] if v in le.classes_ else 0
                        )
                    else:
                        df_score[feat] = pd.to_numeric(df_score[feat], errors="coerce").fillna(0)

                X_batch = scaler.transform(df_score[feature_names].values.astype(np.float32))
                probs   = model.predict_proba(X_batch)[:, 1]
                preds   = (probs >= thresh).astype(int)

            df_upload["fraud_probability"] = probs
            df_upload["prediction"]        = preds
            df_upload["decision"] = df_upload["fraud_probability"].apply(
                lambda p: "FRAUD ALERT" if p >= thresh else ("NEEDS REVIEW" if p >= thresh * 0.6 else "LEGITIMATE")
            )
            df_upload = df_upload.sort_values("fraud_probability", ascending=False)

            n_fraud = int(preds.sum())
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Claims", f"{len(df_upload):,}")
            col_b.metric("Flagged as Fraud", f"{n_fraud:,}", f"{n_fraud/len(df_upload)*100:.1f}%")
            col_c.metric("Likely Legitimate", f"{len(df_upload)-n_fraud:,}")

            st.markdown('<div class="section-header">Risk-Ranked Results (Top 50)</div>', unsafe_allow_html=True)
            cols_show = ["fraud_probability", "decision"] + [
                c for c in ["claim_category", "channel_type", "days_into_policy",
                            "claims_same_policy_num", "TotalClaim", "data_source"]
                if c in df_upload.columns
            ]
            st.dataframe(df_upload[cols_show].head(50), use_container_width=True)

            csv_out = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Full Scored Report (CSV)",
                csv_out,
                file_name="fraud_scored_claims.csv",
                mime="text/csv",
            )
    else:
        if (ARTEFACT_DIR / "sample_claims.csv").exists():
            sample = pd.read_csv(ARTEFACT_DIR / "sample_claims.csv")
            st.markdown("**Preview of `artefacts/sample_claims.csv` (first 5 rows):**")
            st.dataframe(sample.head(5), use_container_width=True)

# ── 10. PAGE: Model Performance ───────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.markdown('<div class="section-header">📈 Model Performance Dashboard</div>', unsafe_allow_html=True)

    if not artefacts.get("loaded"):
        st.error("Model artefacts not loaded.")
        st.stop()

    cfg = artefacts["config"]

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC", f"{cfg.get('test_auc', 0):.4f}",
                delta="✅ target ≥0.93" if cfg.get('test_auc', 0) >= 0.93 else "⚠️ below target")
    col2.metric("Recall",   f"{cfg.get('test_recall', 0):.4f}",
                delta="✅ target ≥0.90" if cfg.get('test_recall', 0) >= 0.90 else "⚠️ below target")
    col3.metric("F1-Score", f"{cfg.get('test_f1', 0):.4f}",
                delta="✅ target ≥0.85" if cfg.get('test_f1', 0) >= 0.85 else "⚠️ below target")
    col4.metric("Decision Threshold", f"{cfg.get('business_threshold', 0.5):.2f}")

    # Model comparison table if available
    csv_path = ARTEFACT_DIR / "model_comparison.csv"
    if csv_path.exists():
        st.markdown('<div class="section-header">All Models Comparison</div>', unsafe_allow_html=True)
        comp_df = pd.read_csv(csv_path, index_col=0)
        st.dataframe(comp_df.style.highlight_max(axis=0, color="#C8E6C9"), use_container_width=True)

    # Saved plots
    for img_name, caption in [
        ("roc_pr_curves.png",    "ROC and Precision-Recall Curves"),
        ("shap_beeswarm.png",    "SHAP Global Feature Importance (Beeswarm)"),
        ("shap_bar.png",         "SHAP Mean |Value| — Top Predictors"),
        ("confusion_threshold.png", "Confusion Matrix at Business Threshold"),
        ("eda_overview.png",     "EDA Overview"),
    ]:
        img_path = ARTEFACT_DIR / img_name
        if img_path.exists():
            st.markdown(f'<div class="section-header">{caption}</div>', unsafe_allow_html=True)
            st.image(str(img_path), use_column_width=True)

# ── 11. PAGE: About ───────────────────────────────────────────────────────────
elif page == "📖 About":
    st.markdown("""
    ## About my System

    ### Project:
    **Predicting Fraudulent Motor Insurance Claims in East Africa Using Hybrid Ensemble
    Learning and Explainable AI**

    | Item | Detail |
    |---|---|
    | **Developer/Student** | Lawrence Gacheru Waithaka (193665) |
    | **Course** | DSA 8502 — Predictive & Optimization Techniques |
    | **Institution** | Strathmore Institute of Mathematical Sciences |
    | **Supervisors** | Dr. Allan Omondi, Dr. Kennedy Senagi & Lee Bundi |
    | **Framework** | CRISP-DM |
    | **Dataset** | 108,783 records × 46 features (3 sources) |

    ---

    ### Data Sources:

    | Source | Records | Fraud Labels |
    |---|---|---|
    | Synthetic (Kaggle - benchmark) | 100,000 | Statistical distributions |
    | Insurance (KE) Real Claims 2023–2024 (Partner Insurance) | 8,694 | IRA Kenya heuristic scoring |
    | Confirmed Declined Claims (Legacy System) | 89 | Investigator-verified |

    ---

    ### Models Implemented:

    1. **Decision Tree** — interpretable baseline
    2. **Random Forest** — bagging ensemble
    3. **Gradient Boosting** — sequential ensemble
    4. **XGBoost (Optuna HPO)** — primary model (Bayesian optimisation)
    5. **AdaBoost** — adaptive boosting comparison
    6. **ANN (MLP)** — neural network baseline
    7. **Stacking Ensemble** — OOF meta-learning (LR meta-learner) ← **deployed model**

    ---

    ### Explainability (XAI):

    - **SHAP TreeExplainer** — exact, fast global + local explanations.
    - **LIME LimeTabularExplainer** — local linear approximations.
    - Both methods produce human-readable, Kenya-specific adjuster summaries.

    ---

    ### Compliance:
    - Kenya Data Protection Act (2019) — No PII retained.
    - IRA Kenya fraud detection guidelines (2022) — thresholds aligned.
    - Explainable AI — every decision can be audited.

    ---

    ### Research References:
    1. Chen, T. & Guestrin, C. (2016). XGBoost. *KDD 2016.*
    2. Lundberg, S. & Lee, S.I. (2017). SHAP. *NeurIPS 2017.*
    3. Ribeiro, M.T. et al. (2016). LIME. *KDD 2016.*
    4. Chawla, N.V. et al. (2002). SMOTE. *JAIR 16.*
    5. Akiba, T. et al. (2019). Optuna. *KDD 2019.*
    6. IRA Kenya (2022). Annual Insurance Industry Report.
    """)
