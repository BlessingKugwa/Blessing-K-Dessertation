import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from datetime import datetime
import io

st.set_page_config(page_title="⚠️ OpRisk Dashboard", layout="wide")

CORE_FEATURES = [
    "FREQUENCY", "SEVERITY", "OPVAR(%)", "ML", "GDP (%)", "VIX",
    "EVENT_ENC", "PROCESS_ENC", "SCENARIO_ENC",
    "MONTH", "QUARTER", "DAY_OF_WEEK",
    "LOSS_PER_FREQ", "SEV_OPVAR_RATIO", "RISK_COMPOSITE",
    "ML_VIX_INTERACT", "GDP_VIX_RATIO"
]

@st.cache_data
def load_models():
    iso_forest = joblib.load('outputs/anomaly_isolation_forest.pkl')
    loss_rf = joblib.load('outputs/loss_regression_rf.pkl')
    risk_rf = joblib.load('outputs/risk_classifier_rf.pkl')
    scaler = joblib.load('outputs/feature_scaler.pkl')
    encoders = joblib.load('outputs/label_encoders.pkl')
    scored_df = pd.read_csv('outputs/scored_risk_data.csv')
    return iso_forest, loss_rf, risk_rf, scaler, encoders, scored_df

iso_forest, loss_rf, risk_rf, scaler, encoders, scored_df = load_models()

st.title("⚠️ Operational Risk Dashboard")

# Sidebar metrics
with st.sidebar:
    st.header("📊 Model Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Loss R²", "0.86")
    with col2:
        st.metric("F1-macro", "0.36")
    with col3:
        st.metric("Anomalies", f"{scored_df['ANOMALY_COMBINED'].mean():.1%}")
    st.header("🔥 Alert Distribution")
    tier_counts = scored_df['ALERT_TIER'].value_counts()
    st.bar_chart(tier_counts)

# Tabs
tab1, tab2 = st.tabs(["📈 Historical Risks", "🔮 New Prediction"])

with tab1:
    st.header("Historical Scored Data")
    col1, col2 = st.columns(2)
    with col1:
        tier_filter = st.multiselect("Alert Tier", scored_df['ALERT_TIER'].unique(), default=['HIGH', 'CRITICAL'])
    with col2:
        event_filter = st.multiselect("Event Type", scored_df['EVENT TYPE'].unique())
    filtered_df = scored_df[
        (scored_df['ALERT_TIER'].isin(tier_filter)) &
        (event_filter == [] or scored_df['EVENT TYPE'].isin(event_filter))
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(filtered_df, names='ALERT_TIER', title="Risk Tiers")
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        fig_loss = px.box(filtered_df, x='ALERT_TIER', y='RISK_SCORE', title="Risk Score Distro")
        st.plotly_chart(fig_loss, use_container_width=True)
    

    high_df = filtered_df[filtered_df['ALERT_TIER'].isin(['HIGH', 'CRITICAL'])].sort_values('RISK_SCORE', ascending=False).head(20)
    st.dataframe(high_df[['SCENARIO ID', 'EVENT TYPE', 'PROCESS AREA', 'RISK_SCORE', 'ALERT_TIER', 'ANOMALY_COMBINED']].style.format({'RISK_SCORE': '{:.1f}'}))
    

    fig_anom = px.histogram(filtered_df, x='ANOMALY_COMBINED', color='ALERT_TIER', title="Anomalies by Tier")
    st.plotly_chart(fig_anom, use_container_width=True)

with tab2:
    st.header("Predict New Raw Data (Full Pipeline + Anomaly Detection)")
    uploaded_file = st.file_uploader("Upload raw CSV (same format as 'Operational risk losses data.csv')", type='csv')
    
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.info(f"Uploaded {len(raw_df)} records")
        st.write(raw_df.head())
        
        if st.button("🚀 Run Full Pipeline (Anomaly + Risk Scoring)"):
            try:
                # Feature engineering (from op_risk_model.py)
                df = raw_df.copy()
                df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
                df["MONTH"] = df["DATE"].dt.month
                df["QUARTER"] = df["DATE"].dt.quarter
                df["YEAR"] = df["DATE"].dt.year
                df["DAY_OF_WEEK"] = df["DATE"].dt.dayofweek
                df["LOSS_PER_FREQ"] = df["LOSS AMOUNT(USD)"] / (df["FREQUENCY"] + 1e-9)
                df["SEV_OPVAR_RATIO"] = df["SEVERITY"] / (df["OPVAR(%)"] + 1e-9)
                df["RISK_COMPOSITE"] = df["SEVERITY"] * df["FREQUENCY"] * df["OPVAR(%)"]
                df["ML_VIX_INTERACT"] = df["ML"] * df["VIX"]
                df["GDP_VIX_RATIO"] = df["GDP (%)"] / (df["VIX"] + 1e-9)
                
                # Encode
                df["SCENARIO_PREFIX"] = df["SCENARIO ID"].str[:3]
                df["EVENT_ENC"] = encoders["event"].transform(df["EVENT TYPE"])
                df["PROCESS_ENC"] = encoders["process"].transform(df["PROCESS AREA"])
                df["SCENARIO_ENC"] = encoders["scenario"].transform(df["SCENARIO_PREFIX"])
                
                # Scale
                X = df[CORE_FEATURES].fillna(0)
                X_scaled = scaler.transform(X)
                
                # Predict
                df["ANOMALY_ISO"] = iso_forest.predict(X_scaled)
                df["ANOMALY_ISO"] = np.where(df["ANOMALY_ISO"] == -1, 1, 0)
                df["PRED_LOSS"] = loss_rf.predict(X_scaled)
                df["PRED_TARGET"] = risk_rf.predict(X_scaled)
                df["PROB_HIGH"] = risk_rf.predict_proba(X_scaled)[:, 2]
                
                # Risk score
                loss_norm = (df["PRED_LOSS"] - df["PRED_LOSS"].min()) / (df["PRED_LOSS"].max() - df["PRED_LOSS"].min())
                df["RISK_SCORE"] = (0.40 * loss_norm + 0.35 * df["PROB_HIGH"] + 0.25 * (1 - df["ANOMALY_ISO"])) * 100
                df["ALERT_TIER"] = pd.cut(df["RISK_SCORE"], bins=[0,30,60,85,100], labels=['LOW','MEDIUM','HIGH','CRITICAL'])
                df["ANOMALY_FLAG"] = df["ANOMALY_ISO"].map({1: "🚨 ANOMALOUS", 0: "✅ Normal"})
                
                st.success(f"✅ Processed {len(df)} records!")
                st.metric("New Anomalies Found", df['ANOMALY_ISO'].sum())
                
                st.dataframe(df[['SCENARIO ID', 'EVENT TYPE', 'LOSS AMOUNT(USD)', 'RISK_SCORE', 'ALERT_TIER', 'ANOMALY_FLAG']].style.format({'RISK_SCORE': '{:.1f}'}))
                
                # Download
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button("📥 Download Scored Data", csv_buffer.getvalue().encode(), "scored_new_data.csv")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.caption("Professional OpRisk Dashboard | Full ML pipeline + anomaly detection")

