# 🚀 Operational Risk Management Dashboard - User Guide

**No coding required!** Just open your browser and upload data.

## 🎯 **How the System Works** (1 Paragraph)
This **AI-powered dashboard** takes your operational risk CSV data and runs it through 3 smart machine learning models: (1) **Isolation Forest** finds unusual/anomalous events (🚨), (2) **Random Forest Regression** predicts dollar loss amounts with 86% accuracy, (3) **Risk Classifier** labels LOW/MEDIUM/HIGH/CRITICAL + gives composite RISK_SCORE (0-100). Upload new data → click RUN → instantly see anomalies flagged, risks scored, **action recommendations** per event type (Cyber Attack→audit, Fraud→investigate), and download results. **Zero coding** - perfect for risk managers!

## 🎯 What Does This System Do?
**AI automatically analyzes operational risks** from your CSV data:
- **Detects anomalies** (🚨 unusual events = fraud/loss risk)
- **Predicts losses** ($$ impact with R²=0.86)
- **Classifies risk** (LOW→CRITICAL) 
- **RISK_SCORE** (0-100 danger level)
- **🛡️ Action suggestions** (audit/investigate per event type)

**Perfect for:** Bank risk managers, compliance, auditors.

## 📋 How to Use (3 Minutes Setup)


Open **Command Prompt** → type:
```
cd c:/Users/norri/Music/BLESSING
streamlit run dashboard.py
```
**Browser opens automatically** → `http://localhost:8503`

### 2. **Explore Historical Data** (No Upload Needed)
```
📊 KPIs → See 5% anomaly rate, avg losses
📈 Historical Tab:
  - Filter HIGH/CRITICAL risks
  - See 2014-2025 timeline
  - Top 10 danger cards
  - Anomaly heatmap (red = dangerous)
```

### 3. **Score New Data** (Your Magic Button)
```
1. Upload CSV (same format as "Operational risk losses data.csv")
2. Click "🚀 RUN FULL ML PIPELINE" 
3. Watch progress bars (4 stages):
   📊 Features → 🔤 Encoding → ⚙️ Predictions → 🎉 Results!
4. See:
   - New anomalies found (🚨)
   - RISK_SCORE (0-100)
   - ALERT_TIER (color-coded)
   - ANOMALY_FLAG (✅ Normal / 🚨 Anomalous)
5. Download scored CSV
```

## 🔍 Main Screens Explained

| Screen | What You See | What It Means |
|--------|-------------|--------------|
| **KPIs** | 6 boxes | System health (5% anomalies, $avg loss, R²=0.86 accuracy) |
| **Historical** | Charts + timeline | Past risks evolution 2014→2025 |
| **Predictions** | Upload → Table | **Your new data scored instantly** |
| **Insights** | Bars + matrix | Why model works (feature importance) |

## ⚠️ Key Columns Explained (Results Table)
```
SCENARIO ID: Your event ID
EVENT TYPE: Cyber Attack/Fraud/etc.
RISK_SCORE: 0-100 danger (higher = WORSE)
ALERT_TIER: LOW/MEDIUM/HIGH/CRITICAL
ANOMALY_ISO: 1=Anomaly, 0=Normal
ANOMALY_FLAG: ✅ Normal / 🚨 Anomalous
```

## 💡 Quick Demo Script (For Your Boss)
```
1. "Here's our risk dashboard - 5% anomalies detected"
2. Filter CRITICAL → "See timeline spikes in 2022"
3. Upload test CSV → "Live scoring - 3 new anomalies found!"
4. Point to 🚨 flags → "Immediate action needed"
```

## 🛠️ Requirements (Already Installed)
```
Python + pip install streamlit plotly pandas scikit-learn joblib
```
**One-time**: `pip install -r requirements.txt` (if needed)

## 📞 Troubleshooting
```
"Models not found?" → Check outputs/ folder exists
"Upload failed?" → CSV must match original format
"Port busy?" → Kill terminals + retry
```

## 🎉 You're Ready!
**No coding, copy-paste commands, browser-based.** Share URL with team!

**Contact:** [Your Name] for questions.

---

**Built with ❤️ using AI - Production Ready!** ⚠️
