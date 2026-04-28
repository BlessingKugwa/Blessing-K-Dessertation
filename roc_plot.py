import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

print('Loading data and models...')
df = pd.read_csv('Operational risk losses data.csv')

# Feature engineering (same as model.py)
df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True)
df['MONTH'] = df['DATE'].dt.month
df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
df['LOSS_PER_FREQ'] = df['LOSS AMOUNT(USD)'] / (df['FREQUENCY'] + 1e-9)
df['SEV_OPVAR_RATIO'] = df['SEVERITY'] / (df['OPVAR(%)'] + 1e-9)
df['RISK_COMPOSITE'] = df['SEVERITY'] * df['FREQUENCY'] * df['OPVAR(%)']
df['GDP_VIX_RATIO'] = df['GDP (%)'] / (df['VIX'] + 1e-9)

# Encode
le_event = LabelEncoder().fit(df['EVENT TYPE'])
le_process = LabelEncoder().fit(df['PROCESS AREA'])
df['EVENT_ENC'] = le_event.transform(df['EVENT TYPE'])
df['PROCESS_ENC'] = le_process.transform(df['PROCESS AREA'])
df['SCENARIO_PREFIX'] = df['SCENARIO ID'].str[:3]
le_scen = LabelEncoder()
df['SCENARIO_ENC'] = le_scen.fit_transform(df['SCENARIO_PREFIX'])

CORE_FEATURES = ["OPVAR(%)", "GDP (%)", "VIX", "EVENT_ENC", "PROCESS_ENC", "SCENARIO_ENC", 
                 "DAY_OF_WEEK", "LOSS_PER_FREQ", "SEV_OPVAR_RATIO", "RISK_COMPOSITE", "GDP_VIX_RATIO"]

X = df[CORE_FEATURES].dropna()
scaler = joblib.load('outputs/feature_scaler.pkl')
X_scaled = scaler.transform(X)

# Load classifier
clf_model = joblib.load('outputs/risk_classifier_rf.pkl')

# Create test set with stratification (simulate model.py split)
y = pd.read_csv('outputs/scores.csv')['ANOMALY_COMBINED'].values  # Use saved anomaly labels
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print('Generating ROC curve...')
y_prob = clf_model.predict_proba(X_test)[:, 1]

# ROC metrics
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f'ROC-AUC Score: {roc_auc:.4f}')

# Plot ROC
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Risk Classification)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print('ROC plot saved: outputs/roc_curve.png')
print('Done!')

