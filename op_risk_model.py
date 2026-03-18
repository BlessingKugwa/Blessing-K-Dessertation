"""
Operational Risk Loss Model
- Anomaly Detection  : Isolation Forest + Z-score
- Loss Regression    : Random Forest (predict LOSS AMOUNT)
- Risk Classification: Random Forest (predict TARGET 0/1/2)
"""

import pandas as pd
import numpy as np
import warnings, joblib, os
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              classification_report, confusion_matrix, f1_score)
from scipy import stats

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("OPERATIONAL RISK LOSS MODEL")
print("=" * 60)

df = pd.read_csv("Operational risk losses data.csv")
print(f"\n[Data] Loaded {len(df)} records, {df.shape[1]} columns")
print(f"[Data] Columns: {list(df.columns)}")
print(f"[Data] Missing values:\n{df.isnull().sum()}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[Feature Engineering] Processing...")

# Parse date
df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
df["MONTH"]     = df["DATE"].dt.month
df["QUARTER"]   = df["DATE"].dt.quarter
df["YEAR"]      = df["DATE"].dt.year
df["DAY_OF_WEEK"] = df["DATE"].dt.dayofweek

# Derived risk features
df["LOSS_PER_FREQ"]     = df["LOSS AMOUNT(USD)"] / (df["FREQUENCY"] + 1e-9)
df["SEV_OPVAR_RATIO"]   = df["SEVERITY"] / (df["OPVAR(%)"] + 1e-9)
df["RISK_COMPOSITE"]    = df["SEVERITY"] * df["FREQUENCY"] * df["OPVAR(%)"]
df["ML_VIX_INTERACT"]   = df["ML"] * df["VIX"]
df["GDP_VIX_RATIO"]     = df["GDP (%)"] / (df["VIX"] + 1e-9)

# Encode categoricals
le_event   = LabelEncoder()
le_process = LabelEncoder()
df["EVENT_ENC"]   = le_event.fit_transform(df["EVENT TYPE"])
df["PROCESS_ENC"] = le_process.fit_transform(df["PROCESS AREA"])

# Prefix from SCENARIO ID (ALE vs OLE)
df["SCENARIO_PREFIX"] = df["SCENARIO ID"].str[:3]
le_scen = LabelEncoder()
df["SCENARIO_ENC"] = le_scen.fit_transform(df["SCENARIO_PREFIX"])

print(f"[Feature Engineering] Event types: {list(le_event.classes_)}")
print(f"[Feature Engineering] Process areas: {list(le_process.classes_)}")
print(f"[Feature Engineering] Features created: LOSS_PER_FREQ, SEV_OPVAR_RATIO, RISK_COMPOSITE, ML_VIX_INTERACT, GDP_VIX_RATIO")

# ─────────────────────────────────────────────
# 3. DEFINE FEATURE SETS
# ─────────────────────────────────────────────
CORE_FEATURES = [
    "FREQUENCY", "SEVERITY", "OPVAR(%)", "ML", "GDP (%)", "VIX",
    "EVENT_ENC", "PROCESS_ENC", "SCENARIO_ENC",
    "MONTH", "QUARTER", "DAY_OF_WEEK",
    "LOSS_PER_FREQ", "SEV_OPVAR_RATIO", "RISK_COMPOSITE",
    "ML_VIX_INTERACT", "GDP_VIX_RATIO"
]

# Drop rows with NaN in core features or target
df_clean = df.dropna(subset=CORE_FEATURES + ["LOSS AMOUNT(USD)", "TARGET"]).copy()
print(f"\n[Clean] Records after dropping NaN: {len(df_clean)}")

X = df_clean[CORE_FEATURES]
y_loss   = df_clean["LOSS AMOUNT(USD)"]
y_target = df_clean["TARGET"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=CORE_FEATURES)

# ─────────────────────────────────────────────
# 4. ANOMALY DETECTION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ANOMALY DETECTION")
print("=" * 60)

# 4a. Isolation Forest
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,    # expect ~5% anomalies
    random_state=42,
    max_features=0.8
)
iso_scores = iso_forest.fit_predict(X_scaled)
iso_raw    = iso_forest.decision_function(X_scaled)   # lower = more anomalous

df_clean = df_clean.copy()
df_clean["ANOMALY_ISO"]   = np.where(iso_scores == -1, 1, 0)
df_clean["ANOMALY_SCORE"] = -iso_raw   # flip so higher = more suspicious

# 4b. Z-score on LOSS AMOUNT
z_scores = np.abs(stats.zscore(df_clean["LOSS AMOUNT(USD)"]))
df_clean["ZSCORE_LOSS"] = z_scores
df_clean["ANOMALY_Z"]   = (z_scores > 2.5).astype(int)

# 4c. Combined flag
df_clean["ANOMALY_COMBINED"] = ((df_clean["ANOMALY_ISO"] == 1) | (df_clean["ANOMALY_Z"] == 1)).astype(int)

n_iso  = df_clean["ANOMALY_ISO"].sum()
n_z    = df_clean["ANOMALY_Z"].sum()
n_comb = df_clean["ANOMALY_COMBINED"].sum()

print(f"\n[Anomaly] Isolation Forest flags : {n_iso}  ({n_iso/len(df_clean)*100:.1f}%)")
print(f"[Anomaly] Z-score flags (>2.5σ)  : {n_z}  ({n_z/len(df_clean)*100:.1f}%)")
print(f"[Anomaly] Combined flags          : {n_comb}  ({n_comb/len(df_clean)*100:.1f}%)")

# Top anomalies
top_anomalies = (df_clean[df_clean["ANOMALY_COMBINED"] == 1]
                 .sort_values("ANOMALY_SCORE", ascending=False)
                 [["SCENARIO ID","DATE","EVENT TYPE","PROCESS AREA",
                   "LOSS AMOUNT(USD)","ANOMALY_SCORE","ZSCORE_LOSS"]]
                 .head(10))
print("\n[Anomaly] Top 10 anomalous records:")
print(top_anomalies.to_string(index=False))

# ─────────────────────────────────────────────
# 5. LOSS REGRESSION (predict LOSS AMOUNT)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("LOSS AMOUNT REGRESSION")
print("=" * 60)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_scaled_df, y_loss, test_size=0.2, random_state=42
)

reg_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
reg_model.fit(X_train_r, y_train_r)
y_pred_r = reg_model.predict(X_test_r)

mae  = mean_absolute_error(y_test_r, y_pred_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2   = r2_score(y_test_r, y_pred_r)

cv_r2 = cross_val_score(reg_model, X_scaled_df, y_loss, cv=5, scoring="r2")

print(f"\n[Regression] MAE  : ${mae:.2f}")
print(f"[Regression] RMSE : ${rmse:.2f}")
print(f"[Regression] R²   : {r2:.4f}")
print(f"[Regression] CV R² (5-fold): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

# Feature importance
feat_imp_r = pd.Series(reg_model.feature_importances_, index=CORE_FEATURES).sort_values(ascending=False)
print("\n[Regression] Top 10 features:")
print(feat_imp_r.head(10).to_string())

# ─────────────────────────────────────────────
# 6. RISK CLASSIFICATION (predict TARGET 0/1/2)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("RISK CLASSIFICATION (TARGET: 0=Low / 1=Medium / 2=High)")
print("=" * 60)

print(f"\n[Classification] Class distribution:\n{y_target.value_counts().sort_index()}")

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_scaled_df, y_target, test_size=0.2, random_state=42, stratify=y_target
)

clf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=3,
    class_weight="balanced",
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
clf_model.fit(X_train_c, y_train_c)
y_pred_c = clf_model.predict(X_test_c)
y_prob_c = clf_model.predict_proba(X_test_c)

f1_macro = f1_score(y_test_c, y_pred_c, average="macro")
cv_f1    = cross_val_score(clf_model, X_scaled_df, y_target, cv=5, scoring="f1_macro")

print(f"\n[Classification] F1 (macro): {f1_macro:.4f}")
print(f"[Classification] CV F1 (5-fold): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
print("\n[Classification] Report:")
print(classification_report(y_test_c, y_pred_c, target_names=["Low(0)","Medium(1)","High(2)"]))
print("[Classification] Confusion Matrix:")
print(confusion_matrix(y_test_c, y_pred_c))

feat_imp_c = pd.Series(clf_model.feature_importances_, index=CORE_FEATURES).sort_values(ascending=False)
print("\n[Classification] Top 10 features:")
print(feat_imp_c.head(10).to_string())

# ─────────────────────────────────────────────
# 7. COMBINED RISK SCORING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMBINED RISK SCORING")
print("=" * 60)

df_clean["PREDICTED_LOSS"]  = reg_model.predict(X_scaled)
df_clean["PREDICTED_TARGET"] = clf_model.predict(X_scaled)
df_clean["PROB_HIGH_RISK"]   = clf_model.predict_proba(X_scaled)[:, 2]

# Composite risk score (0–100)
loss_norm  = (df_clean["PREDICTED_LOSS"] - df_clean["PREDICTED_LOSS"].min()) / \
             (df_clean["PREDICTED_LOSS"].max() - df_clean["PREDICTED_LOSS"].min())
anom_norm  = (df_clean["ANOMALY_SCORE"] - df_clean["ANOMALY_SCORE"].min()) / \
             (df_clean["ANOMALY_SCORE"].max() - df_clean["ANOMALY_SCORE"].min())

df_clean["RISK_SCORE"] = (
    0.40 * loss_norm +
    0.35 * df_clean["PROB_HIGH_RISK"] +
    0.25 * anom_norm
) * 100

# Alert tiers
def alert_tier(row):
    if row["RISK_SCORE"] >= 70:
        return "CRITICAL"
    elif row["RISK_SCORE"] >= 50:
        return "HIGH"
    elif row["RISK_SCORE"] >= 30:
        return "MEDIUM"
    else:
        return "LOW"

df_clean["ALERT_TIER"] = df_clean.apply(alert_tier, axis=1)

print(f"\n[Risk Score] Alert tier distribution:")
print(df_clean["ALERT_TIER"].value_counts())

print("\n[Risk Score] Top 15 highest-risk records:")
top_risk = (df_clean.sort_values("RISK_SCORE", ascending=False)
            [["SCENARIO ID","DATE","EVENT TYPE","PROCESS AREA",
              "LOSS AMOUNT(USD)","PREDICTED_LOSS","PREDICTED_TARGET",
              "PROB_HIGH_RISK","RISK_SCORE","ALERT_TIER"]]
            .head(15))
print(top_risk.to_string(index=False))

# ─────────────────────────────────────────────
# 8. SAVE ARTEFACTS
# ─────────────────────────────────────────────
out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

# Save models
joblib.dump(iso_forest, f"{out_dir}/anomaly_isolation_forest.pkl")
joblib.dump(reg_model,  f"{out_dir}/loss_regression_rf.pkl")
joblib.dump(clf_model,  f"{out_dir}/risk_classifier_rf.pkl")
joblib.dump(scaler,     f"{out_dir}/feature_scaler.pkl")
joblib.dump({"event": le_event, "process": le_process, "scenario": le_scen},
            f"{out_dir}/label_encoders.pkl")

# Save scored dataset
df_clean.to_csv(f"{out_dir}/scored_risk_data.csv", index=False)

# Save model summary report
summary_lines = [
    "OPERATIONAL RISK MODEL — SUMMARY REPORT",
    "=" * 60,
    "",
    "DATA",
    f"  Total records     : {len(df_clean)}",
    f"  Features used     : {len(CORE_FEATURES)}",
    f"  Date range        : {df_clean['DATE'].min().date()} to {df_clean['DATE'].max().date()}",
    "",
    "ANOMALY DETECTION (Isolation Forest + Z-score)",
    f"  Isolation Forest flags : {n_iso}  ({n_iso/len(df_clean)*100:.1f}%)",
    f"  Z-score flags (>2.5σ)  : {n_z}",
    f"  Combined anomalies     : {n_comb}  ({n_comb/len(df_clean)*100:.1f}%)",
    "",
    "LOSS REGRESSION (Random Forest)",
    f"  MAE              : ${mae:.2f}",
    f"  RMSE             : ${rmse:.2f}",
    f"  R²               : {r2:.4f}",
    f"  CV R² (5-fold)   : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}",
    f"  Top feature      : {feat_imp_r.index[0]} ({feat_imp_r.iloc[0]:.4f})",
    "",
    "RISK CLASSIFICATION (Random Forest)",
    f"  F1 macro         : {f1_macro:.4f}",
    f"  CV F1 (5-fold)   : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}",
    f"  Top feature      : {feat_imp_c.index[0]} ({feat_imp_c.iloc[0]:.4f})",
    "",
    "ALERT TIERS",
] + [f"  {k:10s}: {v}" for k, v in df_clean["ALERT_TIER"].value_counts().items()] + [
    "",
    "SAVED FILES",
    "  anomaly_isolation_forest.pkl",
    "  loss_regression_rf.pkl",
    "  risk_classifier_rf.pkl",
    "  feature_scaler.pkl",
    "  label_encoders.pkl",
    "  scored_risk_data.csv",
    "=" * 60,
]

with open(f"{out_dir}/model_summary_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\n" + "=" * 60)
print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
print(f"Output directory: {out_dir}")
print("=" * 60)
