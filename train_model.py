# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

print("Starting model training process...")

# --- Load Data ---
# Use relative paths for portability
df_full = pd.read_csv("final_data.csv")
df_stack = pd.read_csv("Stackmodel_dataset.csv")

# ==============================================================================
# 1. TRAIN CLINICAL MODEL
# ==============================================================================
print("1. Training Clinical Model...")
clinical_features = [
    "Age", "BMI", "BP_Systolic", "BP_Diastolic", "Cholesterol",
    "Diabetes", "Hypertension", "Heart_Disease", "Asthma", "Obesity",
    "Smoking_Status", "Gender", "Blood_Type"
]
df_clin = df_full[clinical_features].copy()

# Simple rule-based risk category for training label
def compute_risk_points(row):
    points = 0
    if row.get("Age", 0) >= 65: points += 2
    if row.get("Heart_Disease", 0) == 1: points += 3
    if row.get("Diabetes", 0) == 1: points += 2
    if row.get("Hypertension", 0) == 1: points += 1
    if str(row.get("Smoking_Status", "")).lower() in {"current","smoker","yes","1"}: points += 1
    return points

df_clin["Risk_Points"] = df_clin.apply(compute_risk_points, axis=1)
df_clin["Risk_Category"] = pd.cut(df_clin["Risk_Points"], bins=[-1, 2, 4, 100], labels=["Low", "Medium", "High"])

X_clin = df_clin.drop(columns=["Risk_Points", "Risk_Category"])
y_clin = df_clin["Risk_Category"]

numeric_cols_clin = X_clin.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols_clin = X_clin.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))])
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor_clin = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_cols_clin), ("cat", categorical_transformer, categorical_cols_clin)])

clinical_clf = ImbPipeline(steps=[("preprocess", preprocessor_clin), ("smote", SMOTE(random_state=42)), ("model", GradientBoostingClassifier(random_state=42))])
clinical_clf.fit(X_clin, y_clin)
print("   Clinical Model Trained.")

# ==============================================================================
# 2. TRAIN SOCIAL MODEL
# ==============================================================================
print("2. Training Social Model...")
social_features_all = [
    "PovertyRate", "MedianFamilyIncome", "TractLOWI", "TractSNAP", "TractHUNV",
    "TractKids", "TractSeniors", "TractBlack", "TractHispanic", "Urban",
    "LILATracts_1And10", "LILATracts_Vehicle", "LATracts1", "LATracts10",
    "lahunv1", "lahunv10", "lasnap1", "lasnap10", "FIPS", "Healthcare_Visits_Last_Year"
]
df_soc = df_full[social_features_all].copy()
social_risk_features = [
    "PovertyRate", "MedianFamilyIncome", "TractLOWI", "TractSNAP", "TractHUNV",
    "TractKids", "TractSeniors", "TractBlack", "TractHispanic"
]
thresholds = {col: tuple(df_soc[col].quantile([0.33, 0.66])) for col in social_risk_features}

def compute_social_risk(row):
    points = 0
    for col in social_risk_features:
        q33, q66 = thresholds[col]
        if row[col] > q66: points += 2
        elif row[col] > q33: points += 1
    return points

df_soc["Social_Risk_Points"] = df_soc.apply(compute_social_risk, axis=1)
df_soc["Risk_Category"] = pd.cut(df_soc["Social_Risk_Points"], bins=[-1, 2, 4, 100], labels=["Low", "Medium", "High"])

X_soc = df_soc.drop(columns=["Risk_Category", "Social_Risk_Points"])
y_soc = df_soc["Risk_Category"]

numeric_cols_soc = X_soc.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols_soc = X_soc.select_dtypes(exclude=[np.number]).columns.tolist()
preprocessor_soc = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_cols_soc), ("cat", categorical_transformer, categorical_cols_soc)])
social_clf = ImbPipeline(steps=[("preprocess", preprocessor_soc), ("smote", SMOTE(random_state=42)), ("model", GradientBoostingClassifier(random_state=42))])
social_clf.fit(X_soc, y_soc)
print("   Social Model Trained.")


# ==============================================================================
# 3. TRAIN META MODEL (STACKING)
# ==============================================================================
print("3. Training Meta Model...")
y_meta = df_stack["Risk_Category"]
X_train_c, X_test_c, y_train, y_test = train_test_split(df_stack[clinical_features], y_meta, stratify=y_meta, test_size=0.2, random_state=42)
X_train_s, X_test_s, _, _ = train_test_split(df_stack[social_features_all], y_meta, stratify=y_meta, test_size=0.2, random_state=42)

proba_clinical_train = clinical_clf.predict_proba(X_train_c)
proba_social_train = social_clf.predict_proba(X_train_s)
X_train_meta = np.hstack([proba_clinical_train, proba_social_train])

meta_features = ["Clin_Low", "Clin_Med", "Clin_High", "Soc_Low", "Soc_Med", "Soc_High"]
X_train_meta_df = pd.DataFrame(X_train_meta, columns=meta_features)

meta_clf = GradientBoostingClassifier(random_state=42)
meta_clf.fit(X_train_meta_df, y_train)
print("   Meta Model Trained.")


# ==============================================================================
# 4. BUNDLE AND SAVE MODELS
# ==============================================================================
print("4. Bundling and saving models...")
model_bundle = {
    "clinical_model": clinical_clf,
    "social_model": social_clf,
    "meta_model": meta_clf,
    "meta_features": meta_features,
    "clinical_feature_names": list(X_clin.columns),
    "social_feature_names": list(X_soc.columns)
}
joblib.dump(model_bundle, "meta_risk_model.joblib")
print("\n✅ Models trained and saved successfully to meta_risk_model.joblib")