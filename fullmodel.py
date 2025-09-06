

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


df = pd.read_csv("C:/Users/Pirit/OneDrive/Desktop/CTS MODEL/final_data.csv")


clinical_features = [
    "Age", "BMI", "BP_Systolic", "BP_Diastolic", "Cholesterol",
    "Diabetes", "Hypertension", "Heart_Disease", "Asthma", "Obesity",
    "Smoking_Status","Gender","Blood_Type"
]

df = df[clinical_features].copy()


def compute_risk_points(row):
    points = 0
    if row.get("Age", 0) >= 65: points += 2
    if row.get("Heart_Disease", 0) == 1: points += 3
    if row.get("Diabetes", 0) == 1: points += 2
    if row.get("Hypertension", 0) == 1: points += 1
    if row.get("Asthma", 0) == 1: points += 1
    if row.get("Obesity", 0) == 1: points += 1
    if row.get("BP_Systolic", 0) >= 140: points += 1
    if row.get("BP_Diastolic", 0) >= 90: points += 1
    if row.get("Cholesterol", 0) >= 240: points += 1
    if str(row.get("Smoking_Status", "")).lower() in {"current","smoker","yes","1"}:
        points += 1
    return points

df["Risk_Points"] = df.apply(compute_risk_points, axis=1)
df["Risk_Category"] = pd.cut(
    df["Risk_Points"],
    bins=[-1, 2, 4, 100],
    labels=["Low", "Medium", "High"]
)

X = df.drop(columns=["Risk_Points", "Risk_Category"])
y = df["Risk_Category"]

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

smote = SMOTE(random_state=42)
clinical_clf = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", smote),
    ("model", GradientBoostingClassifier(random_state=42))
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

clinical_clf.fit(X_train, y_train)


y_pred = clinical_clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["Low","Medium","High"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low","Medium","High"],
            yticklabels=["Low","Medium","High"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Clinical Risk Categories (SMOTE)")
plt.show()


proba_clinical = clinical_clf.predict_proba(X_test)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Oversampling
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


df = pd.read_csv("C:/Users/Pirit/OneDrive/Desktop/CTS MODEL/final_data.csv")
clinical_features = [
    "Age", "BMI", "BP_Systolic", "BP_Diastolic", "Cholesterol",
    "Diabetes", "Hypertension", "Heart_Disease", "Asthma", "Obesity",
    "Smoking_Status","Patient_ID","Gender","Blood_Type"
]
social_features = [
    "PovertyRate","MedianFamilyIncome","TractLOWI","TractSNAP",
    "TractHUNV","TractKids","TractSeniors","TractBlack",
    "TractHispanic","Urban","LILATracts_1And10","LILATracts_Vehicle",
    "LATracts1","LATracts10","lahunv1","lahunv10",
    "lasnap1","lasnap10","FIPS","Healthcare_Visits_Last_Year"
]
df_soc = df.drop(columns=[col for col in clinical_features if col in df.columns])
df_soc

from sklearn.preprocessing import OneHotEncoder, StandardScaler
features = [
    "PovertyRate", "MedianFamilyIncome", "TractLOWI", "TractSNAP", "TractHUNV",
    "TractKids", "TractSeniors", "TractBlack", "TractHispanic"
]

# Compute quantile-based thresholds
thresholds = {}
for col in features:
    q33, q66 = df[col].quantile([0.33, 0.66])
    thresholds[col] = (q33, q66)

print("Dynamic thresholds (33% / 66%):")
for k, v in thresholds.items():
    print(f"{k}: {v}")

def compute_social_risk(row):
    points = 0
    for col in features:
        q33, q66 = thresholds[col]
        if row[col] > q66:
            points += 2
        elif row[col] > q33:
            points += 1
   
    if "LILATracts_1And10" in row and row["LILATracts_1And10"] == 1:
        points += 2
    return points

df_soc["Social_Risk_Points"] = df.apply(compute_social_risk, axis=1)


df_soc["Risk_Category"] = pd.cut(
    df_soc["Social_Risk_Points"],
    bins=[-1, 4, 8, 100],
    labels=["Low", "Medium", "High"]
)

print(df_soc["Risk_Category"].value_counts())


X = df_soc.drop(columns=["Risk_Category", "Social_Risk_Points"])
y = df_soc["Risk_Category"]

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)


smote = SMOTE(random_state=42)
social_clf = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", smote),
    ("model", GradientBoostingClassifier(random_state=42))
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)

social_clf.fit(X_train, y_train)


y_pred = social_clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["Low","Medium","High"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low","Medium","High"],
            yticklabels=["Low","Medium","High"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Non Clinical Risk Categories")
plt.show()
proba_social= social_clf.predict_proba(X_test)

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv("C:/Users/Pirit/OneDrive/Desktop/CTS MODEL/Stackmodel_dataset.csv")
X_clinical = df.drop(columns=["Risk_Category", "Risk_Points"])  
X_social   = df.drop(columns=["Risk_Category", "Social_Risk_Points"])  


X_train_c, X_test_c, y_train, y_test = train_test_split(
    X_clinical, y, stratify=y, test_size=0.2, random_state=42
)
X_train_s, X_test_s, _, _ = train_test_split(
    X_social, y, stratify=y, test_size=0.2, random_state=42
)


proba_clinical_train = clinical_clf.predict_proba(X_train_c)
proba_social_train   = social_clf.predict_proba(X_train_s)

proba_clinical_test = clinical_clf.predict_proba(X_test_c)
proba_social_test   = social_clf.predict_proba(X_test_s)


X_train_meta = np.hstack([proba_clinical_train, proba_social_train])
X_test_meta  = np.hstack([proba_clinical_test, proba_social_test])


meta_features = [
    "Clin_Low", "Clin_Med", "Clin_High",
    "Soc_Low", "Soc_Med", "Soc_High"
]
X_train_meta_df = pd.DataFrame(X_train_meta, columns=meta_features)
X_test_meta_df  = pd.DataFrame(X_test_meta, columns=meta_features)


meta_clf = GradientBoostingClassifier(random_state=42)
meta_clf.fit(X_train_meta_df, y_train)


y_pred_meta = meta_clf.predict(X_test_meta_df)
print("Meta-Model Classification Report:\n", classification_report(y_test, y_pred_meta))

cm = confusion_matrix(y_test, y_pred_meta, labels=["Low","Medium","High"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Low","Medium","High"],
            yticklabels=["Low","Medium","High"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Meta Model")
plt.show()


def predict_meta(new_clinical, new_social):
    """Predict risk category using both models + meta-model"""
    proba_c = clinical_clf.predict_proba(new_clinical)
    proba_s = social_clf.predict_proba(new_social)
    meta_input = np.hstack([proba_c, proba_s])
    final_pred = meta_clf.predict(meta_input)
    final_proba = meta_clf.predict_proba(meta_input)
    return final_pred, final_proba
import joblib


model_bundle = {
    "clinical_model": clinical_clf,
    "social_model": social_clf,
    "meta_model": meta_clf,
    "meta_features": meta_features,
    "clinical_feature_names": clinical_features,
    "social_feature_names": social_features
}


joblib.dump(model_bundle, "meta_risk_model.joblib")
print("✅ Meta-model and base models saved to meta_risk_model.joblib")


loaded_bundle = joblib.load("meta_risk_model.joblib")

clinical_loaded = loaded_bundle["clinical_model"]
social_loaded   = loaded_bundle["social_model"]
meta_loaded     = loaded_bundle["meta_model"]
meta_features   = loaded_bundle["meta_features"]

print("✅ Models loaded successfully!")

clinical_features = X_clinical.columns.tolist()
social_features   = X_social.columns.tolist()
import numpy as np
import pandas as pd
import shap

# --- helpers ---
def _is_tree_model(est):
    name = est.__class__.__name__.lower()
    return any(k in name for k in ["randomforest", "gradientboost", "xgb", "lgbm", "catboost"])

def _safe_tree_shap(estimator, X_proc, pred_label=None, background_proc=None):
    """
    Try TreeExplainer; if unsupported, fall back to KernelExplainer.
    Returns a 1D SHAP array for the predicted class.
    """
    try:
        expl = shap.TreeExplainer(estimator)
        sv = expl.shap_values(X_proc)
        # handle binary vs multiclass
        if isinstance(sv, list):
            if pred_label is None:
                pred_label = estimator.predict(X_proc)[0]
            class_idx = list(estimator.classes_).index(pred_label)
            sv = sv[class_idx][0]
        else:
            sv = sv[0]
        return sv
    except Exception:
       
        if background_proc is None:
        
            background_proc = np.repeat(X_proc, repeats=20, axis=0)
        f = lambda x: estimator.predict_proba(x)
        kexpl = shap.KernelExplainer(f, background_proc)
        sv_list = kexpl.shap_values(X_proc)
        if isinstance(sv_list, list):
            if pred_label is None:
                pred_label = estimator.predict(X_proc)[0]
            class_idx = list(estimator.classes_).index(pred_label)
            sv = sv_list[class_idx][0]
        else:
            sv = sv_list[0]
        return sv

def explain_meta_prediction(new_clinical: pd.DataFrame,
                            new_social: pd.DataFrame,
                            patient_id: str,
                            patient_fips: str) -> str:
    """
    Explain prediction from stacked meta-classifier (meta_clf).
    Assumes globals: df, clinical_features, social_features, clinical_clf, social_clf, meta_clf
    Inputs:
      new_clinical: 1-row DataFrame with columns == clinical_features
      new_social:   1-row DataFrame with columns == social_features
    """

   
    proba_c = clinical_clf.predict_proba(new_clinical)[0]
    proba_s = social_clf.predict_proba(new_social)[0]

    meta_input = np.hstack([proba_c, proba_s]).reshape(1, -1)
    meta_cols = ["Clin_Low","Clin_Med","Clin_High","Soc_Low","Soc_Med","Soc_High"]
    meta_input_df = pd.DataFrame(meta_input, columns=meta_cols)

    final_pred  = meta_clf.predict(meta_input)[0]
    final_proba = meta_clf.predict_proba(meta_input)[0]

   
    try:
        bg_n = min(50, len(df))
        bg_idx = df.sample(n=bg_n, random_state=42).index
        bg_proba_c = clinical_clf.predict_proba(df.loc[bg_idx, clinical_features])
        bg_proba_s = social_clf.predict_proba(df.loc[bg_idx, social_features])
        meta_bg = np.hstack([bg_proba_c, bg_proba_s])
    except Exception:
        meta_bg = np.repeat(meta_input, repeats=50, axis=0)

    kexpl_meta = shap.KernelExplainer(meta_clf.predict_proba, meta_bg)
    shap_vals_meta_list = kexpl_meta.shap_values(meta_input)

    if isinstance(shap_vals_meta_list, list):
        class_idx = list(meta_clf.classes_).index(final_pred)
        shap_values_meta = shap_vals_meta_list[class_idx]
    else:
        shap_values_meta = shap_vals_meta_list

    shap_values_meta = np.array(shap_values_meta).reshape(-1)

    meta_explanations = sorted(
        zip(meta_cols, shap_values_meta),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    
    clf_c = clinical_clf.named_steps["model"]
    pre_c = clinical_clf.named_steps["preprocess"]
    Xc_proc = pre_c.transform(new_clinical)
    pred_c_label = clf_c.predict(Xc_proc)[0]
    try:
        c_bg_raw = df.loc[bg_idx, clinical_features]
        c_bg_proc = pre_c.transform(c_bg_raw)
    except Exception:
        c_bg_proc = np.repeat(Xc_proc, repeats=50, axis=0)

    shap_values_c = _safe_tree_shap(clf_c, Xc_proc, pred_label=pred_c_label, background_proc=c_bg_proc)
    feat_c = pre_c.get_feature_names_out()
    shap_values_c = np.array(shap_values_c).reshape(-1)
    explanations_c = sorted(
        zip(feat_c, shap_values_c),
        key=lambda x: abs(x[1]),
        reverse=True
    )


    clf_s = social_clf.named_steps["model"]
    pre_s = social_clf.named_steps["preprocess"]
    Xs_proc = pre_s.transform(new_social)
    pred_s_label = clf_s.predict(Xs_proc)[0]
    try:
        s_bg_raw = df.loc[bg_idx, social_features]
        s_bg_proc = pre_s.transform(s_bg_raw)
    except Exception:
        s_bg_proc = np.repeat(Xs_proc, repeats=50, axis=0)

    shap_values_s = _safe_tree_shap(clf_s, Xs_proc, pred_label=pred_s_label, background_proc=s_bg_proc)
    feat_s = pre_s.get_feature_names_out()
    shap_values_s = np.array(shap_values_s).reshape(-1)
    explanations_s = sorted(
        zip(feat_s, shap_values_s),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    
    lines = [
        f"--- Meta Risk Report for Patient {patient_id} ---",
        f"FIPS Code: {patient_fips}",
        f"Final Predicted Risk: {final_pred}",
        f"Prediction Probabilities: {dict(zip(meta_clf.classes_, np.round(final_proba, 3)))}",
        "",
        "Meta-Model Contribution (clinical vs. social):"
    ]
    for feat, val in meta_explanations[:4]:
        lines.append(f"- {feat}: {val:.3f} ({'↑' if val > 0 else '↓'})")

    lines.append("\nTop Clinical Factors:")
    for feat, val in explanations_c[:5]:
        lines.append(f"- {feat}: {'increases' if val > 0 else 'reduces'} risk (SHAP={val:.3f})")

    lines.append("\nTop Social Factors:")
    for feat, val in explanations_s[:5]:
        lines.append(f"- {feat}: {'increases' if val > 0 else 'reduces'} risk (SHAP={val:.3f})")

    return "\n".join(lines)


clinical_features = [
    "Age","BMI","BP_Systolic","BP_Diastolic","Cholesterol",
    "Diabetes","Hypertension","Heart_Disease","Asthma",
    "Obesity","Smoking_Status","Gender","Blood_Type"
]

social_features = [
    "PovertyRate","MedianFamilyIncome","TractLOWI","TractSNAP",
    "TractHUNV","TractKids","TractSeniors","TractBlack",
    "TractHispanic","Urban","LILATracts_1And10","LILATracts_Vehicle",
    "LATracts1","LATracts10","lahunv1","lahunv10",
    "lasnap1","lasnap10","FIPS","Healthcare_Visits_Last_Year"
]


patient_row = df.loc[0]  # single row (Series)


patient_clinical = patient_row[clinical_features].to_frame().T
patient_social   = patient_row[social_features].to_frame().T


patient_id       = patient_row["Patient_ID"]
patient_fips     = patient_row["FIPS"]


report = explain_meta_prediction(
    new_clinical=patient_clinical,
    new_social=patient_social,
    patient_id=patient_id,
    patient_fips=patient_fips
)

print(report)
import re
import requests
import google.generativeai as genai


genai.configure(api_key="AIzaSyCo91wCZG1RE4ldx7wK6beiIx0enAHkF8k")




def extract_fips_from_report(report_text):
    """
    Look for a line like 'FIPS Code: XXXXX'
    """
    match = re.search(r"FIPS Code:\s*(\d+)", report_text)
    if match:
        return match.group(1)
    raise ValueError("No FIPS code found in SHAP report")


def fips_to_latlon(fips_code):
    if len(fips_code) > 5:
        fips_code = fips_code[:5]

    state_fips = fips_code[:2]
    county_fips = fips_code[2:].zfill(3)

    # FCC API
    url = f"https://geo.fcc.gov/api/census/area?format=json&fips={state_fips}{county_fips}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "results" in data and len(data["results"]) > 0:
            coords = data["results"][0]["centroid"]
            return coords["y"], coords["x"], "FCC"
    except Exception:
        pass

    # Census + OSM fallback
    try:
        url = f"https://api.census.gov/data/2010/dec/sf1?get=NAME&for=county:{county_fips}&in=state:{state_fips}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if len(data) > 1:
                county_name = data[1][0]
                geo_url = f"https://nominatim.openstreetmap.org/search?format=json&q={county_name}"
                geo_resp = requests.get(geo_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                geo_data = geo_resp.json()
                if geo_data:
                    return float(geo_data[0]["lat"]), float(geo_data[0]["lon"]), "OSM"
    except Exception:
        pass

    state_fallback = {
        "01": (32.8, -86.6),   # Alabama
        "05": (34.8, -92.2),   # Arkansas
        "10": (39.0, -75.5),   # Delaware
        "12": (28.6, -82.4),   # Florida
        "36": (42.9, -75.5),   # New York
        "48": (31.0, -99.9),   # Texas
    }
    if state_fips in state_fallback:
        return state_fallback[state_fips][0], state_fallback[state_fips][1], "State Fallback"

    raise RuntimeError(f"Could not resolve lat/lon for FIPS {fips_code}")



def get_nearby_resources(lat, lon, radius_km=20):
    query = f"""
        [out:json];
        (
          node["amenity"="hospital"](around:{radius_km*1000},{lat},{lon});
          node["amenity"="clinic"](around:{radius_km*1000},{lat},{lon});
          node["amenity"="social_facility"](around:{radius_km*1000},{lat},{lon});
          node["amenity"="doctors"](around:{radius_km*1000},{lat},{lon});
          node["amenity"="pharmacy"](around:{radius_km*1000},{lat},{lon});
        );
        out center;
    """
    url = "https://overpass-api.de/api/interpreter"
    try:
        resp = requests.post(url, data={"data": query}, timeout=30)
        data = resp.json()
        results = []
        for el in data.get("elements", []):
            name = el.get("tags", {}).get("name", "Unnamed")
            amenity = el.get("tags", {}).get("amenity", "Unknown")
            results.append(f"{name} ({amenity})")
        return results[:10] if results else ["No resources found nearby"]
    except Exception:
        return ["Error fetching resources"]

def extract_fips_and_risk(report_text):
    """Extract FIPS, predicted risk, and probabilities from report text."""
    fips_code, final_pred, final_probs = None, None, {}

    for line in report_text.splitlines():
        if line.startswith("FIPS Code:"):
            fips_code = line.split("FIPS Code:")[1].strip()
        elif line.startswith("Final Predicted Risk:"):
            final_pred = line.split("Final Predicted Risk:")[1].strip()
        elif line.startswith("Prediction Probabilities:"):
            try:
                probs_text = line.split("Prediction Probabilities:")[1].strip()
                final_probs = json.loads(probs_text.replace("np.float64", "").replace("'", '"'))
            except Exception:
                final_probs = {}

  
    if final_probs:
        final_probs = {k: round(float(v) * 100, 2) for k, v in final_probs.items()}

    return fips_code, final_pred, final_probs



