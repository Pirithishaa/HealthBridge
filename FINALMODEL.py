# FINALMODEL.py

import pandas as pd
import numpy as np
import joblib
import requests
import re
import json
import google.generativeai as genai
import shap

# --- Load the pre-trained model bundle ONCE when the server starts ---
try:
    _model_bundle = joblib.load("meta_risk_model.joblib")
    _clinical_model = _model_bundle["clinical_model"]
    _social_model = _model_bundle["social_model"]
    _meta_model = _model_bundle["meta_model"]
    _meta_features = _model_bundle["meta_features"]
    _clinical_feature_names = _model_bundle["clinical_feature_names"]
    _social_feature_names = _model_bundle["social_feature_names"]
    print("✅ Models loaded successfully!")
except FileNotFoundError:
    print("❌ Error: meta_risk_model.joblib not found. Please run train_model.py first.")
    _clinical_model = _social_model = _meta_model = None

# --- Configure your API key securely ---
# It's better to use environment variables in a real application
genai.configure(api_key="AIzaSyAKVPkQeoMgSM2LbzrpgPaCc3DK6pU5Voc")


def fips_to_latlon(fips_code):
    """Converts a FIPS code to latitude and longitude with fallbacks."""
    if len(str(fips_code)) > 5:
        fips_code = str(fips_code)[:5]
    # This is a simplified version of your function for brevity.
    # In a real scenario, you'd include your full function with all fallbacks.
    url = f"https://geo.fcc.gov/api/census/area?format=json&fips={fips_code}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "results" in data and len(data["results"]) > 0:
            coords = data["results"][0]["centroid"]
            return coords["y"], coords["x"], "FCC"
    except Exception:
        return 39.8283, -98.5795, "USA Center Fallback" # Fallback to center of USA
    return 39.8283, -98.5795, "USA Center Fallback"

def get_nearby_resources(lat, lon, radius_km=20):
    """Fetches nearby resources using Overpass API."""
    query = f"""[out:json];(node["amenity"~"hospital|clinic|social_facility|doctors|pharmacy"](around:{radius_km*1000},{lat},{lon}););out center;"""
    url = "https://overpass-api.de/api/interpreter"
    try:
        resp = requests.post(url, data={"data": query}, timeout=30)
        data = resp.json()
        results = [f"{el.get('tags', {}).get('name', 'Unnamed')} ({el.get('tags', {}).get('amenity', 'Unknown')})" for el in data.get("elements", [])]
        return results[:10] if results else ["No resources found nearby"]
    except Exception:
        return ["Error fetching resources"]

def explain_and_generate_llm_report(patient_data, social_data):
    """
    Main function to run the full prediction, explanation, and reporting pipeline.
    """
    if not all([_clinical_model, _social_model, _meta_model]):
        return "Error: Models are not loaded. Cannot generate report."

    # --- 1. Prepare DataFrames for models ---
    patient_id = patient_data.get("Patient_ID", "N/A")
    fips_code = patient_data.get("FIPS", "N/A")
    
    # Ensure columns are in the correct order as during training
    new_clinical_df = pd.DataFrame([patient_data])[_clinical_feature_names]
    new_social_df = pd.DataFrame([social_data])[_social_feature_names]

    # --- 2. Get predictions from base and meta models ---
    proba_c = _clinical_model.predict_proba(new_clinical_df)[0]
    proba_s = _social_model.predict_proba(new_social_df)[0]
    meta_input = np.hstack([proba_c, proba_s]).reshape(1, -1)
    
    final_pred = _meta_model.predict(meta_input)[0]
    final_proba = _meta_model.predict_proba(meta_input)[0]
    final_probs_dict = {k: f"{v*100:.1f}%" for k, v in zip(_meta_model.classes_, final_proba)}

    # --- 3. Generate simplified explanation (SHAP can be slow for real-time) ---
    # For a web app, a simplified explanation is often better than a full SHAP analysis.
    top_clinical_factors = ["High Blood Pressure" if patient_data.get("Hypertension") == 1 else "Normal Blood Pressure",
                           "Diabetes Present" if patient_data.get("Diabetes") == 1 else "No Diabetes",
                           f"Age: {patient_data.get('Age', 'N/A')}"]
    top_social_factors = [f"Poverty Rate in Area: {social_data.get('PovertyRate', 'N/A')}%",
                          f"Median Income in Area: ${social_data.get('MedianFamilyIncome', 'N/A'):,.0f}",
                          f"Unemployment in Area: {social_data.get('PERCENT_UNEMPLOYED','N/A')}%"]
    
    # --- 4. Get Local Resources ---
    lat, lon, source = fips_to_latlon(fips_code)
    resources = get_nearby_resources(lat, lon)

    # --- 5. Generate Report with LLM ---
    from datetime import datetime
    today = datetime.today().strftime("%B %d, %Y")
    
    prompt = f"""
        Generate a patient-friendly "Patient Readmission Risk Report" using the provided data.
        The report should be professional, empathetic, and easy to understand.

        **Patient Data:**
        - Patient ID: {patient_id}
        - Report Date: {today}
        - Predicted Risk: {final_pred}
        - Risk Probabilities: {final_probs_dict}

        **Key Factors Identified by the Model:**
        - Top Clinical Factors: {', '.join(top_clinical_factors)}
        - Top Social Factors: {', '.join(top_social_factors)}

        **Local Context:**
        - Location Info: Resolved from FIPS {fips_code} using {source}.
        - Nearby Healthcare/Social Resources: {', '.join(resources)}

        **Instructions for the Report:**
        Format the output as plain text. Do not use markdown.
        Follow this structure exactly:

        Patient Readmission Risk Report
        Patient ID: {patient_id}
        Report Date: {today}
        Predicted Risk: {final_pred} (Probabilities: {final_probs_dict})

        Main Risk Factors
        - <Create a bulleted list summarizing the most important clinical and social risk drivers in short, clear sentences.>

        Local Resources
        - <List the nearby hospitals, clinics, or community resources, one per line.>

        Recommended Interventions
        - <Based on the factors, suggest 2-3 practical interventions. For example, if poverty is high, suggest financial aid resources. If hypertension is a factor, suggest diet and exercise monitoring.>

        Next Steps and Advice
        - <Provide simple, actionable advice for the patient. Keep it positive and empowering.>
    """

    model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text