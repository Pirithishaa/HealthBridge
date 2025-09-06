import pandas as pd
import numpy as np
import re
import joblib

def extract_features_from_report(report_path):
    """
    Extracts the initial risk level, score, and needed interventions from a report file.
    """
    try:
        with open(report_path, "r", encoding="utf-8") as file:
            report_text = file.read()
    except FileNotFoundError:
        print(f"Error: Report file not found at '{report_path}'")
        return None, None, None

    # 1. Extract the initial risk level (Low, Medium, High)
    risk_level_match = re.search(r"Predicted Risk:\s*(\w+)", report_text, re.IGNORECASE)
    risk_level = risk_level_match.group(1).lower() if risk_level_match else "unknown"

    if risk_level == "low":
        print(f"Initial risk level is '{risk_level.capitalize()}'. No intervention analysis needed.")
        return "low", None, None

    # 2. Extract the initial risk score (percentage)
    risk_score_match = re.search(r"\(Probabilities:.*?\'Low\':\s*\'([\d.]+)%\'", report_text, re.IGNORECASE)
    risk_score = 100.0 - float(risk_score_match.group(1)) if risk_score_match else 50.0 # Default if not found

    # 3. Extract needed interventions
    interventions_map = {
        "Nutrition": "Intervention_Nutrition",
        "Smoking": "Intervention_Smoking",
        "Exercise": "Intervention_Exercise",
        "Financial": "Intervention_FinancialAid",
        "Mental": "Intervention_MentalHealth",
        "Community": "Intervention_CommunitySupport",
        "Environmental": "Intervention_Environmental"
    }
    
    extracted_features = {"Overall_Risk_Before": risk_score}
    for keyword, feature_name in interventions_map.items():
        # Check if the keyword is followed by "Needed" or "Recommended"
        pattern = re.compile(rf"{keyword}.*?(Needed|Recommended)", re.IGNORECASE | re.DOTALL)
        if pattern.search(report_text):
            extracted_features[feature_name] = 1
        else:
            extracted_features[feature_name] = 0
            
    # Define all possible feature columns the model expects
    all_feature_names = [
        'Overall_Risk_Before', 'Intervention_Nutrition', 'Intervention_Smoking',
        'Intervention_Exercise', 'Intervention_FinancialAid', 'Intervention_MentalHealth',
        'Intervention_CommunitySupport', 'Intervention_Environmental'
    ]

    # Create a DataFrame with all columns, filling missing ones with 0
    input_df = pd.DataFrame([extracted_features])
    for col in all_feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
            
    return risk_level, risk_score, input_df[all_feature_names] # Ensure correct column order

def main():
    """Main execution function."""
    print("=" * 60)
    print("INTERVENTION IMPACT ANALYSIS SYSTEM")
    print("=" * 60)

    try:
        model = joblib.load("intervention_model.joblib")
        print("✅ Pre-trained intervention model loaded successfully.")
    except FileNotFoundError:
        print("\n❌ Error: 'intervention_model.joblib' not found.")
        print("Please run the 'train_impact_model.py' script first to train and save the model.")
        return

    report_file = "Medical_Report_PT021000.txt" # The report to analyze
    print(f"\n--- Analyzing report: {report_file} ---")

    initial_risk, initial_score, features_df = extract_features_from_report(report_file)

    if initial_risk == "low" or features_df is None:
        print("\nAnalysis complete.")
        return

    print("\nExtracted Features:")
    print(features_df.to_string(index=False))

    # Predict the improved risk score
    predicted_risk_after = model.predict(features_df)[0]
    
    print("\n" + "=" * 60)
    print("FINAL ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Initial Risk Score (from report): {initial_score:.1f}%")
    print(f"Predicted Risk After Interventions: {predicted_risk_after:.1f}%")
    
    reduction = initial_score - predicted_risk_after
    print(f"Predicted Overall Risk Reduction: {reduction:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
