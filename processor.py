# main.py

import pandas as pd
import joblib
import shap
import numpy as np

def load_pipeline(model_path='model_(pipeline).pkl'):
    return joblib.load(model_path)

def parse_feature_name(name, original_row):
    """
    Convert encoded feature name to user-friendly description.
    Example: 'cat__Weather_Condition_Fog' ➝ 'Weather Condition = Fog'
    """
    if '__' in name:
        _, raw = name.split('__', 1)
        if '_' in raw:
            col, val = raw.split('_', 1)
            original_val = original_row[col] if col in original_row else val
            return f"{col.replace('_', ' ')} = {original_val} ({val})"
        else:
            return raw
    else:
        # numeric column
        return name.replace('_', ' ')

def predict_and_explain(input_csv, model_path='model_(pipeline).pkl'):
    # Load model + input
    pipeline = load_pipeline(model_path)
    input_df = pd.read_csv(input_csv)

    # Predict
    prediction_proba = pipeline.predict_proba(input_df)[0][1]
    prediction_label = pipeline.predict(input_df)[0]

    # SHAP explanation
    transformed = pipeline.named_steps['pre'].transform(input_df)
    explainer = shap.TreeExplainer(pipeline.named_steps['clf'])
    shap_values = explainer.shap_values(transformed)

    # Get feature names and values
    feature_names = pipeline.named_steps['pre'].get_feature_names_out()
    shap_row = shap_values[0]
    input_row = input_df.iloc[0].to_dict()

    # Top 2 features (absolute SHAP value)
    top_indices = np.argsort(np.abs(shap_row))[-2:][::-1]
    top_features = []
    for idx in top_indices:
        raw_name = feature_names[idx]
        shap_val = shap_row[idx]
        explanation = parse_feature_name(raw_name, input_row)
        top_features.append((raw_name, shap_val, explanation))

    return {
        "prediction_label": "Incident" if prediction_label == 1 else "No Incident",
        "prediction_probability": round(prediction_proba, 4),
        "top_features": top_features
    }