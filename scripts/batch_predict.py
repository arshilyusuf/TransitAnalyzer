import pandas as pd
import joblib
import os
import numpy as np

model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

df = pd.read_csv("data/processed/final_labeled_ward_data.csv")

feature_cols = [
    "population_density", "stop_density", "route_count",
    "route_density", "avg_fare", "overlap_score", "border_overlap_score"
]

df["predicted_label"] = "N/A"

valid_mask = df[feature_cols].notnull().all(axis=1)

X_valid = df.loc[valid_mask, feature_cols]
X_scaled = scaler.transform(X_valid)

y_pred = model.predict(X_scaled)
decoded_labels = label_encoder.inverse_transform(y_pred)

df.loc[valid_mask, "predicted_label"] = decoded_labels

output_path = "data/processed/wards_with_predictions.csv"
df.to_csv(output_path, index=False)

print(f"✅ Batch predictions saved to {output_path}")
