import geopandas as gpd

wards = gpd.read_file("../geo/delhi_wards.geojson")

wards = wards.to_crs(epsg=32643)

wards["area_km2"] = wards.geometry.area / 1e6
print("Wards area calculated :)")
import pandas as pd
import geopandas as gpd
import re

pop_df = pd.read_csv("../data/external/delhi-ward-wise-population.csv")

pop_df["Ward_No_Clean"] = pop_df["Ward"].str.extract(r'(\d+)$')
pop_df["Ward_No_Clean"] = pop_df["Ward_No_Clean"].astype(float)  

wards = gpd.read_file("../geo/delhi_wards.geojson")

wards["Ward_No_Clean"] = wards["Ward_No"].apply(
    lambda x: float(re.search(r"\d+", str(x)).group()) if pd.notnull(x) and re.search(r"\d+", str(x)) else None
)


wards = wards.to_crs(epsg=32643)

wards["area_km2"] = wards.geometry.area / 1e6

merged = wards.merge(pop_df[["Ward_No_Clean", "Population"]], on="Ward_No_Clean", how="left")

merged["population_density"] = merged["Population"] / merged["area_km2"]

merged.to_file("../data/processed/wards_with_population.geojson", driver="GeoJSON")
merged[["Ward_Name", "Population", "area_km2", "population_density"]].to_csv(
    "../data/processed/wards_with_population.csv", index=False
)

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

wards = gpd.read_file("../data/processed/wards_with_population.geojson")
stops_df = pd.read_csv("../data/kaggle/dmrc_gtfs_dataset/stops.csv")
stop_times = pd.read_csv("../data/kaggle/dmrc_gtfs_dataset/stop_times.csv")
trips = pd.read_csv("../data/kaggle/dmrc_gtfs_dataset/trips.csv")
routes = pd.read_csv("../data/kaggle/dmrc_gtfs_dataset/routes.csv")
fare_rules = pd.read_csv("../data/kaggle/dmrc_gtfs_dataset/fare_rules.csv")
fare_attributes = pd.read_csv("../data/kaggle/dmrc_gtfs_dataset/fare_attributes.csv")

stops_gdf = gpd.GeoDataFrame(
    stops_df,
    geometry=gpd.points_from_xy(stops_df["stop_lon"], stops_df["stop_lat"]),
    crs="EPSG:4326"
)
stops_gdf = stops_gdf.to_crs(epsg=32643)
wards = wards.to_crs(epsg=32643)

stops_in_wards = gpd.sjoin(stops_gdf, wards, how="inner", predicate="within")

stop_counts = stops_in_wards.groupby("Ward_Name").size().reset_index(name="stop_count")
wards = wards.drop(columns=[col for col in wards.columns if "stop_count" in col], errors="ignore")
wards = wards.merge(stop_counts, on="Ward_Name", how="left")
wards["stop_count"] = wards["stop_count"].fillna(0)
wards["stop_density"] = wards["stop_count"] / wards["area_km2"]

stop_routes = stop_times.merge(trips[["trip_id", "route_id"]], on="trip_id", how="left")
stop_route_map = stop_routes[["stop_id", "route_id"]].drop_duplicates()
stops_in_wards_simple = stops_in_wards[["stop_id", "Ward_Name"]]
stop_route_ward = stop_route_map.merge(stops_in_wards_simple, on="stop_id", how="inner")
stop_route_ward = stop_route_ward.dropna(subset=["route_id"])
route_counts = stop_route_ward.groupby("Ward_Name")["route_id"].nunique().reset_index()
route_counts.columns = ["Ward_Name", "route_count"]
wards = wards.drop(columns=[col for col in wards.columns if "route_count" in col], errors="ignore")
wards = wards.merge(route_counts, on="Ward_Name", how="left")
wards["route_count"] = wards["route_count"].fillna(0)
wards["route_density"] = wards["route_count"] / wards["area_km2"]

overlap = stop_route_ward.groupby(["Ward_Name", "stop_id"]).size().reset_index(name="route_per_stop")
overlap_score = overlap.groupby("Ward_Name")["route_per_stop"].mean().reset_index(name="overlap_score")
wards = wards.merge(overlap_score, on="Ward_Name", how="left")
wards["overlap_score"] = wards["overlap_score"].fillna(0)

ward_counts = stops_in_wards.groupby("stop_id")["Ward_Name"].nunique().reset_index(name="ward_count")
shared_stops = ward_counts[ward_counts["ward_count"] > 1]["stop_id"]
stops_multi = stops_in_wards[stops_in_wards["stop_id"].isin(shared_stops)]
border_overlap = stops_multi.groupby("Ward_Name").size().reset_index(name="border_overlap")
wards = wards.merge(border_overlap, on="Ward_Name", how="left")
wards["border_overlap"] = wards["border_overlap"].fillna(0)
wards["border_overlap_score"] = wards["border_overlap"] / wards["stop_count"].replace(0, np.nan)
wards["border_overlap_score"] = wards["border_overlap_score"].fillna(0)

fare_map = fare_rules.merge(fare_attributes, on="fare_id", how="left")
fare_map = fare_map[["route_id", "price"]].dropna()
route_fare = stop_route_ward.merge(fare_map, on="route_id", how="left")
fare_avg = route_fare.groupby("Ward_Name")["price"].mean().reset_index(name="avg_fare")
wards = wards.merge(fare_avg, on="Ward_Name", how="left")
wards["avg_fare"] = wards["avg_fare"].fillna(0)

wards.to_file("../data/processed/wards_enriched.geojson", driver="GeoJSON")
wards.to_csv("../data/processed/wards_enriched.csv", index=False)

print(" Feature engineering complete with advanced features!")


import pandas as pd

df = pd.read_csv("../data/processed/wards_enriched.csv")
df = df.fillna({
    "population_density": 0,
    "stop_density": 0,
    "route_density": 0,
    "overlap_score": 0,
    "border_overlap_score": 0
})
def label_ward(row):
    pop_density = row["population_density"]
    stop_density = row["stop_density"]
    route_density = row["route_density"]
    overlap = row["overlap_score"]
    border_overlap = row["border_overlap_score"]

    high_pop = pop_density > 20000
    med_pop = 5000 < pop_density <= 20000
    low_pop = pop_density <= 5000

    low_stops = stop_density < 2.5
    high_stops = stop_density > 6

    low_routes = route_density < 2.5
    high_routes = route_density > 4.5

    high_overlap = overlap > 0.6
    med_overlap = 0.3 < overlap <= 0.6
    low_overlap = overlap <= 0.3


    if (high_pop or med_pop) and (low_stops or low_routes):
        return "underserved"

    if (low_pop or med_pop) and (high_stops or high_routes) and high_overlap:
        return "cluttered"

    if high_stops and high_overlap:
        return "cluttered"

    if (
        not low_stops and not low_routes and
        low_overlap
    ):
        return "optimal"

    return "optimal"

df["label"] = df.apply(label_ward, axis=1)

df.to_csv("../data/processed/final_labeled_ward_data.csv", index=False)
print("✅ Clean and practical labels assigned.")

df = pd.read_csv("../data/processed/final_labeled_ward_data.csv")
features = [
    "population_density", "stop_density", "route_count", "route_density",
    "overlap_score", "border_overlap_score", "avg_fare"
]
X = df[features]
y = df["label"]
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

le = LabelEncoder()
y_encoded = le.fit_transform(y)

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
)

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

for name, model in models.items():
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
import joblib
import os

os.makedirs("../models", exist_ok=True)

joblib.dump(models["XGBoost"], "../models/xgboost_model.pkl")

joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(le, "../models/label_encoder.pkl")

print(" Model, Scaler, and LabelEncoder saved successfully.")
