# TransitAnalyzer

TransitAnalyzer is a geospatial machine learning project that analyzes public transport connectivity at the ward level. Trained on Delhi's GTFS transit and population data, it classifies wards as **Underserved**, **Optimal**, or **Cluttered** based on route and stop density, population, and spatial features. The web app uses a pre-trained model to display predictions and insights on an interactive map using Streamlit.
## About the data set and ml training
- Utilizes GTFS transit data and ward-wise population statistics
- Engineers features like stop density, route overlap, and population density
- Applies a trained multi-class classification model (e.g., Random Forest) to categorize wards as Underserved, Optimal, or Cluttered


## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
## Project Structure
app.py                    # Streamlit app
models/transit_model.pkl # Trained model
data/processed/           # CSVs with ward-level features and predictions
geo/                      # GeoJSON of ward boundaries
