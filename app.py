from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()

# Example model class
def train_model(csv_path):
    data = pd.read_csv(csv_path)

    # Extract features and target
    X = data[['Production Volume (Tonnes)', 'Cultivated Area (Hectares)', 'Rainfall (mm)', 'Temperature (°C)']]
    y_price = data['Wholesale Price (₹/kg)']
    y_demand = data['Domestic Consumption (Tonnes)']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train models
    price_model = RandomForestRegressor(n_estimators=100, random_state=42)
    demand_model = RandomForestRegressor(n_estimators=100, random_state=42)
    price_model.fit(X_scaled, y_price)
    demand_model.fit(X_scaled, y_demand)

    return scaler, price_model, demand_model

# Load the model
scaler, price_model, demand_model = train_model("Tomato-dataset.csv")

class PredictionInput(BaseModel):
    production_volume: float
    cultivated_area: float
    rainfall: float
    temperature: float

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Predict
        predicted_price = price_model.predict(input_scaled)[0]
        predicted_demand = demand_model.predict(input_scaled)[0]

        return {
            "predicted_wholesale_price": round(predicted_price, 2),
            "predicted_domestic_consumption": round(predicted_demand, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))