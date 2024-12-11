import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
api = Api(app)

class MarketPriceDemandForecaster:
    def __init__(self, csv_path):
        self.original_data = pd.read_csv(csv_path)
        required_columns = [
            'Region', 'Season', 'Production Volume (Tonnes)', 
            'Cultivated Area (Hectares)', 'Yield (kg/ha)', 
            'Irrigation Coverage (%)', 'Seed Type', 
            'Fertilizer Usage (kg/ha)', 'Pesticide Usage (L/ha)', 
            'Storage Availability', 'Rainfall (mm)', 'Temperature (\u00b0C)', 
            'Humidity (%)', 'Labor Availability', 
            'Wholesale Price (\u20b9/kg)', 'Domestic Consumption (Tonnes)'
        ]

        missing_columns = [col for col in required_columns if col not in self.original_data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in CSV: {missing_columns}")
        
        self.prepare_data()

    def prepare_data(self):
        categorical_cols = ['Region', 'Season', 'Seed Type', 'Storage Availability']
        self.data = pd.get_dummies(self.original_data, columns=categorical_cols)

        self.target_columns = ['Wholesale Price (\u20b9/kg)', 'Domestic Consumption (Tonnes)']
        feature_columns = [
            col for col in self.data.columns 
            if col not in self.target_columns and 
            col not in categorical_cols
        ] + [col for col in self.data.columns if col.startswith(tuple(categorical_cols))]

        self.X = self.data[feature_columns]
        self.y_wholesale_price = self.data['Wholesale Price (\u20b9/kg)']
        self.y_domestic_consumption = self.data['Domestic Consumption (Tonnes)']

    def train_models(self, random_state=42):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)

        self.price_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        self.demand_model = RandomForestRegressor(n_estimators=100, random_state=random_state)

        self.price_model.fit(X_scaled, self.y_wholesale_price)
        self.demand_model.fit(X_scaled, self.y_domestic_consumption)

        return self

    def predict(self, new_data):
        new_data_processed = pd.get_dummies(new_data)
        missing_cols = set(self.X.columns) - set(new_data_processed.columns)
        for col in missing_cols:
            new_data_processed[col] = 0
        new_data_processed = new_data_processed[self.X.columns]

        new_data_scaled = self.scaler.transform(new_data_processed)

        predicted_prices = self.price_model.predict(new_data_scaled)
        predicted_demand = self.demand_model.predict(new_data_scaled)

        return predicted_prices[0], predicted_demand[0]

    def get_unique_values(self):
        return {
            'regions': self.original_data['Region'].unique().tolist(),
            'seasons': self.original_data['Season'].unique().tolist(),
            'seed_types': self.original_data['Seed Type'].unique().tolist(),
            'storage_availability': self.original_data['Storage Availability'].unique().tolist()
        }

CSV_PATH = 'Tomato-dataset.csv'
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found: {os.path.abspath(CSV_PATH)}")

forecaster = MarketPriceDemandForecaster(CSV_PATH).train_models()

class UniqueValuesResource(Resource):
    def get(self):
        try:
            unique_values = forecaster.get_unique_values()
            return jsonify(unique_values)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

class PredictionResource(Resource):
    def post(self):
        try:
            input_data = request.get_json()
            input_dict = {
                'Region': input_data['region'],
                'Season': input_data['season'],
                'Production Volume (Tonnes)': input_data['production_volume'],
                'Cultivated Area (Hectares)': input_data['cultivated_area'],
                'Yield (kg/ha)': input_data['yield_per_ha'],
                'Irrigation Coverage (%)': input_data['irrigation_coverage'],
                'Seed Type': input_data['seed_type'],
                'Fertilizer Usage (kg/ha)': input_data['fertilizer_usage'],
                'Pesticide Usage (L/ha)': input_data['pesticide_usage'],
                'Storage Availability': input_data['storage_availability'],
                'Rainfall (mm)': input_data['rainfall'],
                'Temperature (\u00b0C)': input_data['temperature'],
                'Humidity (%)': input_data['humidity'],
                'Labor Availability': input_data['labor_availability']
            }

            input_df = pd.DataFrame([input_dict])
            predicted_price, predicted_demand = forecaster.predict(input_df)

            return jsonify({
                "predicted_wholesale_price": round(predicted_price, 2),
                "predicted_domestic_consumption": round(predicted_demand, 2)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400

api.add_resource(UniqueValuesResource, '/unique-values')
api.add_resource(PredictionResource, '/predict')

if __name__ == "__main__":
    app.run(debug=True)
