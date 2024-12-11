import os
import io
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

class MarketPriceDemandForecaster:
    def __init__(self, csv_path):
        """
        Initialize the forecaster with agricultural market data from CSV
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file containing market data
        """
        # Read data from CSV
        self.original_data = pd.read_csv(csv_path)
        
        # Basic data validation
        required_columns = [
            'Region', 'Season', 'Production Volume (Tonnes)', 
            'Cultivated Area (Hectares)', 'Yield (kg/ha)', 
            'Irrigation Coverage (%)', 'Seed Type', 
            'Fertilizer Usage (kg/ha)', 'Pesticide Usage (L/ha)', 
            'Storage Availability',  # Added this column
            'Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)', 
            'Labor Availability', 'Wholesale Price (₹/kg)', 
            'Domestic Consumption (Tonnes)'
        ]
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in self.original_data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in CSV: {missing_columns}")
        
        self.prepare_data()
    
    def prepare_data(self):
        """Preprocess the data for machine learning"""
        # One-hot encode categorical variables
        categorical_cols = ['Region', 'Season', 'Seed Type', 'Storage Availability']
        self.data = pd.get_dummies(self.original_data, columns=categorical_cols)
        
        # Separate features and targets
        self.target_columns = ['Wholesale Price (₹/kg)', 'Domestic Consumption (Tonnes)']
        
        # Select features 
        feature_columns = [
            col for col in self.data.columns 
            if col not in self.target_columns and 
            col not in categorical_cols
        ] + [col for col in self.data.columns if col.startswith(tuple(categorical_cols))]
        
        self.X = self.data[feature_columns]
        self.y_wholesale_price = self.data['Wholesale Price (₹/kg)']
        self.y_domestic_consumption = self.data['Domestic Consumption (Tonnes)']
    
    def train_models(self, random_state=42):
        """Train machine learning models for price and demand prediction"""
        # Scale the features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Train Random Forest models
        self.price_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        self.demand_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        
        self.price_model.fit(X_scaled, self.y_wholesale_price)
        self.demand_model.fit(X_scaled, self.y_domestic_consumption)
        
        return self
    
    def predict(self, new_data):
        """Make predictions for new data"""
        # Preprocess new data
        new_data_processed = pd.get_dummies(new_data)
        
        # Align columns with training data
        missing_cols = set(self.X.columns) - set(new_data_processed.columns)
        for col in missing_cols:
            new_data_processed[col] = 0
        new_data_processed = new_data_processed[self.X.columns]
        
        # Scale features
        new_data_scaled = self.scaler.transform(new_data_processed)
        
        # Predict
        predicted_prices = self.price_model.predict(new_data_scaled)
        predicted_demand = self.demand_model.predict(new_data_scaled)
        
        return predicted_prices[0], predicted_demand[0]
    
    def get_unique_values(self):
        """
        Get unique values for categorical columns
        
        Returns:
        --------
        dict: A dictionary with unique values for each categorical column
        """
        return {
            'regions': self.original_data['Region'].unique().tolist(),
            'seasons': self.original_data['Season'].unique().tolist(),
            'seed_types': self.original_data['Seed Type'].unique().tolist(),
            'storage_availability': self.original_data['Storage Availability'].unique().tolist()
        }

# Global model instance
# Ensure the CSV file is in the same directory as the script
CSV_PATH = 'Tomato-dataset.csv'

# Check if CSV exists
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found: {os.path.abspath(CSV_PATH)}")

# Initialize and train the model
forecaster = MarketPriceDemandForecaster(CSV_PATH).train_models()

@app.route('/')
def index():
    """Render the main page"""
    # Get unique values
    unique_values = forecaster.get_unique_values()
    
    return render_template('index.html', 
                           regions=unique_values['regions'], 
                           seasons=unique_values['seasons'], 
                           seed_types=unique_values['seed_types'],
                           storage_availability=unique_values['storage_availability'])

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get form data
        input_data = {
            'Region': request.form['region'],
            'Season': request.form['season'],
            'Production Volume (Tonnes)': float(request.form['production_volume']),
            'Cultivated Area (Hectares)': float(request.form['cultivated_area']),
            'Yield (kg/ha)': float(request.form['yield']),
            'Irrigation Coverage (%)': float(request.form['irrigation_coverage']),
            'Seed Type': request.form['seed_type'],
            'Fertilizer Usage (kg/ha)': float(request.form['fertilizer_usage']),
            'Pesticide Usage (L/ha)': float(request.form['pesticide_usage']),
            'Storage Availability': request.form['storage_availability'],  # Added this line
            'Rainfall (mm)': float(request.form['rainfall']),
            'Temperature (°C)': float(request.form['temperature']),
            'Humidity (%)': float(request.form['humidity']),
            'Labor Availability': float(request.form['labor_availability'])
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        predicted_price, predicted_demand = forecaster.predict(input_df)
        
        return jsonify({
            'predicted_wholesale_price': round(predicted_price, 2),
            'predicted_domestic_consumption': round(predicted_demand, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)