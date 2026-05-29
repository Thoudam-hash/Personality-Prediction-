from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

# 1. Load the AI Model
model = joblib.load('personality_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Connect to MongoDB
# We pull the secret connection string from the environment variables securely
MONGO_URI = os.environ.get('MONGO_URI')

if MONGO_URI:
    client = MongoClient(MONGO_URI)
    db = client['personality_db']          # Creates a database named personality_db
    collection = db['test_results']        # Creates a table/collection named test_results
else:
    print("Warning: MONGO_URI not set. Running without database saving.", flush=True)
    collection = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        features = [
            float(data['social_energy']),
            float(data['routine_preference']),
            float(data['empathy_level']),
            float(data['stress_tolerance']),
            float(data['openness_intellect'])
        ]
        
        # Make the prediction
        scaled_features = scaler.transform([features])
        prediction = str(model.predict(scaled_features)[0])
        
        # 3. Save the results to the database
        if collection is not None:
            user_record = {
                "inputs": {
                    "social_energy": features[0],
                    "routine_preference": features[1],
                    "empathy_level": features[2],
                    "stress_tolerance": features[3],
                    "openness_intellect": features[4]
                },
                "predicted_archetype": prediction,
                "timestamp": datetime.utcnow()
            }
            collection.insert_one(user_record)
            print(f"Successfully saved {prediction} to database!", flush=True)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'stats': features
        })
        
    except Exception as e:
        print(f"Error: {str(e)}", flush=True)
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
