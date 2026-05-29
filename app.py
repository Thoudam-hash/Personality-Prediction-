import os

from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('personality_model.pkl')
scaler = joblib.load('scaler.pkl')

MONGO_URI = os.environ.get('MONGO_URI')

if MONGO_URI:
    client = MongoClient(MONGO_URI)
    db = client['personality_db']          # Creates a database named personality_db
    collection = db['test_results']        # Creates a collection named test_results
else:
    print("Warning: MONGO_URI not set. Running without database saving.", flush=True)
    collection = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract the 5 features from the frontend
        features = [
            float(data['social_energy']),
            float(data['routine_preference']),
            float(data['empathy_level']),
            float(data['stress_tolerance']),
            float(data['openness_intellect'])
        ]
        features_df = pd.DataFrame(
            [features],
            columns=['social_energy', 'routine_preference', 'empathy_level', 'stress_tolerance', 'openness_intellect']
        )
        scaled_features = scaler.transform(features_df)

        # Make the prediction
        prediction = model.predict(scaled_features)[0]

        # Save to MongoDB if configured
        if collection is not None:
            record = {
                'social_energy': features[0],
                'routine_preference': features[1],
                'empathy_level': features[2],
                'stress_tolerance': features[3],
                'openness_intellect': features[4],
                'prediction': str(prediction)
            }
            collection.insert_one(record)

        return jsonify({
            'success': True,
            'prediction': str(prediction),
            'stats': features
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)