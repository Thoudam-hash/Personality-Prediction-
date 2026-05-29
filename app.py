from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('personality_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract the 5 features expected by the saved model
        features = [
            float(data['social_energy']),
            float(data['routine_preference']),
            float(data['empathy_level']),
            float(data['stress_tolerance']),
            float(data['openness_intellect'])
        ]
        features_df = pd.DataFrame([features], columns=['social_energy', 'routine_preference', 'empathy_level', 'stress_tolerance', 'openness_intellect'])
        scaled_features = scaler.transform(features_df)
        
        # Make the prediction
        prediction = model.predict(scaled_features)[0]
        
        # We don't need the if/else logic anymore if your CSV already has text labels!
        # The model will output exactly what is in the CSV's target column.
        return jsonify({
            'success': True,
            'prediction': str(prediction), 
            'stats': features 
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)