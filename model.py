import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
print('model.py starting')
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset from the actual data file
dataset_path = 'data-final.csv'
if not os.path.exists(dataset_path):
    print(f"Error: dataset file not found at {dataset_path}")
    exit(1)

# The dataset is tab-separated and the folder name is dataset.csv
try:
    df = pd.read_csv(dataset_path, sep='\t', engine='python')
except Exception as e:
    print('Error reading dataset:', e)
    exit(1)

# 2. Create the FIVE input features from the questionnaire data
EXT_cols = [f'EXT{i}' for i in range(1, 11)]
EST_cols = [f'EST{i}' for i in range(1, 11)]
AGR_cols = [f'AGR{i}' for i in range(1, 11)]
CSN_cols = [f'CSN{i}' for i in range(1, 11)]
OPN_cols = [f'OPN{i}' for i in range(1, 11)] # Added Openness

expected = EXT_cols + EST_cols + AGR_cols + CSN_cols + OPN_cols # Added OPN to expected
missing = [col for col in expected if col not in df.columns]
if missing:
    print('Error: missing columns in dataset:', missing)
    exit(1)

feature_df = df.dropna(subset=expected).copy()
print('total rows after dropna:', len(feature_df))
if len(feature_df) > 10000:
    feature_df = feature_df.sample(n=10000, random_state=42)
    print('sampled rows for training:', len(feature_df))

# Feature Engineering (Multiplying by 2 to match our 1-10 website scale!)
feature_df['social_energy'] = feature_df[EXT_cols].mean(axis=1) * 2
feature_df['routine_preference'] = feature_df[CSN_cols].mean(axis=1) * 2
feature_df['empathy_level'] = feature_df[AGR_cols].mean(axis=1) * 2
feature_df['stress_tolerance'] = (6 - feature_df[EST_cols].mean(axis=1)) * 2
feature_df['openness_intellect'] = feature_df[OPN_cols].mean(axis=1) * 2 # Engineered the 5th trait

# 3. Build a sample personality label from the derived features
conditions = [
    (feature_df['social_energy'] >= 6.0) & (feature_df['empathy_level'] >= 6.0) & (feature_df['routine_preference'] >= 6.0),
    (feature_df['social_energy'] >= 6.0) & (feature_df['openness_intellect'] >= 7.0), # Using Openness here!
    (feature_df['routine_preference'] >= 6.0) & (feature_df['stress_tolerance'] >= 6.0),
    (feature_df['empathy_level'] >= 6.0) & (feature_df['stress_tolerance'] >= 6.0)
]
choices = [
    'Charismatic Leader',
    'Warm Connector',
    'Organized Planner',
    'Calm Supporter'
]
feature_df['personality_type'] = np.select(conditions, choices, default='Thoughtful Explorer')

# Update X to include all 5 features!
X = feature_df[['social_energy', 'routine_preference', 'empathy_level', 'stress_tolerance', 'openness_intellect']]
y = feature_df['personality_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(f"Model trained with 5 derived features and accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, 'personality_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print('Model and scaler saved successfully!')