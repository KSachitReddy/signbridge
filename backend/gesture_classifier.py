import joblib
import numpy as np
import os

MODEL_PATH = "../models/gesture_classifier.pkl"

# Load model if exists
clf = None
if os.path.exists(MODEL_PATH):
    clf = joblib.load(MODEL_PATH)

def classify_gesture(landmarks):
    """Classifies gesture from landmarks."""
    if clf is None:
        return "Model not trained"
    
    # Flatten landmarks
    data = np.array([lm for point in landmarks for lm in [point['x'], point['y'], point['z']]])
    prediction = clf.predict([data])
    return prediction[0]
