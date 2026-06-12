import numpy as np
import os

MODEL_PATH = "../models/gesture_classifier.pkl"
clf = None

try:
    import joblib
    if os.path.exists(MODEL_PATH):
        clf = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load gesture classifier model ({e}). Fallback active.")

def classify_gesture(landmarks):
    """Classifies gesture from landmarks."""
    if clf is not None:
        try:
            # Flatten landmarks
            data = np.array([lm for point in landmarks for lm in [point['x'], point['y'], point['z']]])
            prediction = clf.predict([data])
            return prediction[0]
        except Exception as e:
            print(f"Classifier prediction failed, using rule-based fallback: {e}")
            
    # Rule-based fallback: if we have landmarks, check positions to return a mock label
    if landmarks and len(landmarks) > 8:
        # Check if index finger tip (landmark 8) is higher than knuckle (landmark 5)
        # In MediaPipe landmarks, y-axis goes downwards (0 is top, 1 is bottom)
        tip_y = landmarks[8]['y']
        knuckle_y = landmarks[5]['y']
        
        # If tip is higher (smaller y value), return "Hello"
        if tip_y < knuckle_y:
            return "Hello"
        else:
            return "Thank You"
            
    return "None"
