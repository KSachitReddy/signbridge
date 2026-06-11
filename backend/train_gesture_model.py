import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

DATASET_PATH = "../dataset/"
MODEL_PATH = "../models/gesture_classifier.pkl"

def load_data():
    data = []
    labels = []
    for label in os.listdir(DATASET_PATH):
        label_path = os.path.join(DATASET_PATH, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                with open(os.path.join(label_path, file), "r") as f:
                    sample = json.load(f)
                    # Flatten landmarks
                    landmarks = [lm for point in sample["landmarks"] for lm in [point['x'], point['y'], point['z']]]
                    data.append(landmarks)
                    labels.append(label)
    return np.array(data), np.array(labels)

def train_model():
    X, y = load_data()
    if len(X) == 0:
        print("No data to train.")
        return
    
    clf = RandomForestClassifier()
    clf.fit(X, y)
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print("Model trained and saved.")

if __name__ == "__main__":
    train_model()
