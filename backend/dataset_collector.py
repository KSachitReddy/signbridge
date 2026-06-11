import cv2
import json
import os
import time

DATASET_PATH = "../dataset/"
os.makedirs(DATASET_PATH, exist_ok=True)

def collect_gesture(frame, landmarks, label):
    """Saves gesture landmarks with label to dataset."""
    label_path = os.path.join(DATASET_PATH, label)
    os.makedirs(label_path, exist_ok=True)
    
    timestamp = int(time.time() * 1000)
    filename = f"{label}_{timestamp}.json"
    
    data = {
        "landmarks": landmarks,
        "label": label
    }
    
    with open(os.path.join(label_path, filename), "w") as f:
        json.dump(data, f)
    
    print(f"Saved gesture: {label}")
