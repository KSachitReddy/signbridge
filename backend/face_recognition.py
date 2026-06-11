import cv2
import json
import os
import numpy as np
from deepface import DeepFace

# Configuration
FACE_DB_PATH = "../database/faces.json"
FACE_IMAGE_DIR = "../database/faces/"

# Ensure storage exists
os.makedirs(FACE_IMAGE_DIR, exist_ok=True)
if not os.path.exists(FACE_DB_PATH):
    with open(FACE_DB_PATH, "w") as f:
        json.dump([], f)

def load_db():
    with open(FACE_DB_PATH, "r") as f:
        return json.load(f)

def save_db(db):
    with open(FACE_DB_PATH, "w") as f:
        json.dump(db, f, indent=4)

def recognize_face(frame):
    """Detects and recognizes faces in a frame."""
    try:
        # Use DeepFace for detection and recognition
        # Note: 'enforce_detection=False' allows processing without forcing face detection immediately
        results = DeepFace.find(img_path=frame, db_path=FACE_IMAGE_DIR, model_name='VGG-Face', enforce_detection=False)
        
        # Simple placeholder logic for returning results
        # In a full implementation, integrate with DeepFace results
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def register_face(frame, name):
    """Registers a new face image and saves metadata."""
    face_id = f"Person_{len(load_db()):03d}"
    filename = f"{face_id}.jpg"
    filepath = os.path.join(FACE_IMAGE_DIR, filename)
    
    cv2.imwrite(filepath, frame)
    
    # Generate embedding (simplified)
    embedding = [0.0] * 128 # Placeholder for actual embedding logic
    
    new_entry = {
        "name": name,
        "face_id": face_id,
        "image_path": filepath,
        "embeddings": embedding,
        "registered_date": "2026-06-11"
    }
    
    db = load_db()
    db.append(new_entry)
    save_db(db)
    return new_entry
