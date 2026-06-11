import json
import os
from deepface import DeepFace

# Configuration
FACE_DB_PATH = "../database/faces.json"
FACE_IMAGE_DIR = "../database/faces/"

# Pre-load the model to avoid re-loading on every frame
print("Loading face recognition model...")
MODEL_NAME = 'VGG-Face'
# Pre-warm the model
try:
    DeepFace.build_model(MODEL_NAME)
    print("Face model loaded successfully.")
except Exception as e:
    print(f"Error pre-loading face model: {e}")

# Ensure storage exists
os.makedirs(FACE_IMAGE_DIR, exist_ok=True)
if not os.path.exists(FACE_DB_PATH):
    with open(FACE_DB_PATH, "w") as f:
        json.dump([], f)

def load_db():
    if not os.path.exists(FACE_DB_PATH):
        return []
    with open(FACE_DB_PATH, "r") as f:
        return json.load(f)

def save_db(db):
    with open(FACE_DB_PATH, "w") as f:
        json.dump(db, f, indent=4)

def recognize_face(frame):
    """Detects and recognizes faces in a frame."""
    try:
        # Use pre-loaded model
        results = DeepFace.find(img_path=frame, db_path=FACE_IMAGE_DIR, model_name=MODEL_NAME, enforce_detection=False)
        
        # Simplified result handling
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def register_face(frame, name):
    """Registers a new face image and saves metadata."""
    db = load_db()
    face_id = f"Person_{len(db):03d}"
    filename = f"{face_id}.jpg"
    filepath = os.path.join(FACE_IMAGE_DIR, filename)
    
    # Save image
    import cv2
    cv2.imwrite(filepath, frame)
    
    # Placeholder for actual embedding logic
    embedding = [0.0] * 128 
    
    new_entry = {
        "name": name,
        "face_id": face_id,
        "image_path": filepath,
        "embeddings": embedding,
        "registered_date": "2026-06-11"
    }
    
    db.append(new_entry)
    save_db(db)
    return new_entry
