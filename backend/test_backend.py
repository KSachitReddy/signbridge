import os
import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Make sure we can import app.py correctly
import sys
sys.path.append(os.path.dirname(__file__))

from app import app, SESSIONS

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_auth_flow():
    email = f"test_{secrets_token(4)}@example.com"
    password = "password123"
    
    # 1. Register
    response = client.post("/api/auth/register", json={"email": email, "password": password})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Register duplicate
    response = client.post("/api/auth/register", json={"email": email, "password": password})
    assert response.status_code == 400
    
    # 2. Login
    response = client.post("/api/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "token" in data
    token = data["token"]
    
    # 3. Logout
    response = client.post("/api/auth/logout", data={"token": token})
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_face_enroll_and_recognize():
    # Create dummy image in memory
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = img_encoded.tobytes()
    
    # 1. Enroll
    response = client.post(
        "/api/face/enroll",
        data={"name": "Alice"},
        files={"file": ("alice.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # 2. Recognize
    response = client.post(
        "/api/face/recognize",
        files={"file": ("recognize.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    )
    assert response.status_code == 200
    assert "status" in response.json()

def test_object_detection():
    # Create dummy image in memory
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = img_encoded.tobytes()
    
    response = client.post(
        "/api/object-detection/detect",
        files={"file": ("object.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "results" in response.json()

def test_translation_and_ocr():
    # Create dummy image in memory
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = img_encoded.tobytes()
    
    # 1. File Upload Translation & OCR
    response = client.post(
        "/api/translation/upload",
        data={"language": "hi"},
        files={"file": ("ocr.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "extracted_text" in response.json()
    assert "translated_text" in response.json()
    
    # 2. Camera frame translation
    response = client.post(
        "/api/translation/camera",
        data={"language": "te"},
        files={"file": ("camera.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "gesture" in response.json()
    assert "translated_text" in response.json()

def test_database_endpoints():
    # 1. Records
    response = client.get("/api/database/records")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert isinstance(response.json()["results"], list)
    
    # 2. Queries
    response = client.get("/api/database/queries?person=Alice")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert isinstance(response.json()["results"], list)

def secrets_token(n):
    import secrets
    return secrets.token_hex(n)
