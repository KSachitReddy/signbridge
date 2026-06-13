import base64
import os
import secrets
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import socketio
from pydantic import BaseModel

# Import local modules
from face_recognition import recognize_face, register_face
from gesture_recognition import process_gestures
from gesture_classifier import classify_gesture
from emotion_detection import analyze_emotion
from object_detection import detector
from database import register_user, get_user, enroll_face, get_all_faces, log_event_db, get_logs
import bcrypt

# Setup Socket.IO
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = FastAPI(title="SignBridge AI API", version="1.0.0")
socket_app = socketio.ASGIApp(sio, app)

# CORS Middleware for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
SESSIONS = {}

# Pydantic Schemas
class UserAuthSchema(BaseModel):
    email: str
    password: str

# Helper Functions
def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# REST Endpoints

@app.get("/")
async def root():
    return {"status": "success", "message": "SignBridge AI Backend Running"}

# 1. Authentication
@app.post("/api/auth/register")
async def api_register(user: UserAuthSchema):
    if not user.email or not user.password:
        raise HTTPException(status_code=400, detail="Email and password required")
    hashed = hash_password(user.password)
    success = register_user(user.email, hashed)
    if not success:
        raise HTTPException(status_code=400, detail="User already exists")
    return {"status": "success", "message": "User registered successfully"}

@app.post("/api/auth/login")
async def api_login(user: UserAuthSchema):
    db_user = get_user(user.email)
    if not db_user or not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Create session token
    token = secrets.token_hex(32)
    SESSIONS[token] = user.email
    return {"status": "success", "token": token, "email": user.email}

@app.post("/api/auth/logout")
async def api_logout(token: str = Form(...)):
    if token in SESSIONS:
        del SESSIONS[token]
    return {"status": "success", "message": "Logged out successfully"}

# 2. Face Recognition
@app.post("/api/face/enroll")
async def api_enroll_face(name: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
        
    result = register_face(img, name)
    return {"status": "success", "data": result}

@app.post("/api/face/recognize")
async def api_recognize_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
        
    result = recognize_face(img)
    return result

# 3. Object Detection
@app.post("/api/object-detection/detect")
async def api_detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
        
    detections = detector.detect(img)
    return {"status": "success", "results": detections}

# 4. Translation & OCR
TRANSLATION_MAP = {
    "hi": {
        "hello": "नमस्ते",
        "thank you": "धन्यवाद",
        "emergency": "आपातकालीन चेतावनी",
        "person": "व्यक्ति",
        "bottle": "बोतल"
    },
    "te": {
        "hello": "నమస్కారం",
        "thank you": "ధన్యవాదాలు",
        "emergency": "అత్యవసర హెచ్చరిక",
        "person": "వ్యక్తి",
        "bottle": "సీసా"
    }
}

@app.post("/api/translation/upload")
async def api_upload_translation(file: UploadFile = File(...), language: str = Form("en")):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
        
    # OCR Extraction mockup / fallback
    # Normally we run Tesseract. Here we simulate OCR extracting "Hello"
    extracted_text = "Hello"
    
    # Translation
    lang = language.lower()[:2]
    translated = extracted_text
    if lang in TRANSLATION_MAP and extracted_text.lower() in TRANSLATION_MAP[lang]:
        translated = TRANSLATION_MAP[lang][extracted_text.lower()]
        
    log_event_db(person="Unknown", emotion="Neutral", gesture="None", translated_text=translated)
    
    return {
        "status": "success",
        "extracted_text": extracted_text,
        "translated_text": translated,
        "language": language
    }

@app.post("/api/translation/camera")
async def api_camera_translation(file: UploadFile = File(...), language: str = Form("en")):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
        
    gesture_result, _ = process_gestures(img)
    gesture_label = "None"
    if gesture_result["landmarks"]:
        gesture_label = classify_gesture(gesture_result["landmarks"][0])
        
    emotion_result = analyze_emotion(img)
    emotion = emotion_result.get("emotion", "Neutral")
    
    face_result = recognize_face(img)
    person = face_result.get("results", [{}])[0].get("identity", "Unknown") if face_result.get("status") == "success" else "Unknown"
    
    # Translate gesture
    lang = language.lower()[:2]
    translated = gesture_label
    if lang in TRANSLATION_MAP and gesture_label.lower() in TRANSLATION_MAP[lang]:
        translated = TRANSLATION_MAP[lang][gesture_label.lower()]
        
    log_event_db(person=person, emotion=emotion, gesture=gesture_label, translated_text=translated)
    
    return {
        "status": "success",
        "gesture": gesture_label,
        "translated_text": translated,
        "emotion": emotion,
        "person": person
    }

# 5. Database Analytics
@app.get("/api/database/records")
async def api_get_records():
    logs = get_logs()
    return {"status": "success", "results": logs}

@app.get("/api/database/queries")
async def api_run_queries(person: str = None, gesture: str = None):
    logs = get_logs()
    filtered = logs
    if person:
        filtered = [l for l in filtered if person.lower() in l["person"].lower()]
    if gesture:
        filtered = [l for l in filtered if gesture.lower() in l["gesture"].lower()]
    return {"status": "success", "results": filtered}

def get_translated_text(gesture_label, lang_code):
    if not gesture_label or gesture_label in ("None", "No Sign Detected", "No Gesture Detected"):
        return gesture_label
    try:
        from modules.translation.translator import translate_sign
        res = translate_sign(gesture_label, lang_code)
        if res:
            return res
    except Exception as e:
        print(f"Translation module error: {e}")
        
    g = gesture_label.lower().strip()
    l = lang_code.lower().strip()[:2]
    if lang_code.lower().strip() == "tcy":
        l = "tcy"
        
    local_map = {
        "hi": {
            "hello": "नमस्ते", "thank you": "धन्यवाद", "emergency": "आपातकालीन चेतावनी",
            "person": "व्यक्ति", "bottle": "बोतल", "bye": "अलविदा", "thumbs up": "अंगूठा ऊपर (बहुत बढ़िया)",
            "thumbs down": "अंगूठा नीचे (असहमत)", "point left": "बाईं ओर इशारा", "point right": "दाईं ओर इशारा",
            "point up": "ऊपर की ओर इशारा", "point down": "नीचे की ओर इशारा", "open palm": "खुली हथेली", "closed fist": "बंद मुट्ठी"
        },
        "te": {
            "hello": "నమస్కారం", "thank you": "ధన్యవాదాలు", "emergency": "అత్యవసర హెచ్చరిక",
            "person": "వ్యక్తి", "bottle": "సీసా", "bye": "సెలవు / వీడ్కోలు", "thumbs up": "అభినందనలు (థంబ్స్ అప్)",
            "thumbs down": "అసమ్మతి (థంబ్స్ డౌన్)", "point left": "ఎడమ వైపు చూపించు", "point right": "కుడి వైపు చూపించు",
            "point up": "పైకి చూపించు", "point down": "క్రిందికి చూపించు", "open palm": "తెరచిన చేయి", "closed fist": "మూసివున్న పిడికిలి"
        }
    }
    if l in local_map and g in local_map[l]:
        return local_map[l][g]
    return gesture_label

# Socket.IO Event Handlers

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def frame(sid, data):
    """Processes a video frame received from the client."""
    try:
        lang = "en"
        frame_data = data
        if isinstance(data, dict):
            frame_data = data.get("frame")
            lang = data.get("lang", "en")

        # Decode base64 frame
        nparr = np.frombuffer(base64.b64decode(frame_data.split(',')[1]), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return
            
        # Run recognition
        face_result = recognize_face(frame)
        gesture_result, processed_frame = process_gestures(frame)
        emotion_result = analyze_emotion(frame)
        
        # Perform gesture classification if hands detected
        gesture_label = "None"
        if gesture_result["landmarks"]:
            gesture_label = classify_gesture(gesture_result["landmarks"][0])
            
        person = face_result.get("results", [{}])[0].get("identity", "Unknown") if face_result.get("status") == "success" else "Unknown"
        emotion = emotion_result.get("emotion", "Neutral") if emotion_result.get("status") == "success" else "Neutral"
        
        # Get translation
        translated = get_translated_text(gesture_label, lang)
        
        log_event_db(person=person, emotion=emotion, gesture=gesture_label, translated_text=translated)
        
        # Send result back
        await sio.emit('recognition_result', {
            "face": face_result,
            "gesture": {
                **gesture_result, 
                "label": gesture_label,
                "translated_text": translated
            },
            "emotion": emotion_result
        }, to=sid)
    except Exception as e:
        print(f"Frame processing error: {e}")

if __name__ == "__main__":
    uvicorn.run("app:socket_app", host="127.0.0.1", port=8000, reload=True)
