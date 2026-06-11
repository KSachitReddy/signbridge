from fastapi import FastAPI
import uvicorn
import socketio
import cv2
import numpy as np
import base64
from face_recognition import recognize_face
from gesture_recognition import process_gestures
from logger import log_event

# Setup Socket.IO
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = FastAPI()
socket_app = socketio.ASGIApp(sio, app)

@app.get("/")
async def root():
    return {"message": "SignBridge AI Backend Running"}

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def frame(sid, data):
    """Processes a video frame received from the client."""
    try:
        # Decode base64 frame
        nparr = np.frombuffer(base64.b64decode(data.split(',')[1]), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run recognition
        face_result = recognize_face(frame)
        gesture_result, processed_frame = process_gestures(frame)
        
        # Log event (simplified extraction)
        person = face_result.get("results", [{}])[0].get("identity", "Unknown") if face_result.get("status") == "success" else "Unknown"
        log_event(person=person, emotion="Neutral", gesture="None", alphabet="None")
        
        # Send result back
        await sio.emit('recognition_result', {
            "face": face_result,
            "gesture": gesture_result
        }, to=sid)
    except Exception as e:
        print(f"Frame processing error: {e}")

if __name__ == "__main__":
    uvicorn.run("app:socket_app", host="0.0.0.0", port=8000, reload=True)
