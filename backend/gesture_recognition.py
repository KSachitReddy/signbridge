import mediapipe as mp
import cv2
import numpy as np

# Pre-initialize MediaPipe Hands to avoid re-loading on every frame
print("Initializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
print("MediaPipe Hands initialized.")

def process_gestures(frame):
    """Detects hands and processes landmarks in a frame."""
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with pre-loaded hands model
    results = hands.process(frame_rgb)
    
    gesture_data = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks for the frontend
            landmarks = [
                {"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark
            ]
            gesture_data.append(landmarks)
            
            # Optional: Draw on frame for backend debugging
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    return {"landmarks": gesture_data}, frame
