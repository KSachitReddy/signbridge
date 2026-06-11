import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def process_gestures(frame):
    """Detects hands and processes landmarks in a frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    gesture_data = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks for the frontend to draw
            landmarks = [
                {"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark
            ]
            gesture_data.append(landmarks)
            
            # Draw landmarks on the frame (optional, for debugging/visualization)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    return {"landmarks": gesture_data}, frame
