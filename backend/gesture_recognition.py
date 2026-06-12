import cv2
import numpy as np

# Pre-initialize MediaPipe Hands to avoid re-loading on every frame
print("Initializing MediaPipe Hands...")
mp_hands = None
hands = None
mp_drawing = None

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    print("MediaPipe Hands initialized successfully.")
except Exception as e:
    print(f"Warning: Could not initialize MediaPipe Hands ({e}). Fallback to mock landmarks active.")

def process_gestures(frame):
    """Detects hands and processes landmarks in a frame."""
    if hands is None or mp_hands is None:
        # Robust Fallback: Mock hand landmarks
        h, w = frame.shape[:2]
        # Simulating standard 21 landmarks of a hand in the center
        landmarks = []
        base_x, base_y = 0.5, 0.6
        for i in range(21):
            landmarks.append({
                "x": base_x + 0.08 * np.sin(i * 0.5),
                "y": base_y - 0.01 * i + 0.03 * np.cos(i * 0.5),
                "z": -0.01 * i
            })
        
        # Draw green mock landmark circle on the frame
        cv2.circle(frame, (w // 2, h // 2), 15, (0, 255, 0), 2)
        cv2.putText(frame, "HAND DETECTED (MOCK)", (w // 2 - 80, h // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return {"landmarks": [landmarks]}, frame

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
            
            # Draw on frame for backend debugging
            if mp_drawing:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    return {"landmarks": gesture_data}, frame
