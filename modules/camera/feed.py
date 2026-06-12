import cv2
import numpy as np

def decode_image_bytes(image_bytes):
    """Converts uploaded image bytes into OpenCV frame format."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def generate_mock_frame(text="Webcam Feed Simulation"):
    """Generates a standard BGR mock frame with shapes for visual landmarks overlay."""
    # Create dark gray background
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = [30, 23, 15] # Dark blue/gray (BGR)
    
    # Draw simple shapes representing a face/body structure
    cv2.circle(frame, (320, 200), 70, (80, 80, 80), -1) # Head
    cv2.ellipse(frame, (320, 380), (120, 100), 0, 0, 360, (120, 120, 120), -1) # Shoulders
    
    # Draw simple hand representations
    cv2.circle(frame, (200, 320), 25, (100, 100, 200), -1) # Left hand
    cv2.circle(frame, (440, 320), 25, (100, 200, 100), -1) # Right hand
    
    # Info label
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "SignBridge AI Camera Mode", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    return frame
