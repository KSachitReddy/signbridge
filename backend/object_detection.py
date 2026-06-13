import os
import urllib.request
import numpy as np
import cv2

# Configuration
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(MODEL_DIR, exist_ok=True)

# MediaPipe Config
MP_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
MP_MODEL_PATH = os.path.join(MODEL_DIR, "efficientdet_lite0.tflite")

# OpenCV DNN Config (MobileNet-SSD)
SSD_PROTO_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
SSD_MODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
SSD_PROTO_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
SSD_MODEL_PATH = os.path.join(MODEL_DIR, "mobilenet_iter_73000.caffemodel")

# COCO Classes for MobileNet-SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def download_file(url, path):
    if not os.path.exists(path):
        try:
            print(f"Downloading {url} to {path}...")
            urllib.request.urlretrieve(url, path)
            print("Download completed.")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

class ObjectDetector:
    def __init__(self):
        self.method = "mock"
        self.net = None
        self.mp_detector = None
        
        # 1. Try to initialize MediaPipe Object Detector
        if os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("SPACE_ID"):
            print("MediaPipe Object Detector skipped on Cloud environment.")
        else:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                download_file(MP_MODEL_URL, MP_MODEL_PATH)
                if os.path.exists(MP_MODEL_PATH):
                    base_options = python.BaseOptions(model_asset_path=MP_MODEL_PATH)
                    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
                    self.mp_detector = vision.ObjectDetector.create_from_options(options)
                    self.method = "mediapipe"
                    print("Initialized MediaPipe Object Detector successfully.")
                    return
            except Exception as e:
                print(f"MediaPipe Object Detector initialization skipped: {e}")
            
        # 2. Fallback to OpenCV DNN (MobileNet-SSD)
        try:
            download_file(SSD_PROTO_URL, SSD_PROTO_PATH)
            download_file(SSD_MODEL_URL, SSD_MODEL_PATH)
            if os.path.exists(SSD_PROTO_PATH) and os.path.exists(SSD_MODEL_PATH):
                self.net = cv2.dnn.readNetFromCaffe(SSD_PROTO_PATH, SSD_MODEL_PATH)
                self.method = "opencv_dnn"
                print("Initialized OpenCV DNN Object Detector (MobileNet-SSD) successfully.")
                return
        except Exception as e:
            print(f"OpenCV DNN Object Detector initialization skipped: {e}")
            
        print("Using Mock Object Detector fallback.")

    def detect(self, frame):
        """Runs object detection on the frame and returns bounding boxes, labels, and scores."""
        h, w = frame.shape[:2]
        
        if self.method == "mediapipe" and self.mp_detector is not None:
            try:
                import mediapipe as mp
                # MediaPipe requires image format conversion
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                results = self.mp_detector.detect(mp_image)
                
                detections = []
                for detection in results.detections:
                    box = detection.bounding_box
                    category = detection.categories[0]
                    detections.append({
                        "label": category.category_name,
                        "confidence": float(category.score),
                        "box": [
                            int(box.origin_x),
                            int(box.origin_y),
                            int(box.width),
                            int(box.height)
                        ]
                    })
                return detections
            except Exception as e:
                print(f"MediaPipe inference failed, falling back: {e}")
                
        if self.method == "opencv_dnn" or self.net is not None:
            try:
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                self.net.setInput(blob)
                detections = self.net.forward()
                
                results = []
                for i in range(detections.shape[2]):
                    confidence = float(detections[0, 0, i, 2])
                    if confidence > 0.5:
                        idx = int(detections[0, 0, i, 1])
                        if 0 <= idx < len(CLASSES):
                            label = CLASSES[idx]
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            results.append({
                                "label": label,
                                "confidence": confidence,
                                "box": [int(startX), int(startY), int(endX - startX), int(endY - startY)]
                            })
                return results
            except Exception as e:
                print(f"OpenCV DNN inference failed: {e}")
                
        # 3. Mock fallback
        # Simulate detecting a person in the center of the frame
        centerX, centerY = w // 2, h // 2
        box_w, box_h = w // 3, h // 2
        return [{
            "label": "person",
            "confidence": 0.92,
            "box": [centerX - box_w // 2, centerY - box_h // 2, box_w, box_h]
        }]

# Single instance
detector = ObjectDetector()
