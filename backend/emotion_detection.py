from deepface import DeepFace

# Pre-load emotion model (DeepFace loads on demand, but we can call it to warm up)
print("Emotion model warming up...")
# DeepFace.analyze loads models dynamically, this is just a comment to indicate design.
print("Emotion model ready.")

def analyze_emotion(frame):
    """Analyzes emotion in the frame using DeepFace."""
    try:
        # analyze emotion
        results = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
        # DeepFace returns a list of dictionaries if multiple faces are detected
        # Return the dominant emotion for the first face detected
        if results and isinstance(results, list):
            return {
                "status": "success",
                "emotion": results[0]['dominant_emotion'],
                "confidence": results[0]['face_confidence']
            }
        return {"status": "no_face"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
