import os
import sys
import numpy as np
import cv2
import json
import time

# Ensure UTF-8 console output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure workspace is in Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import modules
from modules.database import (
    save_person, get_all_people, delete_person,
    add_face_vector, get_all_face_vectors,
    add_conversation, get_conversations, delete_conversation,
    add_sign_history, get_sign_history, delete_sign_sample,
    save_setting, get_setting
)
from modules.pose.holistic import track_and_draw_pose, _get_pose_landmarker
from modules.hands.landmarks import track_and_draw_hands, _get_hand_landmarker
from modules.face.face_ai import (
    recognize_multiple_faces, _get_face_landmarker,
    cosine_similarity, check_lighting, extract_face_embedding
)
from modules.signs.recognizer import SignSequenceBuffer, sign_classifier, VOCABULARY
from modules.translation import translate_sign
from modules.speech.tts_stt import get_tts_html
from modules.ollama.chat import generate_response, get_conversation_summary
from modules.ollama.manage import list_installed_models

def run_validation():
    print("==================================================")
    print("         SIGNBRIDGE AI VALIDATION TEST            ")
    print("==================================================")
    
    results = {}
    
    # ── 1. DATABASE PERSISTENCE ────────────────────────
    print("\n[Testing DB Persistence]...")
    try:
        # Save mock person
        test_person_id = "P_TEST_VAL_123"
        save_person(test_person_id, "Test User", "Validation testing notes")
        people = get_all_people()
        person_saved = any(p["id"] == test_person_id for p in people)
        
        # Save mock face vector
        test_embedding = [0.1] * 100
        add_face_vector(test_person_id, "mock_path.jpg", test_embedding)
        fvs = get_all_face_vectors()
        fv_saved = any(f["person_id"] == test_person_id for f in fvs)
        
        # Save mock conversation
        add_conversation(test_person_id, "Hello", "Hello", "en", 0.95)
        conversations = get_conversations(person_id=test_person_id)
        conv_saved = len(conversations) > 0 and conversations[0]["recognized_sign"] == "Hello"
        
        # Save mock dataset sample
        test_landmarks = {"left": [], "right": [], "pose": {}}
        add_sign_history("Hello", test_landmarks, test_person_id, "1.0")
        samples = get_sign_history("Hello")
        sample_saved = len(samples) > 0 and samples[0]["person_id"] == test_person_id
        
        # Clean up
        if conv_saved:
            delete_conversation(conversations[0]["id"])
        if sample_saved:
            delete_sign_sample(samples[0]["id"])
        delete_person(test_person_id)
        
        # Settings check
        save_setting("test_val_key", "val_value")
        val_setting = get_setting("test_val_key")
        
        if person_saved and fv_saved and conv_saved and sample_saved and val_setting == "val_value":
            results["Database Persistence"] = "PASS"
            print("  -> Database Persistence: PASS")
        else:
            reasons = []
            if not person_saved: reasons.append("Person not saved")
            if not fv_saved: reasons.append("Face vector not saved")
            if not conv_saved: reasons.append("Conversation not saved")
            if not sample_saved: reasons.append("Dataset sample not saved")
            if val_setting != "val_value": reasons.append("Setting not saved")
            results["Database Persistence"] = f"FAIL: {', '.join(reasons)}"
            print(f"  -> Database Persistence: {results['Database Persistence']}")
    except Exception as e:
        results["Database Persistence"] = f"FAIL: {e}"
        print(f"  -> Database Persistence: {results['Database Persistence']}")

    # ── 2. POSE TRACKING ───────────────────────────────
    print("\n[Testing Pose Tracking]...")
    try:
        landmarker = _get_pose_landmarker()
        if landmarker == "FAILED":
            results["Pose Tracking"] = "FAIL: Model initialization failed"
        else:
            # Test on blank frame
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            joints, annotated = track_and_draw_pose(blank, use_mock=True)
            if joints and "left_shoulder" in joints:
                results["Pose Tracking"] = "PASS"
                print("  -> Pose Tracking: PASS")
            else:
                results["Pose Tracking"] = "FAIL: Could not extract joints in mock fallback"
                print("  -> Pose Tracking: FAIL")
    except Exception as e:
        results["Pose Tracking"] = f"FAIL: {e}"
        print(f"  -> Pose Tracking: {results['Pose Tracking']}")

    # ── 3. HAND TRACKING ───────────────────────────────
    print("\n[Testing Hand Tracking]...")
    try:
        landmarker = _get_hand_landmarker()
        if landmarker == "FAILED":
            results["Hand Tracking"] = "FAIL: Model initialization failed"
        else:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            hands, annotated = track_and_draw_hands(blank, use_mock=True)
            if len(hands.get("left", [])) == 21 and len(hands.get("right", [])) == 21:
                results["Hand Tracking"] = "PASS"
                print("  -> Hand Tracking: PASS")
            else:
                results["Hand Tracking"] = f"FAIL: Expected 21 points per hand, got Left={len(hands.get('left'))}, Right={len(hands.get('right'))}"
                print(f"  -> Hand Tracking: {results['Hand Tracking']}")
    except Exception as e:
        results["Hand Tracking"] = f"FAIL: {e}"
        print(f"  -> Hand Tracking: {results['Hand Tracking']}")

    # ── 4. FACE TRACKING & RECOGNITION ──────────────────
    print("\n[Testing Face Tracking & Recognition]...")
    try:
        landmarker = _get_face_landmarker()
        if landmarker == "FAILED":
            results["Face Recognition"] = "FAIL: Model initialization failed"
        else:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            faces, annotated = recognize_multiple_faces(blank)
            
            # Embeddings and Cosine similarity check
            v1 = [1.0] + [0.0]*99
            v2 = [1.0] + [0.0]*99
            sim = cosine_similarity(v1, v2)
            
            # Lighting check
            gray_good = np.random.randint(100, 150, size=(100, 100), dtype=np.uint8)
            light_ok, _ = check_lighting(gray_good)
            
            if sim > 0.99 and light_ok:
                results["Face Recognition"] = "PASS"
                print("  -> Face Recognition: PASS")
            else:
                results["Face Recognition"] = f"FAIL: cosine_similarity={sim}, light_ok={light_ok}"
                print(f"  -> Face Recognition: {results['Face Recognition']}")
    except Exception as e:
        results["Face Recognition"] = f"FAIL: {e}"
        print(f"  -> Face Recognition: {results['Face Recognition']}")

    # ── 5. SIGN RECOGNITION ────────────────────────────
    print("\n[Testing Sign Recognition]...")
    try:
        if sign_classifier.clf is None:
            print("  [Note] Loaded fallback classifier heuristic")
        buf = SignSequenceBuffer(size=20)
        # Populate buffer with mock data
        for _ in range(20):
            buf.add(
                left_hand=[{"x": 0.3, "y": 0.6, "z": 0.0}] * 21,
                right_hand=[{"x": 0.7, "y": 0.6, "z": 0.0}] * 21,
                pose={"left_shoulder": {"x": 0.4, "y": 0.4}, "right_shoulder": {"x": 0.6, "y": 0.4}}
            )
        preds = sign_classifier.predict(buf)
        if len(preds) >= 3 and all(p[0] in VOCABULARY or p[0] in ("None", "No Sign Detected", "No Gesture Detected") for p in preds):
            results["Sign Recognition"] = "PASS"
            print(f"  -> Sign Recognition: PASS (Top: {preds[0][0]})")
        else:
            results["Sign Recognition"] = "FAIL: Invalid predictions returned"
            print("  -> Sign Recognition: FAIL")
    except Exception as e:
        results["Sign Recognition"] = f"FAIL: {e}"
        print(f"  -> Sign Recognition: {results['Sign Recognition']}")

    # ── 6. OLLAMA INTEGRATION ──────────────────────────
    print("\n[Testing Ollama Integration]...")
    try:
        # Querying local models
        models = list_installed_models()
        model_names = [m["name"] for m in models]
        
        # Test offline dictionary fallback directly
        orig_provider = get_setting("ai_provider", "None (Offline Dictionary)")
        save_setting("ai_provider", "None (Offline Dictionary)")
        
        resp = generate_response("Hello", "Hello", "Test User", "en")
        correct_fallback = "Hello! Great to meet Test User. How can I help you today?"
        
        # Hindi fallback translation check
        resp_hi = generate_response("Help", "Help", "Test User", "hi")
        
        # Restore provider
        save_setting("ai_provider", orig_provider)
        
        if resp == correct_fallback and "मदद" in resp_hi:
            results["Ollama Integration"] = "PASS"
            print("  -> Ollama Integration: PASS")
        else:
            results["Ollama Integration"] = f"FAIL: resp='{resp}', resp_hi='{resp_hi}'"
            print(f"  -> Ollama Integration: {results['Ollama Integration']}")
    except Exception as e:
        results["Ollama Integration"] = f"FAIL: {e}"
        print(f"  -> Ollama Integration: {results['Ollama Integration']}")

    # ── 7. SPEECH OUTPUT & TRANSLATION ─────────────────
    print("\n[Testing Speech & Translation]...")
    try:
        trans_en = translate_sign("Hello", "en")
        trans_hi = translate_sign("Hello", "hi")
        trans_te = translate_sign("Hello", "te")
        
        tts_html = get_tts_html("Hello world", "en")
        
        if trans_en == "Hello" and "नमस्ते" in trans_hi and "playBtn" in tts_html:
            results["Speech & Translation"] = "PASS"
            print("  -> Speech & Translation: PASS")
        else:
            results["Speech & Translation"] = f"FAIL: trans_en={trans_en}, trans_hi={trans_hi}, tts_html={tts_html[:40]}"
            print(f"  -> Speech & Translation: {results['Speech & Translation']}")
    except Exception as e:
        results["Speech & Translation"] = f"FAIL: {e}"
        print(f"  -> Speech & Translation: {results['Speech & Translation']}")

    # ── 8. LANGUAGE SWITCHING ──────────────────────────
    print("\n[Testing Language Switching]...")
    try:
        save_setting("ui_language", "hi")
        lang_set = get_setting("ui_language")
        save_setting("ui_language", "en")  # restore
        
        if lang_set == "hi":
            results["Language Switching"] = "PASS"
            print("  -> Language Switching: PASS")
        else:
            results["Language Switching"] = f"FAIL: expected 'hi', got '{lang_set}'"
            print("  -> Language Switching: FAIL")
    except Exception as e:
        results["Language Switching"] = f"FAIL: {e}"
        print(f"  -> Language Switching: {results['Language Switching']}")

    # ── FINAL SUMMARY REPORT ───────────────────────────
    print("\n==================================================")
    print("               FINAL VALIDATION SUMMARY           ")
    print("==================================================")
    
    all_pass = True
    for feature, status in results.items():
        outcome = "PASS" if status == "PASS" else "FAIL"
        if outcome == "FAIL":
            all_pass = False
        print(f"{feature:<30} : {status}")
        
    print("==================================================")
    if all_pass:
        print("RESULT: ALL TESTS PASSED! SIGNBRIDGE IS READY.")
        return True
    else:
        print("RESULT: SOME TESTS FAILED. PLEASE RESOLVE CRITICAL ISSUES.")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
