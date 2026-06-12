import pytest
import numpy as np
import os
import json
from modules.locales import t
from modules.database import (
    save_person,
    get_all_people,
    update_person_name,
    delete_person,
    add_conversation,
    get_conversations,
    delete_all_conversations
)
from modules.face.face_ai import check_lighting
from modules.signs.recognizer import SignSequenceBuffer, sign_classifier, VOCABULARY

def test_i18n_localization():
    """Verify that translate lookups and locales fallbacks work."""
    # Standard English keys
    assert t("nav.home", "en") == "Home"
    assert t("home.title", "en") == "SignBridge AI"
    
    # Hindi translations
    assert t("nav.home", "hi") == "होम"
    assert t("home.title", "hi") == "साइनब्रिज एआई (SignBridge AI)"
    
    # Telugu fallback / translations
    assert t("nav.home", "te") == "హోమ్"
    
    # Fallback to key or English for missing key
    assert t("non_existent.key", "en") == "non_existent.key"
    assert t("non_existent.key", "hi") == "non_existent.key"

def test_database_crud_operations():
    """Verify SQLite CRUD functionalities for profiles and conversations."""

    
    # Clear logs first
    delete_all_conversations()
    
    # Ensure test profile clean
    delete_person("test_mock_id_123")
    
    # Create profile
    save_person("test_mock_id_123", "Amit Kumar", "ISL Student")
    people = get_all_people()
    
    matched = [p for p in people if p["id"] == "test_mock_id_123"]
    assert len(matched) == 1
    assert matched[0]["name"] == "Amit Kumar"
    assert matched[0]["notes"] == "ISL Student"
    
    # Update profile
    update_person_name("test_mock_id_123", "Amit K. Sharma")
    people_updated = get_all_people()
    matched_updated = [p for p in people_updated if p["id"] == "test_mock_id_123"]
    assert matched_updated[0]["name"] == "Amit K. Sharma"
    
    # Add conversation log
    add_conversation("test_mock_id_123", "Hello", "Hello / Namaste", "hi", 0.96)
    logs = get_conversations(person_id="test_mock_id_123")
    assert len(logs) == 1
    assert logs[0]["recognized_sign"] == "Hello"
    assert logs[0]["translated_text"] == "Hello / Namaste"
    assert logs[0]["language"] == "hi"
    
    # Clean up
    delete_person("test_mock_id_123")
    people_post = get_all_people()
    assert not any(p["id"] == "test_mock_id_123" for p in people_post)

def test_face_enrollment_lighting_check():
    """Verify face recognition lighting validation (brightness and contrast bounds)."""
    # 1. Dark crop (low lighting)
    dark_face = np.zeros((64, 64), dtype=np.uint8)
    ok, msg = check_lighting(dark_face)
    assert not ok
    assert "too dark" in msg.lower()
    
    # 2. Overexposed crop (too bright)
    bright_face = np.ones((64, 64), dtype=np.uint8) * 240
    ok, msg = check_lighting(bright_face)
    assert not ok
    assert "too bright" in msg.lower()
    
    # 3. Low contrast crop (completely uniform gray)
    flat_face = np.ones((64, 64), dtype=np.uint8) * 128
    ok, msg = check_lighting(flat_face)
    assert not ok
    assert "low contrast" in msg.lower()
    
    # 4. Good lighting/contrast crop
    good_face = np.linspace(45, 215, 4096, dtype=np.uint8).reshape((64, 64))
    ok, msg = check_lighting(good_face)
    assert ok
    assert "good lighting" in msg.lower()

def test_temporal_vocabulary_classifier():
    """Verify sequence buffer predictions and vocabulary matching."""
    buffer = SignSequenceBuffer(size=10)
    
    # Add stationary mock landmark frames
    for _ in range(10):
        buffer.add(
            left_hand=[{"x": 0.1, "y": 0.5}],
            right_hand=[{"x": 0.9, "y": 0.5}],
            pose={}
        )
        
    predictions = sign_classifier.predict(buffer)
    assert len(predictions) == 3
    for label, conf in predictions:
        assert label in VOCABULARY
        assert 0.0 <= conf <= 1.0
