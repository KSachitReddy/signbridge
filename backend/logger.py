import json
import os
from datetime import datetime

LOG_FILE = "../logs/events.json"

# Ensure log file exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f)

def log_event(person, emotion, gesture, alphabet):
    """Logs an event to the events.json file."""
    event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "person": person,
        "emotion": emotion,
        "gesture": gesture,
        "alphabet": alphabet
    }
    
    try:
        with open(LOG_FILE, "r+") as f:
            data = json.load(f)
            data.append(event)
            f.seek(0)
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Logging error: {e}")
