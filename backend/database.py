import sqlite3
import os
import json
from datetime import datetime

DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database"))
DB_PATH = os.path.join(DB_DIR, "signbridge.db")

def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table for auth
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    
    # Create faces table for face recognition metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            embedding TEXT,
            registered_date TEXT NOT NULL
        )
    """)
    
    # Create logs table for telemetry and event tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            person TEXT NOT NULL,
            emotion TEXT NOT NULL,
            gesture TEXT NOT NULL,
            translated_text TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"Database initialized successfully at: {DB_PATH}")

def get_db_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# User Auth Functions
def register_user(email, password_hash):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (email, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user(email):
    conn = get_db_connection()
    cursor = conn.cursor()
    user = cursor.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    if user:
        return dict(user)
    return None

# Face Recognition Metadata Functions
def enroll_face(face_id, name, image_path, embedding=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    registered_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    embedding_json = json.dumps(embedding) if embedding else None
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO faces (id, name, image_path, embedding, registered_date) VALUES (?, ?, ?, ?, ?)",
            (face_id, name, image_path, embedding_json, registered_date)
        )
        conn.commit()
        return {
            "id": face_id,
            "name": name,
            "image_path": image_path,
            "registered_date": registered_date
        }
    finally:
        conn.close()

def get_all_faces():
    conn = get_db_connection()
    cursor = conn.cursor()
    rows = cursor.execute("SELECT * FROM faces").fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Logging Functions
def log_event_db(person, emotion, gesture, translated_text):
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute(
            "INSERT INTO logs (timestamp, person, emotion, gesture, translated_text) VALUES (?, ?, ?, ?, ?)",
            (timestamp, person, emotion, gesture, translated_text)
        )
        conn.commit()
        return True
    finally:
        conn.close()

def get_logs():
    conn = get_db_connection()
    cursor = conn.cursor()
    rows = cursor.execute("SELECT * FROM logs ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Initialize db automatically on load
init_db()
