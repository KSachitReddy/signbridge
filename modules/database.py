import json
import os
import sqlite3
from datetime import datetime


DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database"))
DB_PATH = os.path.join(DB_DIR, "signbridge.db")


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_db_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _json_load(value, default):
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS people (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            last_seen TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS face_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            image_path TEXT DEFAULT '',
            embedding TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT DEFAULT 'Unknown',
            timestamp TEXT NOT NULL,
            recognized_sign TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            language TEXT NOT NULL,
            confidence REAL DEFAULT 0.0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sign_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            sign_label TEXT NOT NULL,
            landmarks TEXT NOT NULL,
            person_id TEXT DEFAULT 'Unknown',
            model_version TEXT DEFAULT '1.0'
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT NOT NULL,
            accuracy REAL DEFAULT 0.0,
            sample_count INTEGER DEFAULT 0,
            architecture TEXT DEFAULT '',
            status TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_setting(key, value):
    init_db()
    conn = get_db_connection()
    conn.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        (str(key), str(value)),
    )
    conn.commit()
    conn.close()


def get_setting(key, default=None):
    init_db()
    conn = get_db_connection()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (str(key),)).fetchone()
    conn.close()
    return row["value"] if row else default


def save_person(person_id, name, notes=""):
    init_db()
    conn = get_db_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO people (id, name, notes, created_at, last_seen)
        VALUES (?, ?, ?, COALESCE((SELECT created_at FROM people WHERE id = ?), ?),
                COALESCE((SELECT last_seen FROM people WHERE id = ?), NULL))
        """,
        (person_id, name, notes, person_id, _now(), person_id),
    )
    conn.commit()
    conn.close()
    return person_id


def get_all_people():
    init_db()
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM people ORDER BY name COLLATE NOCASE").fetchall()
    conn.close()
    people = []
    for row in rows:
        item = dict(row)
        item["date_added"] = item.get("created_at") or ""
        item["last_seen"] = item.get("last_seen") or "Never"
        people.append(item)
    return people


def update_person_name(person_id, new_name):
    init_db()
    conn = get_db_connection()
    conn.execute("UPDATE people SET name = ? WHERE id = ?", (new_name, person_id))
    conn.commit()
    conn.close()


def delete_person(person_id):
    init_db()
    conn = get_db_connection()
    conn.execute("DELETE FROM face_vectors WHERE person_id = ?", (person_id,))
    conn.execute("DELETE FROM conversations WHERE person_id = ?", (person_id,))
    conn.execute("DELETE FROM sign_history WHERE person_id = ?", (person_id,))
    conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
    conn.commit()
    conn.close()


def update_last_seen(person_id):
    if not person_id or person_id == "Unknown":
        return
    init_db()
    conn = get_db_connection()
    conn.execute("UPDATE people SET last_seen = ? WHERE id = ?", (_now(), person_id))
    conn.commit()
    conn.close()


def add_face_vector(person_id, image_path, embedding):
    init_db()
    emb = [float(x) for x in embedding] if embedding is not None else []
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO face_vectors (person_id, image_path, embedding, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (person_id, image_path or "", json.dumps(emb), _now()),
    )
    conn.commit()
    conn.close()


def get_all_face_vectors():
    init_db()
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT id, person_id, image_path, embedding, created_at FROM face_vectors"
    ).fetchall()
    conn.close()
    out = []
    for row in rows:
        item = dict(row)
        item["embedding"] = _json_load(item.get("embedding"), [])
        out.append(item)
    return out


def add_conversation(person_id, recognized_sign, translated_text, language, confidence):
    init_db()
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO conversations
            (person_id, timestamp, recognized_sign, translated_text, language, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            person_id or "Unknown",
            _now(),
            recognized_sign or "",
            translated_text or "",
            (language or "en")[:2],
            float(confidence or 0.0),
        ),
    )
    conn.commit()
    conn.close()


def get_conversations(person_id=None, date_filter=None, lang_filter=None):
    init_db()
    query = "SELECT * FROM conversations WHERE 1=1"
    params = []
    if person_id:
        query += " AND person_id = ?"
        params.append(person_id)
    if date_filter:
        query += " AND timestamp LIKE ?"
        params.append(f"{date_filter}%")
    if lang_filter:
        query += " AND language = ?"
        params.append(lang_filter)
    query += " ORDER BY id DESC"
    conn = get_db_connection()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_conversation(conversation_id):
    init_db()
    conn = get_db_connection()
    conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()


def delete_all_conversations():
    init_db()
    conn = get_db_connection()
    conn.execute("DELETE FROM conversations")
    conn.commit()
    conn.close()


def add_sign_history(sign_label, landmarks_data, person_id="Unknown", model_version="1.0"):
    init_db()
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO sign_history
            (timestamp, sign_label, landmarks, person_id, model_version)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            _now(),
            sign_label,
            json.dumps(landmarks_data),
            person_id or "Unknown",
            model_version or "1.0",
        ),
    )
    conn.commit()
    conn.close()


def get_sign_history(label=None):
    init_db()
    conn = get_db_connection()
    if label:
        rows = conn.execute(
            "SELECT * FROM sign_history WHERE sign_label = ? ORDER BY id DESC",
            (label,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM sign_history ORDER BY id DESC").fetchall()
    conn.close()
    out = []
    for row in rows:
        item = dict(row)
        item["landmarks"] = _json_load(item.get("landmarks"), [])
        out.append(item)
    return out


def delete_sign_sample(sample_id):
    init_db()
    conn = get_db_connection()
    conn.execute("DELETE FROM sign_history WHERE id = ?", (sample_id,))
    conn.commit()
    conn.close()


def add_model_version(version, accuracy, sample_count, architecture, status):
    init_db()
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO model_versions
            (version, accuracy, sample_count, architecture, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (version, float(accuracy or 0.0), int(sample_count or 0), architecture, status, _now()),
    )
    conn.commit()
    conn.close()


def export_database_json():
    init_db()
    conn = get_db_connection()
    payload = {}
    for table in [
        "people",
        "face_vectors",
        "conversations",
        "settings",
        "sign_history",
        "model_versions",
    ]:
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        payload[table] = [dict(r) for r in rows]
    conn.close()
    return json.dumps(payload, indent=2, ensure_ascii=False)


def import_database_json(backup_content):
    init_db()
    data = json.loads(backup_content)
    conn = get_db_connection()
    cur = conn.cursor()
    table_columns = {
        "people": ["id", "name", "notes", "created_at", "last_seen"],
        "face_vectors": ["id", "person_id", "image_path", "embedding", "created_at"],
        "conversations": [
            "id", "person_id", "timestamp", "recognized_sign",
            "translated_text", "language", "confidence"
        ],
        "settings": ["key", "value"],
        "sign_history": ["id", "timestamp", "sign_label", "landmarks", "person_id", "model_version"],
        "model_versions": [
            "id", "version", "accuracy", "sample_count", "architecture", "status", "created_at"
        ],
    }
    for table, columns in table_columns.items():
        rows = data.get(table, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            values = [row.get(col) for col in columns]
            placeholders = ",".join(["?"] * len(columns))
            cur.execute(
                f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                values,
            )
    conn.commit()
    conn.close()


init_db()
