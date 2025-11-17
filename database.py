
import os
import sqlite3
import hashlib
import json
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DB_PATH = os.path.join(BASE_DIR, "db", "users_chat.db")


os.makedirs(os.path.join(BASE_DIR, "db"), exist_ok=True)


def init_user_db():
    """Initialize SQLite database for users and chat history"""
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            username TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            products TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_id ON chat_history(user_id)
    """)
    
    conn.commit()
    conn.close()
    print("User database initialized")


def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == password_hash


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email"""
    conn = sqlite3.connect(USER_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Get user by ID"""
    conn = sqlite3.connect(USER_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def create_user(email: str, username: str, password: str) -> Optional[int]:
    """Create a new user and return user_id"""
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (email, username, password_hash) VALUES (?, ?, ?)",
            (email, username, password_hash)
        )
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        return None


def save_chat_history(user_id: int, query: str, response: str, products: Optional[List[Dict]] = None):
    """Save chat history for a user"""
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        products_json = json.dumps(products) if products else None
        cursor.execute(
            "INSERT INTO chat_history (user_id, query, response, products) VALUES (?, ?, ?, ?)",
            (user_id, query, response, products_json)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving chat history: {e}")


def get_chat_history(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Get chat history for a user"""
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit)
        )
        rows = cursor.fetchall()
        conn.close()
        history = []
        for row in rows:
            item = dict(row)
            try:
                item['products'] = json.loads(item['products']) if item['products'] else None
            except:
                item['products'] = None
            history.append(item)
        return history
    except Exception as e:
        print(f"Error getting chat history: {e}")
        return []


def user_exists(user_id: int) -> bool:
    """Check if user exists"""
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result is not None
