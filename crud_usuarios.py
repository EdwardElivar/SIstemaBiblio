import os
import sqlite3
from datetime import datetime

import bcrypt  # Asegúrate de tenerlo en requirements.txt

DB_PATH = os.getenv("BIBLIO_DB_PATH", "biblioteca.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_users_table():
    """Crea la tabla de usuarios si no existe y un admin por defecto."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()

    # Crear admin por defecto si no hay usuarios
    cursor.execute("SELECT COUNT(*) FROM usuarios")
    count = cursor.fetchone()[0]
    if count == 0:
        password = "admin123"
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        cursor.execute(
            "INSERT INTO usuarios (username, password_hash, created_at) VALUES (?, ?, ?)",
            ("admin", password_hash.decode("utf-8"), datetime.utcnow().isoformat()),
        )
        conn.commit()

    conn.close()


def create_user(username: str, password: str):
    """Registra un nuevo usuario con password hasheado."""
    username = (username or "").strip().lower()
    password = (password or "").strip()

    if not username or not password:
        return False, "Usuario y contraseña son obligatorios."

    if len(username) < 3:
        return False, "El usuario debe tener al menos 3 caracteres."

    if len(password) < 6:
        return False, "La contraseña debe tener al menos 6 caracteres."

    conn = get_connection()
    cursor = conn.cursor()

    # Verificar si ya existe
    cursor.execute("SELECT 1 FROM usuarios WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        return False, "El usuario ya existe."

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    cursor.execute(
        "INSERT INTO usuarios (username, password_hash, created_at) VALUES (?, ?, ?)",
        (username, password_hash.decode("utf-8"), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return True, "Usuario registrado correctamente."


def verify_user(username: str, password: str):
    """Valida credenciales contra la BD."""
    username = (username or "").strip().lower()
    password = (password or "").strip()

    if not username or not password:
        return False, "Usuario o contraseña incorrectos."

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT password_hash FROM usuarios WHERE username = ?", (username,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return False, "Usuario o contraseña incorrectos."

    stored_hash = row[0].encode("utf-8")
    if bcrypt.checkpw(password.encode("utf-8"), stored_hash):
        return True, "OK"

    return False, "Usuario o contraseña incorrectos."
