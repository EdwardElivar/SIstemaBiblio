import sqlite3

DB_NAME = "biblioteca.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS libros (
            isbn TEXT PRIMARY KEY,
            titulo TEXT NOT NULL,
            autor TEXT,
            anio INTEGER,
            editorial TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def insertar_libro(isbn, titulo, autor, anio, editorial):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO libros (isbn, titulo, autor, anio, editorial) VALUES (?, ?, ?, ?, ?)",
            (isbn, titulo, autor, anio, editorial),
        )
        conn.commit()
        return True, "Libro registrado correctamente."
    except sqlite3.IntegrityError:
        return False, "Ya existe un libro con ese ISBN."
    finally:
        conn.close()


def buscar_libro(isbn):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT isbn, titulo, autor, anio, editorial FROM libros WHERE isbn = ?",
        (isbn,),
    )
    libro = cursor.fetchone()
    conn.close()
    return libro


def actualizar_libro(isbn, titulo, autor, anio, editorial):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE libros
        SET titulo = ?, autor = ?, anio = ?, editorial = ?
        WHERE isbn = ?
        """,
        (titulo, autor, anio, editorial, isbn),
    )
    conn.commit()
    cambios = cursor.rowcount
    conn.close()
    if cambios == 0:
        return False, "No se encontró un libro con ese ISBN."
    return True, "Libro actualizado correctamente."


def eliminar_libro(isbn):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM libros WHERE isbn = ?", (isbn,))
    conn.commit()
    cambios = cursor.rowcount
    conn.close()
    if cambios == 0:
        return False, "No se encontró un libro con ese ISBN."
    return True, "Libro eliminado correctamente."


def obtener_todos():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT isbn, titulo, autor, anio, editorial FROM libros ORDER BY titulo ASC"
    )
    data = cursor.fetchall()
    conn.close()
    return data
