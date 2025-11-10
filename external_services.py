import os
import re
import json
import base64
import requests
from openai import OpenAI # pyright: ignore[reportMissingImports]

# Cliente OpenAI usando variable de entorno OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def limpiar_isbn(isbn: str) -> str:
    """Normaliza ISBN dejando solo dígitos y X; acepta 10 o 13 caracteres."""
    if not isbn:
        return ""
    isbn = isbn.upper()
    isbn = re.sub(r"[^0-9X]", "", isbn)
    if len(isbn) in (10, 13):
        return isbn
    return ""


def buscar_en_google_books(isbn=None, titulo=None, autor=None):
    """
    Busca datos del libro en Google Books usando:
    1) ISBN (si hay)
    2) Si no es suficiente, intenta con título/autor.
    Devuelve dict con isbn, titulo, autor, anio, editorial o None.
    """

    def _parse_volume(vol, isbn_fallback=""):
        info = vol.get("volumeInfo", {})

        # ISBN
        isbn_res = isbn_fallback
        for ident in info.get("industryIdentifiers", []):
            t = ident.get("type")
            v = ident.get("identifier", "")
            if t in ("ISBN_13", "ISBN_10") and v:
                isbn_res = v
                if t == "ISBN_13":
                    break

        # Año
        anio = 0
        pub_date = info.get("publishedDate", "")
        if len(pub_date) >= 4 and pub_date[:4].isdigit():
            anio = int(pub_date[:4])

        autores = ", ".join(info.get("authors", []))

        return {
            "isbn": isbn_res,
            "titulo": info.get("title", ""),
            "autor": autores,
            "anio": anio,
            "editorial": info.get("publisher", "")
        }

    # 1) Intento con ISBN
    if isbn:
        isbn_clean = limpiar_isbn(isbn)
        if isbn_clean:
            url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn_clean}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data.get("items"):
                    gb = _parse_volume(data["items"][0], isbn_clean)
                    if gb["titulo"] or gb["autor"]:
                        return gb

    # 2) Intento con título/autor
    q_parts = []
    if titulo:
        q_parts.append(f"intitle:{titulo}")
    if autor:
        q_parts.append(f"inauthor:{autor}")
    if not q_parts:
        return None

    query = "+".join(q_parts)
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None

    data = r.json()
    if not data.get("items"):
        return None

    gb = _parse_volume(data["items"][0], limpiar_isbn(isbn or ""))
    if gb["titulo"] or gb["autor"]:
        return gb

    return None


def identificar_libro_por_imagen(image_bytes):
    """
    Usa OpenAI (visión) para leer portada (ISBN / título / autor)
    y luego Google Books para completar la información.
    Devuelve (dict, error) donde dict tiene:
    isbn, titulo, autor, anio, editorial
    """
    if client is None:
        return None, "OPENAI_API_KEY no configurada en el entorno."

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        "Analiza la portada del libro y devuelve SOLO un JSON válido con esta estructura:\n"
        "{\n"
        '  \"isbn\": \"texto o \"\",\n'
        '  \"titulo\": \"texto o \"\",\n'
        '  \"autor\": \"texto o \"\",\n'
        '  \"anio\": numero (0 si no sabes),\n'
        '  \"editorial\": \"texto o \"\"\n'
        "}\n"
        "Sin texto adicional."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",  # requiere modelo con visión habilitada
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0,
        )

        content = resp.choices[0].message.content or "{}"
        ia = json.loads(content)

        isbn_ia = limpiar_isbn(ia.get("isbn", ""))
        titulo_ia = (ia.get("titulo") or "").strip()
        autor_ia = (ia.get("autor") or "").strip()
        anio_ia = ia.get("anio") or 0
        editorial_ia = (ia.get("editorial") or "").strip()

        # Intentar completar con Google Books
        gb = buscar_en_google_books(
            isbn=isbn_ia or None,
            titulo=titulo_ia or None,
            autor=autor_ia or None
        )

        if gb:
            combinado = {
                "isbn": gb["isbn"] or isbn_ia,
                "titulo": gb["titulo"] or titulo_ia,
                "autor": gb["autor"] or autor_ia,
                "anio": gb["anio"] or anio_ia or 0,
                "editorial": gb["editorial"] or editorial_ia
            }
        else:
            combinado = {
                "isbn": isbn_ia,
                "titulo": titulo_ia,
                "autor": autor_ia,
                "anio": anio_ia or 0,
                "editorial": editorial_ia
            }

        return combinado, None

    except Exception as e:
        return None, f"Error al consultar la IA: {e}"
