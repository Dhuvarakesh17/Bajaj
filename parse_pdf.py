import fitz
import requests
import os
import hashlib

def extract_text_from_pdf_url(url: str) -> str:
    hash_name = hashlib.md5(url.encode()).hexdigest()
    cache_file = f"cache_{hash_name}.txt"

    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    response = requests.get(url)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)

    doc = fitz.open("temp.pdf")
    text = "\n".join([page.get_text() for page in doc])

    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(text)

    return text
