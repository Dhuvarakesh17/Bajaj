import fitz  # PyMuPDF
import requests

def extract_text_from_pdf_url(url: str) -> str:
    response = requests.get(url)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)
    doc = fitz.open("temp.pdf")
    return "\n".join([page.get_text() for page in doc])