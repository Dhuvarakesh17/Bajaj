#embedder.py
from sentence_transformers import SentenceTransformer

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    return EMBED_MODEL.encode(text).tolist()

def embed_chunks(chunks: list[str]):
    return [get_embedding(chunk) for chunk in chunks]