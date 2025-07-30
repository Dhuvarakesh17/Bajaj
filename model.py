# main.py

from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import os
from dotenv import load_dotenv

# === Load Env Variables ===
load_dotenv()

# === Gemini Setup ===
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")

# === FastAPI App ===
app = FastAPI()

# === Embedding Model Setup ===
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# === Data Models ===
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# === Helper Functions ===
def extract_text_from_pdf_url(url: str) -> str:
    import requests
    response = requests.get(url)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)
    doc = fitz.open("temp.pdf")
    full_text = "\n".join([page.get_text() for page in doc])
    return full_text

def chunk_text(text: str, max_length=500) -> List[str]:
    import textwrap
    return textwrap.wrap(text, max_length)

def get_top_chunks(question: str, chunks: List[str], top_k=5) -> List[str]:
    question_embed = EMBEDDING_MODEL.encode(question, convert_to_tensor=True)
    chunk_embeds = EMBEDDING_MODEL.encode(chunks, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_embed, chunk_embeds)[0]
    top_results = scores.argsort(descending=True)[:top_k]
    return [chunks[i] for i in top_results]

def get_answer_rag(question: str, context_chunks: List[str]) -> str:
    prompt = f"""Use the context below to answer the user's question accurately and precisely.

Context:
{''.join(context_chunks)}

Question: {question}

Answer:"""
    response = GEMINI_MODEL.generate_content(prompt)
    return response.text.strip()

# === Main Endpoint ===
@app.post("/hackrx/run", response_model=QueryResponse)
def handle_query(request: QueryRequest, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token")

    text = extract_text_from_pdf_url(request.documents)
    chunks = chunk_text(text)

    answers = []
    for q in request.questions:
        top_chunks = get_top_chunks(q, chunks)
        answer = get_answer_rag(q, top_chunks)
        answers.append(answer)

    return QueryResponse(answers=answers)

# === Local Development Server ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
