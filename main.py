#main.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
from parse_pdf import extract_text_from_pdf_url
from chunker import chunk_text
from embedder import get_embedding, embed_chunks
from vectorstore import upsert_chunks, query_top_chunks
from llm import get_answer_rag
from db import SessionLocal
from models import QueryLog

app = FastAPI()

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/hackrx/run", response_model=QueryResponse)
def handle_query(request: QueryRequest, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token")

    text = extract_text_from_pdf_url(request.documents)
    chunks = chunk_text(text)
    chunk_embeddings = embed_chunks(chunks)
    doc_id = request.documents.split("/")[-1].split(".")[0]
    upsert_chunks(doc_id, chunks, chunk_embeddings)

    db = SessionLocal()
    answers = []
    for q in request.questions:
        q_embed = get_embedding(q)
        top_chunks = query_top_chunks(q_embed)
        answer = get_answer_rag(q, top_chunks)
        answers.append(answer)

        # log to DB
        log = QueryLog(
            question=q,
            answer=answer,
            document_url=request.documents,
            token_used=authorization
        )
        db.add(log)
    db.commit()
    db.close()

    return QueryResponse(answers=answers)