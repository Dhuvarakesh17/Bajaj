# main.py (Optimized with async fix)
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
from parse_pdf import extract_text_from_pdf_url
from chunker import chunk_text
from embedder import get_embedding, embed_chunks
from vectorstore import upsert_chunks, query_top_chunks, is_doc_indexed
from llm import get_answer_rag_async
from db import SessionLocal
from models import QueryLog
import time
import asyncio

app = FastAPI()

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@app.get("/")
async def health():
    return {"status": "ok"}

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def handle_query(request: QueryRequest, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token")

    doc_id = request.documents.split("/")[-1].split(".")[0]
    text = extract_text_from_pdf_url(request.documents)
    chunks = chunk_text(text)

    if not is_doc_indexed(doc_id):
        chunk_embeddings = embed_chunks(chunks)
        upsert_chunks(doc_id, chunks, chunk_embeddings)

    async def process_question(q):
        q_embed = get_embedding(q)
        top_chunks = query_top_chunks(q_embed, top_k=3)
        start = time.perf_counter()
        answer = await get_answer_rag_async(q, top_chunks)
        elapsed = round(time.perf_counter() - start, 2)
        return q, answer, elapsed

    results = await asyncio.gather(*[process_question(q) for q in request.questions])
    db = SessionLocal()
    answers = []

    for q, a, t in results:
        answers.append(a)
        log = QueryLog(
            question=q,
            answer=a,
            document_url=request.documents,
            token_used=authorization,
            response_time=t
        )
        db.add(log)

    db.commit()
    db.close()

    return QueryResponse(answers=answers)
