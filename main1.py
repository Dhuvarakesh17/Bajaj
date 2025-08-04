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
    document_type: str = "document"

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

    # async task with scheduled delay to avoid 429
    async def process_question_with_delay(q, delay):
        await asyncio.sleep(delay)
        q_embed = get_embedding(q)
        top_chunks = query_top_chunks(q_embed, top_k=3)
        fallback_chunks = query_top_chunks(q_embed, top_k=10)

        start = time.perf_counter()
        answer = await get_answer_rag_async(
            question=q,
            context_chunks=top_chunks,
            fallback_chunks=fallback_chunks,
            document_type=request.document_type,
            debug=True,
            full_text=text,
            keyword_fallback=True,
            keyword_list=q.lower().split()
        )
        elapsed = round(time.perf_counter() - start, 2)
        return q, answer, elapsed

    # stagger tasks every 6.5s to stay within 10/min limit
    tasks = [
        process_question_with_delay(q, i * 6.5)
        for i, q in enumerate(request.questions)
    ]
    results = await asyncio.gather(*tasks)

    db = SessionLocal()
    answers = []
    for q, a, t in results:
        answers.append(a)
        db.add(QueryLog(
            question=q,
            answer=a,
            document_url=request.documents,
            token_used=authorization,
            response_time=t
        ))
    db.commit()
    db.close()

    return QueryResponse(answers=answers)
