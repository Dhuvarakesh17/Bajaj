#llm.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")

def get_answer_rag(question: str, context_chunks: list[str]) -> str:
    prompt = f"""
You are a helpful assistant. Use ONLY the context to answer. Don’t make up anything.

Context:
{''.join(context_chunks)}

Question:
{question}

Answer:"""
    response = GEMINI_MODEL.generate_content(prompt)
    return response.text.strip()