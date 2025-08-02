# # llm.py
# import os
# from dotenv import load_dotenv
# import google.generativeai as genai

# # Load Gemini API key
# load_dotenv()
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")

# async def get_answer_rag_async(question: str, context_chunks: list[str], debug: bool = False) -> str:
#     context_text = "\n".join(context_chunks)

#     prompt = (
#         "You are an expert insurance assistant.\n"
#         "Only use the given policy context to answer the question. "
#         "Do not use external knowledge.\n\n"
#         "If the answer is not clearly stated in the context, respond with: "
#         "'Not mentioned in the policy context.'\n\n"
#         "Respond formally and precisely, like an insurance clause.\n\n"
#         f"Context:\n{context_text}\n\n"
#         f"Question:\n{question}\n\n"
#         "Answer in one clear sentence:"
#     )

#     if debug:
#         print("🔍 Prompt to Gemini:\n", prompt)
#         print("📄 Top Context Chunks:\n", context_chunks)

#     try:
#         response = await GEMINI_MODEL.generate_content_async(prompt)
#         return response.text.strip()
#     except Exception as e:
#         return f"⚠️ Gemini error: {str(e)}"
# llm.py
# llm.py (Groq version)
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"  # or "mixtral-8x7b-32768"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

async def get_answer_rag_async(question: str, context_chunks: list[str], debug: bool = False) -> str:
    context_text = "\n".join(context_chunks)
    prompt = (
        "You are an expert insurance assistant.\n"
        "Only use the given policy context to answer the question.\n"
        "If the answer is not clearly stated, reply with: 'Not mentioned in the policy context.'\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question:\n{question}\n\n"
        "Answer in one clear sentence:"
    )

    if debug:
        print("📤 Prompt:\n", prompt)

    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert insurance assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 256
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(GROQ_API_URL, headers=HEADERS, json=body)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            return f"⚠️ Groq error: {e.response.status_code} - {e.response.text}"
        except Exception as e:
            return f"⚠️ Unexpected error: {str(e)}"
