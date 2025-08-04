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
# import os
# import cohere
# from dotenv import load_dotenv

# load_dotenv()

# COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# COHERE_MODEL = "command-r-plus"  # Or another Cohere chat model

# co = cohere.AsyncClient(COHERE_API_KEY)

# def get_system_role(document_type: str) -> str:
#     roles = {
#         "insurance policy": (
#             "You are an expert insurance assistant. Only use the provided policy context to answer. "
#             "If the answer is not in the context, reply: 'Not mentioned in the policy context.' Respond formally."
#         ),
#         "government constitution": (
#             "You are a constitutional law expert. Only use the provided constitution context to answer. "
#             "If the answer is not in the context, reply: 'Not mentioned in the constitution.' Respond precisely."
#         ),
#         "car manual": (
#             "You are a car technical assistant. Only use the provided manual context to answer. "
#             "If the answer is not in the context, reply: 'Not mentioned in the manual.' Respond clearly."
#         ),
#         "physics textbook": (
#             "You are a physics tutor. Only use the provided textbook context to answer. "
#             "If the answer is not in the context, reply: 'Not mentioned in the textbook.' Respond concisely."
#         ),
#         "document": (
#             "You are a helpful assistant. Only use the provided document context to answer. "
#             "If the answer is not in the context, reply: 'Not mentioned in the provided context.' Respond clearly."
#         )
#     }
#     return roles.get(document_type.lower(), roles["document"])

# async def get_answer_rag_async(
#     question: str,
#     context_chunks: list[str],
#     document_type: str = "document",
#     debug: bool = False
# ) -> str:
#     context_text = "\n".join(context_chunks)
#     system_role = get_system_role(document_type)
#     prompt = (
#         f"{system_role}\n\n"
#         f"Context:\n{context_text}\n\n"
#         f"Question:\n{question}\n\n"
#         "Answer in one clear sentence:"
#     )

#     if debug:
#         print("📤 Prompt to Cohere:\n", prompt)

#     try:
#         response = await co.chat(
#             model=COHERE_MODEL,
#             message=prompt,
#             temperature=0.2,
#             max_tokens=256
#         )
#         return response.text.strip()
#     except Exception as e:
#         return f"⚠️ Cohere error: {str(e)}"
import os
from typing import List
import openai

# Point to your local Ollama instance
openai.api_base = "http://localhost:11434/v1"
openai.api_key = "ollama"  # dummy key
OLLAMA_MODEL = "mistral"  # You can change this to "llama3", "phi3", etc.

def get_system_role(document_type: str) -> str:
    roles = {
        "insurance policy": (
            "You are an expert insurance policy analyst. Use only the given policy document excerpts "
            "to answer questions about coverage, conditions, eligibility, exclusions, and benefits. "
            "If the answer is not present, respond with: 'Not mentioned in the policy context.' "
            "Do not use external knowledge. Maintain a professional and formal tone. "
            "Focus on payout limits, claim rules, age restrictions, pre-existing condition clauses, etc."
        ),
        "government constitution": (
            "You are a constitutional law expert specializing in the Indian Constitution. Use only the "
            "provided constitutional context (such as Articles, Preamble, Schedules, and Amendments) to answer. "
            "Do not infer or assume beyond the given content. If not found, reply: 'Not mentioned in the constitution.' "
            "Be concise, accurate, and formal. Prioritize direct mentions of articles, rights, duties, or powers."
        ),
        "car manual": (
            "You are a certified automobile technical assistant. Use only the car manual content to answer questions "
            "about vehicle components, controls, warnings, maintenance procedures, or safety protocols. "
            "If a topic is not in the manual, say: 'Not mentioned in the manual.' "
            "Avoid assumptions. Stick to technical descriptions and clear explanations suitable for car users."
        ),
        "physics textbook": (
            "You are a physics subject matter expert. Use only the provided textbook context to explain laws, formulas, "
            "concepts, derivations, or experiment descriptions. If the context doesn’t include the information, respond with: "
            "'Not mentioned in the textbook.' Do not bring in external physics knowledge. Respond in a clear and academic tone."
        ),
        "document": (
            "You are a general-purpose assistant. Use only the given document excerpts to answer. "
            "If the answer is not found, respond: 'Not mentioned in the provided context.' "
            "Avoid using any external facts. Respond with clarity, relevance, and accuracy."
        )
    }
    return roles.get(document_type.lower()) or roles["document"]

async def get_answer_rag_async(
    question: str,
    context_chunks: List[str],
    document_type: str = "document",
    debug: bool = False,
    fallback_chunks: List[str] = None,
    keyword_fallback: bool = False,
    full_text: str = None,
    keyword_list: List[str] = None,
) -> str:
    def build_prompt(system_role: str, context: str, question: str) -> List[dict]:
        return [
            {"role": "system", "content": system_role},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer in one clear sentence:"}
        ]

    system_role = get_system_role(document_type)
    context_text = "\n".join(context_chunks)
    messages = build_prompt(system_role, context_text, question)

    if debug:
        print("\n📤 Prompt to Ollama:\n", messages)

    try:
        response = await openai.ChatCompletion.acreate(
            model=OLLAMA_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=256
        )
        answer = response.choices[0].message.content.strip()

        if debug:
            print("📥 Initial Answer:\n", answer)

        # Fallback logic (same as yours)
        if "not mentioned" in answer.lower() and fallback_chunks:
            retry_context = "\n".join(fallback_chunks)
            retry_messages = build_prompt(system_role, retry_context, question)
            retry_response = await openai.ChatCompletion.acreate(
                model=OLLAMA_MODEL,
                messages=retry_messages,
                temperature=0.2,
                max_tokens=256
            )
            retry_answer = retry_response.choices[0].message.content.strip()

            if debug:
                print("🔁 Retried with fallback context:\n", retry_answer)

            if "not mentioned" not in retry_answer.lower():
                return retry_answer

        if "not mentioned" in answer.lower() and keyword_fallback and full_text and keyword_list:
            matches = []
            for kw in keyword_list:
                if kw.lower() in full_text.lower():
                    idx = full_text.lower().index(kw.lower())
                    snippet = full_text[max(0, idx - 100): idx + 300]
                    matches.append(snippet)
            if matches:
                keyword_context = "\n".join(matches[:3])
                keyword_messages = build_prompt(system_role, keyword_context, question)
                keyword_response = await openai.ChatCompletion.acreate(
                    model=OLLAMA_MODEL,
                    messages=keyword_messages,
                    temperature=0.2,
                    max_tokens=256
                )
                keyword_answer = keyword_response.choices[0].message.content.strip()

                if debug:
                    print("🔍 Keyword fallback answer:\n", keyword_answer)

                return keyword_answer

        return answer

    except Exception as e:
        return f"⚠️ Ollama error: {str(e)}"
