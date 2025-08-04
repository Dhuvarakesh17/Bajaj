import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "gemini-docs"

# Ensure index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

def upsert_chunks(doc_id: str, chunks: list[str], embeddings: list[list[float]], batch_size: int = 50):
    vectors = [
        {
            "id": f"{doc_id}-{i}",
            "values": embeddings[i],
            "metadata": {
                "text": chunks[i],
                "doc_id": doc_id
            }
        }
        for i in range(len(chunks))
    ]

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch)
        except Exception as e:
            print(f"❌ Upsert failed at batch {i // batch_size}: {e}")

def query_top_chunks(query_embed: list[float], top_k=3):
    result = index.query(vector=query_embed, top_k=top_k, include_metadata=True)
    return [match['metadata']['text'] for match in result['matches']]

def is_doc_indexed(doc_id: str) -> bool:
    try:
        response = index.fetch(ids=[f"{doc_id}-0"])
        return bool(response.vectors)
    except Exception as e:
        print(f"⚠️ Error checking Pinecone index: {e}")
        return False
# vectorstore.py
# import os
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec

# load_dotenv()
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index_name = "gemini-docs"

# # Ensure index exists
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=384,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )

# index = pc.Index(index_name)

# def upsert_chunks(doc_id: str, chunks: list[str], embeddings: list[list[float]], batch_size: int = 50):
#     vectors = [
#         {
#             "id": f"{doc_id}-{i}",
#             "values": embeddings[i],
#             "metadata": {
#                 "text": chunks[i],  # ✅ ensure 'text' key is present
#                 "doc_id": doc_id
#             }
#         }
#         for i in range(len(chunks))
#     ]

#     for i in range(0, len(vectors), batch_size):
#         batch = vectors[i:i + batch_size]
#         try:
#             index.upsert(vectors=batch)
#         except Exception as e:
#             print(f"❌ Upsert failed at batch {i // batch_size}: {e}")

# def query_top_chunks(query_embed: list[float], top_k=3) -> list[str]:
#     result = index.query(vector=query_embed, top_k=top_k, include_metadata=True)
#     return [match['metadata']['text'] for match in result['matches'] if 'text' in match['metadata']]

# def is_doc_indexed(doc_id: str) -> bool:
#     try:
#         response = index.fetch(ids=[f"{doc_id}-0"])
#         return bool(response.vectors)
#     except Exception as e:
#         print(f"⚠️ Error checking Pinecone index: {e}")
#         return False
