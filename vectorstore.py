#vectorsore.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "gemini-docs"

# Step 1: Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # for MiniLM-L6
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Step 2: Connect to the index
index = pc.Index(index_name)

# Step 3: Upsert function
def upsert_chunks(doc_id: str, chunks: list[str], embeddings: list[list[float]]):
    vectors = [
        {
            "id": f"{doc_id}-{i}",
            "values": embeddings[i],
            "metadata": {"text": chunks[i], "doc_id": doc_id}
        }
        for i in range(len(chunks))
    ]
    index.upsert(vectors=vectors)

# Step 4: Query function
def query_top_chunks(query_embed: list[float], top_k=5):
    result = index.query(vector=query_embed, top_k=top_k, include_metadata=True)
    return [match['metadata']['text'] for match in result['matches']]
