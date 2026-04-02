from google import genai
import numpy as np
import faiss
import pickle
import os

# load index
index = faiss.read_index("RAG/vector.index")

with open("RAG/documents.pkl", "rb") as f:
    documents = pickle.load(f)

# init client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def get_similar_texts(query, top_k=1):
    # embed query
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=[query]
    )
    
    query_embedding = np.array([response.embeddings[0].values]).astype("float32")
    
    # search
    distances, indices = index.search(query_embedding, top_k)
    
    return [documents[idx] for idx in indices[0]]
    