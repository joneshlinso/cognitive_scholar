# main.py

import uvicorn
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import faiss

print("Initializing the API...")

# Initialize FastAPI app
app = FastAPI(
    title="Cognitive Scholar API",
    description="An API for semantic search of arXiv research papers.",
    version="1.0.0",
)

# --- 1. Load all necessary files at startup ---
print("Loading models and data...")
# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the FAISS index
index = faiss.read_index('faiss_index.idx')

# Load the paper IDs and the original paper data
paper_ids = np.load('paper_ids.npy', allow_pickle=True)
df = pd.read_csv('arxiv_papers.csv')
df.drop_duplicates(subset=['id'], keep='first', inplace=True)
# Create a mapping from paper ID to its data for quick lookups
id_to_paper_data = df.set_index('id').to_dict('index')
print("Models and data loaded successfully.")
# -----------------------------------------------


# --- 2. Define the API endpoint ---
@app.get("/search")
def search_papers(query: str, k: int = 5):
    """
    Performs a semantic search for papers based on a query.

    - **query**: The search string (e.g., "language models for code generation").
    - **k**: The number of top results to return.
    """
    print(f"Received search query: '{query}' with k={k}")

    # 1. Convert the query to an embedding
    query_embedding = model.encode([query]).astype('float32')

    # 2. Search the FAISS index for the k most similar vectors
    # D: distances, I: indices of the results in the original embeddings.npy
    distances, indices = index.search(query_embedding, k)
    
    # 3. Fetch the results from our paper data
    results = []
    for i in range(k):
        result_index = indices[0][i]
        paper_id = paper_ids[result_index]
        paper_info = id_to_paper_data.get(paper_id)
        if paper_info:
            results.append({
                'title': paper_info['title'],
                'summary': paper_info['summary'],
                'url': paper_id, # The ID is the URL
                'published_date': paper_info['published']
            })
            
    return {"query": query, "results": results}
# ------------------------------------------


# --- 3. Add a root endpoint for health check ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Cognitive Scholar API!"}

# --- This block allows running the app directly with 'python main.py' ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)