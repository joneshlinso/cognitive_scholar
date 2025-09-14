# 02_generate_embeddings.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading the dataset...")
# Load the CSV file we created in the previous step
df = pd.read_csv('arxiv_papers.csv')

# Ensure there are no missing values in the summary column
df.dropna(subset=['summary'], inplace=True)
# Also, keep track of the IDs for mapping later
paper_ids = df['id'].tolist()

print(f"Loaded {len(df)} papers.")

print("Loading the sentence-transformer model...")
# Load a pre-trained model. 'all-MiniLM-L6-v2' is a great, lightweight model
# perfect for semantic search.
model = SentenceTransformer('all-MiniLM-L6-v2')
# Note: The first time you run this, it will download the model (~90MB).

print("Generating embeddings for paper summaries... (This may take a while)")
# This is the core step. The model reads each summary and converts it into
# a 384-dimensional vector that represents its meaning.
embeddings = model.encode(
    df['summary'].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"Generated embeddings with shape: {embeddings.shape}")

print("Saving embeddings and paper IDs...")
# Save the embeddings and the corresponding paper IDs to separate files
np.save('paper_embeddings.npy', embeddings)
np.save('paper_ids.npy', np.array(paper_ids))

print("Embeddings and IDs saved successfully.")