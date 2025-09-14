# 03_build_index.py

import numpy as np
import faiss

print("Loading paper embeddings...")
# Load the embeddings we created in the previous step
embeddings = np.load('paper_embeddings.npy')

# FAISS requires the embeddings to be in a specific format (float32)
embeddings = embeddings.astype('float32')

print(f"Embeddings loaded. Shape: {embeddings.shape}")

# Get the dimension of our embeddings (e.g., 384 for MiniLM)
d = embeddings.shape[1]

print(f"Building a FAISS index of dimension {d}...")

# We'll use a simple IndexIVFFlat index.
# It's a good balance between search speed and accuracy.
nlist = 100  # How many cells to partition the vector space into
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

# Training the index requires a representative sample of the data
if not index.is_trained:
    print("Training the index...")
    index.train(embeddings)

print("Adding embeddings to the index...")
# Add all our paper embeddings to the trained index
index.add(embeddings)

print(f"Index built successfully. Total vectors in index: {index.ntotal}")

print("Saving the index to a file...")
# Save the index to disk for later use in our API
faiss.write_index(index, 'faiss_index.idx')

print("Index saved as 'faiss_index.idx'.")