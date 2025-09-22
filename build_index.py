import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

# Directory variables.
DATA_PATH = "data/faq.csv"
OUT_DIR = "index_files"

# Prevents an error if the directory already exists
os.makedirs(OUT_DIR, exist_ok=True) 

# Converting csv file to panda DataFrame
df = pd.read_csv(DATA_PATH)

# Takes the "question" column from DataFrame and converts values to strings. Varriable defined for further processing.
questions = df["question"].astype(str).tolist()

# Gonna use a recommended sentence-transformer model. 
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Vector databases indexes and stores vector embeddings for fast retrieval and similarity search. Both are good in our use case
# Encodes string of "questions" into vectors for AI manipulating. Converts to numpy array format cause thats what we're using.
embs = model.encode(questions, convert_to_numpy=True, show_progress_bar=True)

# Normalizing vectors so they have the same length, you make similarity comparisons more meaningful and consistent.
faiss.normalize_L2(embs)

d = embs.shape[1]
index = faiss.IndexFlatIP(d) #FlatIP is a way of indexing that uses "inner product" for its similarity metric. 
index.add(embs) # Adds embs to FAISS index for faster search 


faiss.write_index(index, os.path.join(OUT_DIR, "faq_index.faiss"))
np.save(os.path.join(OUT_DIR, "faq_embeddings.npy"), embs)
df.to_pickle(os.path.join(OUT_DIR, "faq_meta.pkl"))

print("Index is built and saved to: ", OUT_DIR)