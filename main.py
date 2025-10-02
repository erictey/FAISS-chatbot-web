import faiss
import numpy as np
import os
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel

OUT_DIR = "index_files"
MODEL_NAME = "all-MiniLM-L6-v2"
THRESHOLD = 0.65

app = FastAPI()

# Pydantic uses this code block to validate that any instances of "Query" will have text field with a string value
# This reduces the risks of runtime errors and simplifies downstream processing.


class Query(BaseModel):
    text: str


model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(os.path.join(OUT_DIR, "faq_index.faiss"))
with open(os.path.join(OUT_DIR, "faq_meta.pkl"), "rb") as f:
    df = pd.read_pickle(f)

# Pickling the DataFrame saves the processed metadata alongside your vector index.
# Unpickling restores the DataFrame for use in your application, enabling efficient retrieval and response generation.


@app.post("/chat")
async def chat(q: Query):
    text = q.text.strip()
    if not text:
        return {"Please type your question."}
    emb = model.encode([text], convert_to_numpy=True)
    faiss.normalize_L2(emb)

    k = 3
    # D are inner products (cosine because normalized)
    D, I = index.search(emb, k)
    best_score = float(D[0][0])
    best_idx = int(I[0][0])

    if best_score >= THRESHOLD:
        answer = df.iloc[best_idx]["answer"]
        return {"answer": answer, "score": best_score, "source_id": int(df.iloc[best_idx]["id"])}
    else:
        return {
            "answer": "I couldn't confidently find an answer. Please try rephrasing or contact us at help@example.org",
            "score": best_score,
            "source_id": None
        }
