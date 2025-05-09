import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "faiss_index/data.pkl"


def load_documents(folder):
    docs = []
    for fname in os.listdir(folder):
        with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs


def split_text(text, chunk_size=300):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def build_and_save_index(docs_folder):
    print("Building FAISS index...")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)

    docs = load_documents(docs_folder)
    chunks = []
    for doc in docs:
        chunks.extend(split_text(doc))

    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save to .pkl file
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump({"index": index, "embeddings": embeddings, "chunks": chunks}, f)

    print("Index saved.")


def load_index():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            "Index file not found. Run build_and_save_index() first"
        )
    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["embeddings"], data["chunks"]


def get_top_k_chunks(query, embedder, index, docs, k=3):
    query_embedding = embedder.encode([query])

    dists, idxs = index.search(np.array(query_embedding).astype("float32"), k)

    top_chunks = [docs[i] for i in idxs[0]]

    return top_chunks


if __name__ == "__main__":
    build_and_save_index("../zzDatasets/LoTR_Texts")
