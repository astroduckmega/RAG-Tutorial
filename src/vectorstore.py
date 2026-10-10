"""A small persistent FAISS vector store."""

from __future__ import annotations

import os
import pickle
from typing import Any, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata: List[dict[str, Any]] = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def build_from_documents(self, documents: List[Any]) -> None:
        if not documents:
            raise ValueError("Cannot build a vector store from an empty document list.")
        pipeline = EmbeddingPipeline(self.embedding_model, self.chunk_size, self.chunk_overlap)
        chunks = pipeline.chunk_documents(documents)
        if not chunks:
            raise ValueError("Document splitting produced no chunks.")
        embeddings = pipeline.embed_chunks(chunks)
        metadatas = [{**dict(chunk.metadata), "text": chunk.page_content} for chunk in chunks]
        self.index, self.metadata = None, []
        self.add_embeddings(embeddings, metadatas)
        self.save()

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[dict[str, Any]] | None = None) -> None:
        vectors = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("Embeddings must be a non-empty two-dimensional array.")
        if metadatas is not None and len(metadatas) != len(vectors):
            raise ValueError("Each embedding must have exactly one metadata entry.")
        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])
        elif vectors.shape[1] != self.index.d:
            raise ValueError(f"Embedding dimension {vectors.shape[1]} does not match index dimension {self.index.d}.")
        self.index.add(vectors)
        self.metadata.extend(metadatas or [{} for _ in range(len(vectors))])

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save an empty vector store.")
        faiss.write_index(self.index, os.path.join(self.persist_dir, "faiss.index"))
        with open(os.path.join(self.persist_dir, "metadata.pkl"), "wb") as file:
            pickle.dump(self.metadata, file)

    def load(self) -> None:
        index_path = os.path.join(self.persist_dir, "faiss.index")
        metadata_path = os.path.join(self.persist_dir, "metadata.pkl")
        if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
            raise FileNotFoundError(f"No persisted vector store found in {self.persist_dir!r}.")
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as file:
            self.metadata = pickle.load(file)
        if self.index.ntotal != len(self.metadata):
            raise ValueError("FAISS index and metadata contain different numbers of vectors.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Vector store is empty; build or load it before querying.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        query = np.ascontiguousarray(np.asarray(query_embedding, dtype=np.float32))
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[0] != 1 or query.shape[1] != self.index.d:
            raise ValueError(f"Query embedding must have shape (1, {self.index.d}).")
        distances, indices = self.index.search(query, min(top_k, self.index.ntotal))
        return [{"index": int(i), "distance": float(d), "metadata": self.metadata[i]} for i, d in zip(indices[0], distances[0]) if i != -1]

    def query(self, query_text: str, top_k: int = 5) -> List[dict[str, Any]]:
        if not query_text or not query_text.strip():
            raise ValueError("query_text must not be empty.")
        return self.search(self.model.encode([query_text]).astype(np.float32), top_k)
