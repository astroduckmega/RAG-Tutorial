"""Retrieval-augmented search backed by FAISS and Groq."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Support both `python src/search.py` and `python -m src.search`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

load_dotenv()


class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gemma2-9b-it", data_dir: str = "data"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        if (Path(persist_dir) / "faiss.index").exists() and (Path(persist_dir) / "metadata.pkl").exists():
            self.vectorstore.load()
        else:
            self.vectorstore.build_from_documents(load_all_documents(data_dir))

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is required to summarize search results.")
        self.llm = ChatGroq(api_key=api_key, model=llm_model)

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k)
        context = "\n\n".join(result["metadata"].get("text", "") for result in results)
        if not context.strip():
            return "No relevant documents found."
        prompt = f"""Answer the query using only the supplied context. If it does not contain the answer, say so.

Query: {query}

Context:
{context}
"""
        return self.llm.invoke(prompt).content


if __name__ == "__main__":
    print(RAGSearch().search_and_summarize("What is attention mechanism?", top_k=3))


