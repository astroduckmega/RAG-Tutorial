"""Load supported files into LangChain Document objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader


def _load_json(path: Path) -> List[Document]:
    """Load JSON without requiring jq or a JSONLoader schema."""
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    records = value if isinstance(value, list) else [value]
    return [Document(page_content=json.dumps(record, ensure_ascii=False), metadata={"source": str(path)}) for record in records]


def load_all_documents(data_dir: str | Path) -> List[Any]:
    """Load PDFs, text, CSV, Excel, Word, and JSON files below data_dir."""
    data_path = Path(data_dir).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")
    if not data_path.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_path}")

    documents: List[Any] = []
    loader_types = {".pdf": PyPDFLoader, ".txt": TextLoader, ".csv": CSVLoader, ".docx": Docx2txtLoader}
    for path in sorted(file for file in data_path.rglob("*") if file.is_file()):
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                loaded = _load_json(path)
            elif suffix in {".xlsx", ".xls"}:
                # Import lazily: Excel support has optional dependencies.
                from langchain_community.document_loaders.excel import UnstructuredExcelLoader
                loaded = UnstructuredExcelLoader(str(path)).load()
            elif suffix in loader_types:
                loaded = loader_types[suffix](str(path)).load()
            else:
                continue
            documents.extend(loaded)
            print(f"[INFO] Loaded {len(loaded)} document(s) from {path}")
        except Exception as exc:
            print(f"[WARNING] Skipped unreadable file {path}: {exc}")

    print(f"[INFO] Loaded {len(documents)} document(s) from {data_path}")
    return documents


if __name__ == "__main__":
    print(f"Loaded {len(load_all_documents('data'))} documents.")
