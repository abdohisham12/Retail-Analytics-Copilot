"""Lightweight document retriever built on TF-IDF for local markdown docs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class DocChunk:
    """Container for a single document chunk."""

    chunk_id: str
    source: str
    heading: str
    text: str

    def as_dict(self, score: float) -> Dict[str, str]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "heading": self.heading,
            "text": self.text,
            "score": score,
        }


class DocRetriever:
    """TF-IDF based retriever over the markdown corpus in `docs/`."""

    def __init__(self, docs_dir: Path | str = Path("docs")) -> None:
        self.docs_dir = Path(docs_dir)
        self.chunks: List[DocChunk] = self._load_chunks(self.docs_dir)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = (
            self.vectorizer.fit_transform([c.text for c in self.chunks])
            if self.chunks
            else None
        )

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, str]]:
        """Return top-k chunks most similar to the query with scores."""
        if not query.strip() or self.doc_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = (self.doc_matrix @ query_vec.T).toarray().ravel()
        top_indices = np.argsort(-scores)[:top_k]

        results: List[Dict[str, str]] = []
        for idx in top_indices:
            chunk = self.chunks[int(idx)]
            score = float(scores[int(idx)])
            results.append(chunk.as_dict(score=score))
        return results

    def _load_chunks(self, docs_dir: Path) -> List[DocChunk]:
        chunks: List[DocChunk] = []
        for md_file in sorted(docs_dir.glob("*.md")):
            chunks.extend(self._chunk_file(md_file))
        return chunks

    def _chunk_file(self, file_path: Path) -> Sequence[DocChunk]:
        text = file_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        current_heading = ""
        buffer: List[str] = []
        chunks: List[DocChunk] = []
        chunk_idx = 0

        def flush_buffer() -> None:
            nonlocal chunk_idx, buffer
            if not buffer:
                return
            paragraph = " ".join(buffer).strip()
            contextual_text = f"{current_heading}: {paragraph}" if current_heading else paragraph
            chunk_id = f"{file_path.stem}::chunk{chunk_idx}"
            chunks.append(
                DocChunk(
                    chunk_id=chunk_id,
                    source=str(file_path.name),
                    heading=current_heading,
                    text=contextual_text,
                )
            )
            chunk_idx += 1
            buffer = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                flush_buffer()
                continue
            if line.startswith("#"):
                flush_buffer()
                current_heading = line.lstrip("#").strip()
                continue
            buffer.append(line)

        flush_buffer()
        return chunks
