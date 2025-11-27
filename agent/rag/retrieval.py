"""RAG retrieval module for document search"""
import os
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Configuration constants (merged from config.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RAG = 3

# Type definitions (merged from analytics_types.py)
class Citation(BaseModel):
    """Citation for a source used in answering"""
    source_type: Literal["document", "database", "sql_query"]
    source: str = Field(description="Source identifier (file path, table name, or SQL query)")
    content: Optional[str] = Field(None, description="Relevant content excerpt")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


class RAGRetrieval:
    """RAG retrieval for retrieving relevant documents"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self._init_vector_store()
        self._load_documents()
    
    def _init_vector_store(self):
        """Initialize ChromaDB vector store"""
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(VECTOR_STORE_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="retail_docs",
            metadata={"hnsw:space": "cosine"}
        )
    
    def _load_documents(self):
        """Load and index documents from docs/ directory"""
        if not DOCS_DIR.exists():
            DOCS_DIR.mkdir(parents=True, exist_ok=True)
            return
        
        # Check if collection is empty
        if self.collection.count() > 0:
            return  # Already indexed
        
        # Load documents (only markdown and text files)
        loaders = []
        for ext in ["*.txt", "*.md"]:
            loader = DirectoryLoader(
                str(DOCS_DIR),
                glob=ext,
                loader_cls=TextLoader,
                show_progress=True
            )
            loaders.append(loader)
        
        documents = []
        for loader in loaders:
            try:
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Warning: Could not load documents with loader: {e}")
        
        if not documents:
            return
        
        # Split documents
        chunks = self.text_splitter.split_documents(documents)
        
        # Generate embeddings and store
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {
                "source": chunk.metadata.get("source", "unknown"),
                "chunk_index": i
            }
            for i, chunk in enumerate(chunks)
        ]
        ids = [f"doc_{i}" for i in range(len(chunks))]
        
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Indexed {len(chunks)} document chunks")
    
    def search(self, query: str, top_k: int = TOP_K_RAG) -> tuple:
        """Search for relevant documents with chunk IDs and scores
        
        Returns:
            tuple: (citations, chunk_ids, scores) where:
                - citations: List[Citation] with chunk IDs included
                - chunk_ids: List[str] of chunk IDs
                - scores: List[float] of similarity scores
        """
        if self.collection.count() == 0:
            return [], [], []
        
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances", "ids"]
        )
        
        citations = []
        chunk_ids = []
        scores = []
        
        if results["documents"] and len(results["documents"][0]) > 0:
            for doc, metadata, distance, chunk_id in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0]
            ):
                confidence = 1.0 - distance  # Convert distance to confidence
                score = float(confidence)
                
                # Include chunk ID in source
                source_with_id = f"{metadata.get('source', 'unknown')}#{chunk_id}"
                
                citations.append(Citation(
                    source_type="document",
                    source=source_with_id,
                    content=doc[:500],  # First 500 chars (constraint: compact prompts ≤1k tokens)
                    confidence=score
                ))
                chunk_ids.append(chunk_id)
                scores.append(score)
        
        return citations, chunk_ids, scores
    
    def get_context(self, query: str) -> str:
        """Get formatted context from documents"""
        citations, chunk_ids, scores = self.search(query)
        if not citations:
            return ""
        
        context_parts = []
        for i, (citation, chunk_id, score) in enumerate(zip(citations, chunk_ids, scores), 1):
            source_name = Path(citation.source.split('#')[0]).name  # Remove chunk ID from source
            context_parts.append(f"[Doc {i} from {source_name} (chunk: {chunk_id}, score: {score:.3f})]\n{citation.content}")
        
        return "\n\n".join(context_parts)

