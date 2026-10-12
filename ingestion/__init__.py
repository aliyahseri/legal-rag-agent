"""
Document ingestion module for Legal RAG Assistant.
Phase 3: Document loading, chunking, and embedding.
"""

from .loader import LegalDocumentLoader
from .chunker import LegalTextChunker
from .embedder import EmbeddingFactory

__all__ = ["LegalDocumentLoader", "LegalTextChunker", "EmbeddingFactory"]
