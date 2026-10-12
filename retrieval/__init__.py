"""
Retrieval module for Legal RAG Assistant.
Phase 4: Vector store management and document retrieval.
"""

from .vector_store import LegalVectorStoreManager
from .retriever import LegalRetriever

__all__ = ["LegalVectorStoreManager", "LegalRetriever"]
