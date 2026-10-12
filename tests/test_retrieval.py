"""
Tests for retrieval pipeline.
Phase 12: Test vector store and retrieval functionality.
"""

import pytest
from pathlib import Path
from langchain_core.documents import Document
from ingestion.chunker import LegalTextChunker
from retrieval.vector_store import LegalVectorStoreManager
from retrieval.retriever import LegalRetriever


def test_vector_store_builds():
    """Test that FAISS vector store builds successfully."""
    # Create test chunks
    docs = [
        Document(
            page_content="Force majeure clause includes pandemics and natural disasters.",
            metadata={"source": "test.pdf", "doc_type": "case_law", "page": 1}
        ),
        Document(
            page_content="Non-compete agreements are generally void in California.",
            metadata={"source": "test.pdf", "doc_type": "contract", "page": 1}
        )
    ]
    
    chunker = LegalTextChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.chunk_documents(docs)
    
    # Build vector store
    vector_store_manager = LegalVectorStoreManager(
        embedding_model="huggingface",  # Use HuggingFace to avoid API key requirement
        openai_api_key=None
    )
    
    case_law_store = vector_store_manager.build_case_law_store(chunks)
    
    # Check that store was created
    assert case_law_store is not None, "Vector store should be built"
    
    # Check that index file was created
    index_path = Path("vector_stores/case_law_index")
    assert index_path.exists(), "Index directory should exist"


def test_retrieval_returns_results():
    """Test that retrieval returns k documents."""
    # Create test chunks
    docs = [
        Document(
            page_content="Force majeure clause includes pandemics and natural disasters.",
            metadata={"source": "test.pdf", "doc_type": "case_law", "page": 1}
        ),
        Document(
            page_content="Non-compete agreements are generally void in California.",
            metadata={"source": "test.pdf", "doc_type": "contract", "page": 1}
        )
    ]
    
    chunker = LegalTextChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.chunk_documents(docs)
    
    # Build vector store
    vector_store_manager = LegalVectorStoreManager(
        embedding_model="huggingface",
        openai_api_key=None
    )
    
    case_law_store = vector_store_manager.build_case_law_store(chunks)
    contracts_store = vector_store_manager.build_contracts_store(chunks)
    
    # Initialize retriever
    retriever = LegalRetriever(
        case_law_store=case_law_store,
        contracts_store=contracts_store,
        k=2
    )
    
    # Test retrieval
    results = retriever.retrieve_case_law("force majeure")
    
    # Check that results were returned
    assert len(results) > 0, "Retrieval should return results"
    assert len(results) <= 2, f"Should return at most k=2 results, got {len(results)}"


def test_parallel_retrieval():
    """Test that parallel retrieval searches both stores."""
    # Create test chunks
    docs = [
        Document(
            page_content="Force majeure clause includes pandemics and natural disasters.",
            metadata={"source": "test.pdf", "doc_type": "case_law", "page": 1}
        ),
        Document(
            page_content="Non-compete agreements are generally void in California.",
            metadata={"source": "test.pdf", "doc_type": "contract", "page": 1}
        )
    ]
    
    chunker = LegalTextChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.chunk_documents(docs)
    
    # Build vector stores
    vector_store_manager = LegalVectorStoreManager(
        embedding_model="huggingface",
        openai_api_key=None
    )
    
    case_law_store = vector_store_manager.build_case_law_store(chunks)
    contracts_store = vector_store_manager.build_contracts_store(chunks)
    
    # Initialize retriever
    retriever = LegalRetriever(
        case_law_store=case_law_store,
        contracts_store=contracts_store,
        k=2
    )
    
    # Test parallel retrieval
    results = retriever.retrieve_parallel("force majeure")
    
    # Check that both stores were searched
    assert "case_law" in results, "Results should include case_law"
    assert "contracts" in results, "Results should include contracts"
    assert isinstance(results["case_law"], list), "case_law results should be a list"
    assert isinstance(results["contracts"], list), "contracts results should be a list"


def test_format_context():
    """Test that context formatting includes citations."""
    # Create test documents
    docs = [
        Document(
            page_content="Force majeure clause includes pandemics.",
            metadata={"source": "test.pdf", "doc_type": "case_law", "page": 1}
        )
    ]
    
    # Initialize retriever (stores can be None for this test)
    retriever = LegalRetriever(case_law_store=None, contracts_store=None, k=5)
    
    # Format context
    context = retriever.format_context(docs)
    
    # Check that citations are included
    assert "[Source:" in context, "Context should include source citation"
    assert "Page:" in context, "Context should include page citation"
