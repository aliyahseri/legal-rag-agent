"""
Tests for document ingestion pipeline.
Phase 12: Test ingestion functionality.
"""

import pytest
from pathlib import Path
from ingestion.loader import LegalDocumentLoader
from ingestion.chunker import LegalTextChunker


def test_synthetic_docs_generated():
    """Test that synthetic documents are generated when no PDFs exist."""
    loader = LegalDocumentLoader()
    
    # Generate synthetic docs
    docs = loader.generate_synthetic_documents("data/case_law")
    
    # Check that docs were created
    assert len(docs) == 5, f"Expected 5 synthetic docs, got {len(docs)}"
    
    # Check that files were created
    data_dir = Path("data/case_law")
    assert data_dir.exists(), "data/case_law directory should exist"
    
    txt_files = list(data_dir.glob("*.txt"))
    assert len(txt_files) == 5, f"Expected 5 .txt files, got {len(txt_files)}"


def test_chunker_preserves_metadata():
    """Test that chunker preserves source metadata."""
    from langchain_core.documents import Document
    
    # Create test documents with metadata
    docs = [
        Document(
            page_content="This is a test document. " * 50,
            metadata={"source": "test.pdf", "doc_type": "case_law", "page": 1}
        )
    ]
    
    # Chunk documents
    chunker = LegalTextChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.chunk_documents(docs)
    
    # Check that metadata is preserved
    for chunk in chunks:
        assert "source" in chunk.metadata, "Source metadata should be preserved"
        assert chunk.metadata["source"] == "test.pdf", "Source should match original"
        assert "doc_type" in chunk.metadata, "Doc type metadata should be preserved"
        assert "page" in chunk.metadata, "Page metadata should be preserved"


def test_chunk_size_bounds():
    """Test that all chunks are within reasonable size bounds."""
    from langchain_core.documents import Document
    
    # Create test documents
    docs = [
        Document(
            page_content="This is a test document. " * 100,
            metadata={"source": "test.pdf", "doc_type": "case_law", "page": 1}
        )
    ]
    
    # Chunk documents
    chunker = LegalTextChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.chunk_documents(docs)
    
    # Check chunk sizes
    for chunk in chunks:
        size = len(chunk.page_content)
        assert 200 <= size <= 1000, f"Chunk size {size} outside bounds [200, 1000]"


def test_chunk_stats():
    """Test chunk statistics calculation."""
    from langchain_core.documents import Document
    
    # Create test documents
    docs = [
        Document(
            page_content="This is a test document. " * 50,
            metadata={"source": "test.pdf", "doc_type": "case_law", "page": 1}
        )
    ]
    
    # Chunk documents
    chunker = LegalTextChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.chunk_documents(docs)
    
    # Get stats
    stats = chunker.get_chunk_stats(chunks)
    
    # Check stats
    assert "count" in stats, "Stats should include count"
    assert "avg_size" in stats, "Stats should include avg_size"
    assert "min_size" in stats, "Stats should include min_size"
    assert "max_size" in stats, "Stats should include max_size"
    assert stats["count"] == len(chunks), "Count should match chunk count"
    assert stats["avg_size"] > 0, "Average size should be positive"
