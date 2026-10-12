"""
Legal text chunker for document splitting.
Phase 3: Text chunking with metadata preservation.
"""

from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

console = Console()


class LegalTextChunker:
    """Chunk legal documents while preserving metadata."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in tokens.
            chunk_overlap: Number of tokens to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Chunk a list of documents while preserving metadata.
        
        Args:
            docs: List of Document objects to chunk.
            
        Returns:
            List of chunked Document objects with preserved metadata.
        """
        console.print(f"[blue]Chunking {len(docs)} documents...[/blue]")
        
        chunks = []
        for doc in docs:
            doc_chunks = self.splitter.split_documents([doc])
            # Preserve original metadata on each chunk
            for chunk in doc_chunks:
                chunk.metadata.update(doc.metadata)
            chunks.extend(doc_chunks)
        
        console.print(f"[green]Created {len(chunks)} chunks.[/green]")
        return chunks
    
    def get_chunk_stats(self, chunks: List[Document]) -> Dict[str, any]:
        """
        Calculate statistics about the chunks.
        
        Args:
            chunks: List of chunked Document objects.
            
        Returns:
            Dictionary with chunk statistics.
        """
        if not chunks:
            return {
                "count": 0,
                "avg_size": 0,
                "min_size": 0,
                "max_size": 0
            }
        
        sizes = [len(chunk.page_content) for chunk in chunks]
        
        stats = {
            "count": len(chunks),
            "avg_size": sum(sizes) / len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes)
        }
        
        return stats
