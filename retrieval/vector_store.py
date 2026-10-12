"""
Vector store manager for legal documents.
Phase 4: FAISS vector store creation and loading.
"""

from pathlib import Path
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console

console = Console()


class LegalVectorStoreManager:
    """Manage FAISS vector stores for case law and contracts."""
    
    def __init__(self, embedding_model: str = "huggingface", openai_api_key: str = None):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_model: Type of embedding model ("openai" or "huggingface").
            openai_api_key: OpenAI API key if using OpenAI embeddings.
        """
        self.embedding_model = embedding_model
        self.openai_api_key = openai_api_key
        self.embeddings = self._get_embeddings()
    
    def _get_embeddings(self):
        """Get the appropriate embedding model."""
        if self.embedding_model == "openai" and self.openai_api_key:
            return OpenAIEmbeddings(
                model="text-embedding-3-large",
                api_key=self.openai_api_key
            )
        else:
            console.print("[yellow]Using HuggingFace embeddings as fallback[/yellow]")
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en-v1.5",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
    
    def build_case_law_store(self, chunks: List[Document]) -> FAISS:
        """
        Build and save a FAISS index for case law documents.
        
        Args:
            chunks: List of chunked Document objects.
            
        Returns:
            FAISS vector store for case law.
        """
        console.print("[blue]Building case law vector store...[/blue]")
        
        if not chunks:
            console.print("[yellow]No chunks provided for case law store.[/yellow]")
            return None
        
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # Save the vector store
        save_path = Path("./vector_stores/case_law_index")
        save_path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(save_path))
        
        console.print(f"[green]Case law vector store saved to {save_path}[/green]")
        return vector_store
    
    def build_contracts_store(self, chunks: List[Document]) -> FAISS:
        """
        Build and save a FAISS index for contract documents.
        
        Args:
            chunks: List of chunked Document objects.
            
        Returns:
            FAISS vector store for contracts.
        """
        console.print("[blue]Building contracts vector store...[/blue]")
        
        if not chunks:
            console.print("[yellow]No chunks provided for contracts store.[/yellow]")
            return None
        
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # Save the vector store
        save_path = Path("./vector_stores/contracts_index")
        save_path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(save_path))
        
        console.print(f"[green]Contracts vector store saved to {save_path}[/green]")
        return vector_store
    
    def load_case_law_store(self) -> FAISS:
        """
        Load the case law vector store from disk.
        
        Returns:
            FAISS vector store for case law, or None if not found.
        """
        load_path = Path("./vector_stores/case_law_index")
        
        if not load_path.exists():
            console.print(f"[yellow]Case law store not found at {load_path}[/yellow]")
            return None
        
        console.print(f"[blue]Loading case law vector store from {load_path}[/blue]")
        vector_store = FAISS.load_local(str(load_path), self.embeddings, allow_dangerous_deserialization=True)
        console.print("[green]Case law vector store loaded successfully.[/green]")
        return vector_store
    
    def load_contracts_store(self) -> FAISS:
        """
        Load the contracts vector store from disk.
        
        Returns:
            FAISS vector store for contracts, or None if not found.
        """
        load_path = Path("./vector_stores/contracts_index")
        
        if not load_path.exists():
            console.print(f"[yellow]Contracts store not found at {load_path}[/yellow]")
            return None
        
        console.print(f"[blue]Loading contracts vector store from {load_path}[/blue]")
        vector_store = FAISS.load_local(str(load_path), self.embeddings, allow_dangerous_deserialization=True)
        console.print("[green]Contracts vector store loaded successfully.[/green]")
        return vector_store
    
    def build_or_load_stores(self, case_law_chunks: List[Document], 
                            contract_chunks: List[Document]) -> Tuple[FAISS, FAISS]:
        """
        Build vector stores if not cached, otherwise load from disk.
        
        Args:
            case_law_chunks: List of case law document chunks.
            contract_chunks: List of contract document chunks.
            
        Returns:
            Tuple of (case_law_store, contracts_store).
        """
        # Try to load existing stores
        case_law_store = self.load_case_law_store()
        contracts_store = self.load_contracts_store()
        
        # Build if not found
        if case_law_store is None and case_law_chunks:
            case_law_store = self.build_case_law_store(case_law_chunks)
        
        if contracts_store is None and contract_chunks:
            contracts_store = self.build_contracts_store(contract_chunks)
        
        return case_law_store, contracts_store
