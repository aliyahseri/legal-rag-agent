"""
Embedding factory for different embedding models.
Phase 3: Embedding model creation and comparison.
"""

from typing import Dict
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rich.console import Console

console = Console()


class EmbeddingFactory:
    """Factory for creating different embedding models."""
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the embedding factory.
        
        Args:
            openai_api_key: OpenAI API key for OpenAI embeddings.
        """
        self.openai_api_key = openai_api_key
    
    def get_openai_embeddings(self) -> OpenAIEmbeddings:
        """
        Get OpenAI embeddings model.
        
        Returns:
            OpenAIEmbeddings instance using text-embedding-3-large.
        """
        console.print("[blue]Initializing OpenAI embeddings (text-embedding-3-large)...[/blue]")
        return OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=self.openai_api_key
        )
    
    def get_huggingface_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Get HuggingFace embeddings model.
        
        Returns:
            HuggingFaceEmbeddings instance using BAAI/bge-base-en-v1.5.
        """
        console.print("[blue]Initializing HuggingFace embeddings (BAAI/bge-base-en-v1.5)...[/blue]")
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def get_default_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Returns HuggingFace embeddings so no OpenAI key is needed.
        
        Returns:
            HuggingFaceEmbeddings instance using BAAI/bge-base-en-v1.5.
        """
        return self.get_huggingface_embeddings()
    
    def compare_embeddings(self, text: str) -> Dict[str, any]:
        """
        Compare embeddings from different models on the same text.
        
        Args:
            text: Text to embed and compare.
            
        Returns:
            Dictionary with similarity scores and dimension info.
        """
        console.print("[blue]Comparing embedding models...[/blue]")
        
        try:
            openai_emb = self.get_openai_embeddings()
            openai_vector = openai_emb.embed_query(text)
            openai_dim = len(openai_vector)
        except Exception as e:
            console.print(f"[yellow]OpenAI embeddings failed: {e}[/yellow]")
            openai_vector = None
            openai_dim = 0
        
        try:
            hf_emb = self.get_huggingface_embeddings()
            hf_vector = hf_emb.embed_query(text)
            hf_dim = len(hf_vector)
        except Exception as e:
            console.print(f"[yellow]HuggingFace embeddings failed: {e}[/yellow]")
            hf_vector = None
            hf_dim = 0
        
        result = {
            "openai_dimension": openai_dim,
            "huggingface_dimension": hf_dim,
            "cosine_similarity": None
        }
        
        if openai_vector is not None and hf_vector is not None:
            # Reshape for cosine similarity calculation
            openai_arr = np.array(openai_vector).reshape(1, -1)
            hf_arr = np.array(hf_vector).reshape(1, -1)
            similarity = cosine_similarity(openai_arr, hf_arr)[0][0]
            result["cosine_similarity"] = float(similarity)
            console.print(f"[green]Cosine similarity: {similarity:.4f}[/green]")
        
        return result
