"""
Configuration settings for Legal RAG Assistant.
Phase 2: Pydantic BaseSettings class for environment variable management.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    google_api_key: Optional[str] = Field(default=None, description="Google API key")
    deepseek_api_key: Optional[str] = Field(default=None, description="DeepSeek API key")
    huggingface_api_token: Optional[str] = Field(default=None, description="HuggingFace API token")
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    xai_api_key: Optional[str] = Field(default=None, description="xAI API key")
    
    # Vector Store Settings
    vector_store_path: str = Field(default="./vector_stores", description="Path to vector store directory")
    
    # Chunking Settings
    chunk_size: int = Field(default=800, description="Chunk size for text splitting")
    chunk_overlap: int = Field(default=100, description="Chunk overlap for text splitting")
    
    # Model Settings
    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5", description="Embedding model to use")
    primary_llm: str = Field(default="groq", description="Primary LLM model")
    groq_model: str = Field(default="llama-3.3-70b-versatile", description="Groq model to use")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def vector_store_dir(self) -> Path:
        """Get the vector store directory path, creating it if it doesn't exist."""
        path = Path(self.vector_store_path)
        path.mkdir(parents=True, exist_ok=True)
        return path


# Singleton instance
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the singleton Settings instance.
    
    Returns:
        Settings: The application settings instance.
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
