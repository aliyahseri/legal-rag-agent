"""
LLM factory module for Legal RAG Assistant.
Provides unified LLM instantiation with Groq/OpenAI fallback.
"""

from .llm_factory import get_llm

__all__ = ["get_llm"]
