"""
Agent module for Legal RAG Assistant.
Phase 5-7: Agent tools, prompts, and implementations.
"""

from .tools import (
    search_case_law,
    search_contracts,
    search_all_legal_docs,
    analyze_legal_clause,
    check_jurisdiction
)
from .prompts import (
    LEGAL_SYSTEM_PROMPT,
    RAG_PROMPT,
    REACT_AGENT_PROMPT,
    CHAIN_OF_THOUGHT_PROMPT
)

__all__ = [
    "search_case_law",
    "search_contracts",
    "search_all_legal_docs",
    "analyze_legal_clause",
    "check_jurisdiction",
    "LEGAL_SYSTEM_PROMPT",
    "RAG_PROMPT",
    "REACT_AGENT_PROMPT",
    "CHAIN_OF_THOUGHT_PROMPT"
]
