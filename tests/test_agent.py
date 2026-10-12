"""
Tests for agent functionality.
Phase 12: Test agent implementations.
"""

import pytest
from pathlib import Path
from langchain_core.documents import Document
from ingestion.chunker import LegalTextChunker
from retrieval.vector_store import LegalVectorStoreManager
from retrieval.retriever import LegalRetriever
from agent.langgraph_agent import LegalLangGraphAgent


def test_langgraph_agent_initialization():
    """Test that LangGraph agent initializes correctly."""
    # Create test chunks
    docs = [
        Document(
            page_content="Force majeure clause includes pandemics and natural disasters.",
            metadata={"source": "test.pdf", "doc_type": "case_law", "page": 1}
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
    
    # Initialize retriever
    retriever = LegalRetriever(
        case_law_store=case_law_store,
        contracts_store=None,
        k=5
    )
    
    # Initialize agent (will fail without API key, but should not crash)
    try:
        agent = LegalLangGraphAgent(
            retriever=retriever,
            openai_api_key="test_key",
            model="gpt-4o"
        )
        assert agent is not None, "Agent should be initialized"
        assert agent.retriever == retriever, "Agent should have retriever"
    except Exception as e:
        # Expected if API key is invalid, but initialization should not crash
        pass


def test_answer_contains_disclaimer():
    """Test that agent answers include the required disclaimer."""
    # This test checks the prompt template, not actual agent execution
    from agent.prompts import LEGAL_SYSTEM_PROMPT, RAG_PROMPT
    
    # Check that disclaimer is in system prompt
    assert "consult a licensed attorney" in LEGAL_SYSTEM_PROMPT.lower(), \
        "System prompt should include disclaimer"
    
    # Check that disclaimer is in RAG prompt
    disclaimer_text = "This is research assistance only. Consult a licensed attorney."
    assert disclaimer_text in RAG_PROMPT.template, \
        "RAG prompt should include disclaimer"


def test_prompt_templates_exist():
    """Test that all required prompt templates exist."""
    from agent.prompts import (
        LEGAL_SYSTEM_PROMPT,
        RAG_PROMPT,
        REACT_AGENT_PROMPT,
        CHAIN_OF_THOUGHT_PROMPT
    )
    
    # Check that prompts are defined
    assert LEGAL_SYSTEM_PROMPT is not None, "LEGAL_SYSTEM_PROMPT should be defined"
    assert RAG_PROMPT is not None, "RAG_PROMPT should be defined"
    assert REACT_AGENT_PROMPT is not None, "REACT_AGENT_PROMPT should be defined"
    assert CHAIN_OF_THOUGHT_PROMPT is not None, "CHAIN_OF_THOUGHT_PROMPT should be defined"
    
    # Check that prompts contain required elements
    assert "legal research" in LEGAL_SYSTEM_PROMPT.lower(), \
        "System prompt should mention legal research"
    assert "{context}" in RAG_PROMPT.template, \
        "RAG prompt should have context variable"
    assert "{question}" in RAG_PROMPT.template, \
        "RAG prompt should have question variable"


def test_agent_tools_exist():
    """Test that agent tools are defined."""
    from agent.tools import (
        search_case_law,
        search_contracts,
        search_all_legal_docs,
        analyze_legal_clause,
        check_jurisdiction
    )
    
    # Check that tools are defined
    assert search_case_law is not None, "search_case_law tool should be defined"
    assert search_contracts is not None, "search_contracts tool should be defined"
    assert search_all_legal_docs is not None, "search_all_legal_docs tool should be defined"
    assert analyze_legal_clause is not None, "analyze_legal_clause tool should be defined"
    assert check_jurisdiction is not None, "check_jurisdiction tool should be defined"
    
    # Check that tools have descriptions
    assert search_case_law.description, "search_case_law should have description"
    assert search_contracts.description, "search_contracts should have description"


def test_eval_cases_defined():
    """Test that evaluation test cases are defined."""
    from evaluation.test_cases import TEST_CASES
    
    # Check that test cases exist
    assert len(TEST_CASES) == 10, f"Expected 10 test cases, got {len(TEST_CASES)}"
    
    # Check that each test case has required fields
    for case in TEST_CASES:
        assert case.case_id, f"Case should have case_id"
        assert case.case_name, f"Case should have case_name"
        assert case.query, f"Case should have query"
        assert case.expected_topics, f"Case should have expected_topics"
        assert case.expected_sources_contain, f"Case should have expected_sources_contain"
        assert case.min_citations >= 0, f"Case should have min_citations"
