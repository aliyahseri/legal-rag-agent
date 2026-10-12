"""
LangChain tools for legal research agent.
Phase 6: Tool definitions for ReAct agent.
"""

from langchain_core.tools import tool
from typing import Optional
from retrieval.retriever import LegalRetriever
from langchain_openai import ChatOpenAI
from rich.console import Console

console = Console()

# Global retriever instance (will be set by agent initialization)
_retriever: Optional[LegalRetriever] = None
_llm: Optional[ChatOpenAI] = None


def set_retriever(retriever: LegalRetriever):
    """Set the global retriever instance for tools."""
    global _retriever
    _retriever = retriever


def set_llm(llm: ChatOpenAI):
    """Set the global LLM instance for tools."""
    global _llm
    _llm = llm


@tool
def search_case_law(query: str) -> str:
    """
    Search case law documents for relevant information.
    
    Args:
        query: Search query for case law.
        
    Returns:
        Formatted string with retrieved case law documents and sources.
    """
    if _retriever is None:
        return "Error: Retriever not initialized."
    
    docs = _retriever.retrieve_case_law(query)
    if not docs:
        return "No case law documents found for the query."
    
    formatted = _retriever.format_context(docs)
    return f"CASE LAW RESULTS:\n{formatted}"


@tool
def search_contracts(query: str) -> str:
    """
    Search contract documents for relevant information.
    
    Args:
        query: Search query for contracts.
        
    Returns:
        Formatted string with retrieved contract documents and sources.
    """
    if _retriever is None:
        return "Error: Retriever not initialized."
    
    docs = _retriever.retrieve_contracts(query)
    if not docs:
        return "No contract documents found for the query."
    
    formatted = _retriever.format_context(docs)
    return f"CONTRACT RESULTS:\n{formatted}"


@tool
def search_all_legal_docs(query: str) -> str:
    """
    Search both case law and contracts in parallel.
    
    Args:
        query: Search query for all legal documents.
        
    Returns:
        Formatted string with retrieved documents from both stores.
    """
    if _retriever is None:
        return "Error: Retriever not initialized."
    
    results = _retriever.retrieve_parallel(query)
    
    output_parts = []
    
    if results["case_law"]:
        output_parts.append("CASE LAW RESULTS:")
        output_parts.append(_retriever.format_context(results["case_law"]))
    
    if results["contracts"]:
        output_parts.append("\n\nCONTRACT RESULTS:")
        output_parts.append(_retriever.format_context(results["contracts"]))
    
    if not results["case_law"] and not results["contracts"]:
        return "No documents found for the query."
    
    return "\n".join(output_parts)


@tool
def analyze_legal_clause(clause_text: str) -> str:
    """
    Analyze a specific legal clause for enforceability and issues.
    
    Args:
        clause_text: The legal clause text to analyze.
        
    Returns:
        Structured analysis including enforceability, jurisdiction issues,
        red flags, and suggested improvements.
    """
    if _llm is None:
        return "Error: LLM not initialized."
    
    analysis_prompt = f"""Analyze the following legal clause and provide a structured analysis:

CLAUSE:
{clause_text}

Provide analysis in the following format:
1. ENFORCEABILITY: Assessment of whether the clause is likely enforceable
2. JURISDICTION ISSUES: Any jurisdiction-specific concerns
3. RED FLAGS: Potential problems or risks
4. SUGGESTED IMPROVEMENTS: Recommendations for strengthening the clause

This is research assistance only. Consult a licensed attorney."""
    
    try:
        response = _llm.invoke(analysis_prompt)
        return response.content
    except Exception as e:
        return f"Error analyzing clause: {str(e)}"


@tool
def check_jurisdiction(query: str, jurisdiction: str) -> str:
    """
    Filter retrieved documents for jurisdiction-specific results.
    
    Args:
        query: Search query.
        jurisdiction: Jurisdiction to filter by (e.g., "California", "New York", "EU").
        
    Returns:
        Jurisdiction-filtered search results.
    """
    if _retriever is None:
        return "Error: Retriever not initialized."
    
    # Search all documents
    results = _retriever.retrieve_parallel(query)
    
    # Filter by jurisdiction mentions
    filtered_case_law = []
    for doc in results["case_law"]:
        if jurisdiction.lower() in doc.page_content.lower():
            filtered_case_law.append(doc)
    
    filtered_contracts = []
    for doc in results["contracts"]:
        if jurisdiction.lower() in doc.page_content.lower():
            filtered_contracts.append(doc)
    
    output_parts = [f"JURISDICTION FILTER: {jurisdiction}"]
    
    if filtered_case_law:
        output_parts.append("\nCASE LAW RESULTS:")
        output_parts.append(_retriever.format_context(filtered_case_law))
    
    if filtered_contracts:
        output_parts.append("\nCONTRACT RESULTS:")
        output_parts.append(_retriever.format_context(filtered_contracts))
    
    if not filtered_case_law and not filtered_contracts:
        output_parts.append("\nNo documents found for this jurisdiction.")
    
    return "\n".join(output_parts)
