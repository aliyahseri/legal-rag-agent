"""
LangGraph stateful agent for legal research.
Phase 7: Multi-node agent with state management.
"""

from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from llm.llm_factory import get_llm
from retrieval.retriever import LegalRetriever
from rich.console import Console

console = Console()


class LegalAgentState(TypedDict):
    """State schema for the legal agent."""
    question: str
    sub_queries: List[str]
    retrieved_case_law: List[Document]
    retrieved_contracts: List[Document]
    context_sufficient: bool
    retrieval_attempts: int
    answer: str
    sources: List[str]
    messages: List[BaseMessage]


class LegalLangGraphAgent:
    """LangGraph-based stateful legal research agent."""
    
    def __init__(self, retriever: LegalRetriever):
        """
        Initialize the LangGraph agent.
        
        Args:
            retriever: LegalRetriever instance for document retrieval.
        """
        self.retriever = retriever
        
        # Initialize LLM using factory
        self.llm = get_llm()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph with nodes and edges.
        
        Returns:
            Compiled StateGraph.
        """
        # Create graph
        workflow = StateGraph(LegalAgentState)
        
        # Add nodes
        workflow.add_node("decompose_question", self.decompose_question)
        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("check_context_sufficiency", self.check_context_sufficiency)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("format_response", self.format_response)
        
        # Set entry point
        workflow.set_entry_point("decompose_question")
        
        # Add edges
        workflow.add_edge("decompose_question", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "check_context_sufficiency")
        
        # Conditional edge for context sufficiency
        workflow.add_conditional_edges(
            "check_context_sufficiency",
            self.should_retrieve_more,
            {
                "retrieve": "retrieve_documents",
                "generate": "generate_answer"
            }
        )
        
        workflow.add_edge("generate_answer", "format_response")
        workflow.add_edge("format_response", END)
        
        # Compile graph
        return workflow.compile()
    
    def decompose_question(self, state: LegalAgentState) -> LegalAgentState:
        """
        Decompose complex legal question into sub-queries.
        
        Args:
            state: Current agent state.
            
        Returns:
            Updated state with sub-queries.
        """
        console.print("[blue]Node: decompose_question[/blue]")
        
        question = state["question"]
        
        prompt = f"""Decompose the following legal question into 2-3 specific sub-queries that would help retrieve relevant legal documents.

Question: {question}

Provide the sub-queries as a numbered list, one per line. Each sub-query should be specific and searchable."""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            
            # Parse sub-queries
            sub_queries = []
            for line in content.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering
                    clean_line = line.lstrip('0123456789.-). ')
                    if clean_line:
                        sub_queries.append(clean_line)
            
            if not sub_queries:
                sub_queries = [question]
            
            console.print(f"[green]Generated {len(sub_queries)} sub-queries[/green]")
            
        except Exception as e:
            console.print(f"[yellow]Error decomposing question: {e}[/yellow]")
            sub_queries = [question]
        
        state["sub_queries"] = sub_queries
        state["retrieval_attempts"] = 0
        return state
    
    def retrieve_documents(self, state: LegalAgentState) -> LegalAgentState:
        """
        Retrieve documents for all sub-queries in parallel.
        
        Args:
            state: Current agent state.
            
        Returns:
            Updated state with retrieved documents.
        """
        console.print("[blue]Node: retrieve_documents[/blue]")
        
        sub_queries = state["sub_queries"]
        all_case_law = []
        all_contracts = []
        
        for query in sub_queries:
            console.print(f"  [cyan]Searching: {query}[/cyan]")
            results = self.retriever.retrieve_parallel(query)
            all_case_law.extend(results["case_law"])
            all_contracts.extend(results["contracts"])
        
        # Deduplicate by content
        seen_content = set()
        unique_case_law = []
        for doc in all_case_law:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_case_law.append(doc)
        
        seen_content = set()
        unique_contracts = []
        for doc in all_contracts:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_contracts.append(doc)
        
        state["retrieved_case_law"] = unique_case_law
        state["retrieved_contracts"] = unique_contracts
        state["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1
        
        console.print(f"[green]Retrieved {len(unique_case_law)} case law docs, {len(unique_contracts)} contract docs[/green]")
        
        return state
    
    def check_context_sufficiency(self, state: LegalAgentState) -> LegalAgentState:
        """
        Check if retrieved context is sufficient to answer the question.
        
        Args:
            state: Current agent state.
            
        Returns:
            Updated state with context sufficiency flag.
        """
        console.print("[blue]Node: check_context_sufficiency[/blue]")
        
        question = state["question"]
        case_law = state["retrieved_case_law"]
        contracts = state["retrieved_contracts"]
        attempts = state["retrieval_attempts"]
        
        # Format context
        context_parts = []
        if case_law:
            context_parts.append("CASE LAW:")
            for doc in case_law[:5]:
                context_parts.append(f"- {doc.page_content[:200]}...")
        
        if contracts:
            context_parts.append("CONTRACTS:")
            for doc in contracts[:5]:
                context_parts.append(f"- {doc.page_content[:200]}...")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Determine if the following retrieved context is sufficient to answer the legal question.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

Answer with either "SUFFICIENT" or "INSUFFICIENT". If insufficient, briefly explain what information is missing."""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.upper()
            
            is_sufficient = "SUFFICIENT" in content or attempts >= 3
            
            if is_sufficient:
                console.print("[green]Context is sufficient[/green]")
            else:
                console.print("[yellow]Context is insufficient, will retrieve more[/yellow]")
            
            state["context_sufficient"] = is_sufficient
            
        except Exception as e:
            console.print(f"[yellow]Error checking sufficiency: {e}[/yellow]")
            state["context_sufficient"] = attempts >= 3
        
        return state
    
    def should_retrieve_more(self, state: LegalAgentState) -> str:
        """
        Determine whether to retrieve more documents or generate answer.
        
        Args:
            state: Current agent state.
            
        Returns:
            "retrieve" or "generate" based on context sufficiency.
        """
        if state["context_sufficient"]:
            return "generate"
        elif state["retrieval_attempts"] < 3:
            return "retrieve"
        else:
            return "generate"
    
    def generate_answer(self, state: LegalAgentState) -> LegalAgentState:
        """
        Generate final answer from accumulated context.
        
        Args:
            state: Current agent state.
            
        Returns:
            Updated state with generated answer.
        """
        console.print("[blue]Node: generate_answer[/blue]")
        
        question = state["question"]
        case_law = state["retrieved_case_law"]
        contracts = state["retrieved_contracts"]
        
        # Format context with citations
        context_parts = []
        
        if case_law:
            context_parts.append("CASE LAW DOCUMENTS:")
            for i, doc in enumerate(case_law, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                context_parts.append(f"[{i}] {doc.page_content}\n   [Source: {source}, Page: {page}]")
        
        if contracts:
            context_parts.append("\nCONTRACT DOCUMENTS:")
            for i, doc in enumerate(contracts, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                context_parts.append(f"[{i}] {doc.page_content}\n   [Source: {source}, Page: {page}]")
        
        context = "\n".join(context_parts)
        
        prompt = f"""RETRIEVED LEGAL CONTEXT:
{context}

LEGAL QUESTION: {question}

Provide a comprehensive legal analysis based strictly on the context above.
Structure your answer with:
1. Summary
2. Analysis
3. Relevant Precedents
4. Conclusion

Include citations to specific documents using [Source: filename, Page: X] format.
If the context is insufficient, state what additional information would be needed.

This is research assistance only. Consult a licensed attorney."""
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
            console.print("[green]Answer generated successfully[/green]")
        except Exception as e:
            console.print(f"[red]Error generating answer: {e}[/red]")
            answer = f"Error generating answer: {str(e)}"
        
        state["answer"] = answer
        return state
    
    def format_response(self, state: LegalAgentState) -> LegalAgentState:
        """
        Format the final response with sources and confidence.
        
        Args:
            state: Current agent state.
            
        Returns:
            Updated state with formatted response.
        """
        console.print("[blue]Node: format_response[/blue]")
        
        answer = state["answer"]
        case_law = state["retrieved_case_law"]
        contracts = state["retrieved_contracts"]
        
        # Extract sources
        sources = []
        for doc in case_law + contracts:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            source_str = f"{source}, Page: {page}"
            if source_str not in sources:
                sources.append(source_str)
        
        state["sources"] = sources
        
        # Add to messages
        messages = state.get("messages", [])
        messages.append(HumanMessage(content=state["question"]))
        messages.append(AIMessage(content=answer))
        state["messages"] = messages
        
        console.print(f"[green]Response formatted with {len(sources)} sources[/green]")
        
        return state
    
    def run(self, question: str) -> dict:
        """
        Run the agent on a single question.
        
        Args:
            question: Legal question to answer.
            
        Returns:
            Dictionary with answer, sources, and metadata.
        """
        console.print(f"\n[blue]Running LangGraph Agent on question:[/blue]")
        console.print(f"[cyan]{question}[/cyan]\n")
        
        initial_state = {
            "question": question,
            "sub_queries": [],
            "retrieved_case_law": [],
            "retrieved_contracts": [],
            "context_sufficient": False,
            "retrieval_attempts": 0,
            "answer": "",
            "sources": [],
            "messages": []
        }
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            response = {
                "answer": final_state["answer"],
                "sources": final_state["sources"],
                "retrieval_attempts": final_state["retrieval_attempts"],
                "sub_queries": final_state["sub_queries"],
                "success": True
            }
            
            console.print(f"\n[green]LangGraph Agent completed successfully.[/green]")
            return response
            
        except Exception as e:
            console.print(f"[red]Error running agent: {e}[/red]")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "retrieval_attempts": 0,
                "sub_queries": [],
                "success": False
            }
    
    def run_conversation(self, messages: List[BaseMessage]) -> dict:
        """
        Run the agent in conversation mode with memory.
        
        Args:
            messages: List of conversation messages.
            
        Returns:
            Dictionary with response and conversation history.
        """
        # Get the last user message as the question
        last_message = messages[-1] if messages else None
        if not last_message:
            return {"answer": "No question provided", "sources": [], "success": False}
        
        question = last_message.content
        
        # Run single query
        result = self.run(question)
        
        # Update messages
        messages.append(AIMessage(content=result["answer"]))
        
        result["messages"] = messages
        return result
