"""
LangChain ReAct agent for legal research.
Phase 6: ReAct agent implementation with verbose logging.
"""

from typing import Dict, List
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from agent.tools import (
    search_case_law,
    search_contracts,
    search_all_legal_docs,
    analyze_legal_clause,
    check_jurisdiction,
    set_retriever,
    set_llm
)
from llm.llm_factory import get_llm
from retrieval.retriever import LegalRetriever
from rich.console import Console

console = Console()


class LegalReActAgent:
    """LangChain ReAct agent for legal research."""
    
    def __init__(self, retriever: LegalRetriever):
        """
        Initialize the ReAct agent.
        
        Args:
            retriever: LegalRetriever instance for document retrieval.
        """
        self.retriever = retriever
        
        # Set global instances for tools
        set_retriever(retriever)
        
        # Initialize LLM using factory
        self.llm = get_llm()
        set_llm(self.llm)
        
        # Define tools
        self.tools = [
            search_case_law,
            search_contracts,
            search_all_legal_docs,
            analyze_legal_clause,
            check_jurisdiction
        ]
        
        # Create agent
        self.agent_executor = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """
        Create the ReAct agent executor.
        
        Returns:
            Configured AgentExecutor.
        """
        # Get ReAct prompt from hub or use custom
        try:
            prompt = hub.pull("hwchase17/react")
        except Exception:
            # Fallback to custom prompt if hub pull fails
            from agent.prompts import REACT_AGENT_PROMPT
            from langchain_core.prompts import PromptTemplate
            prompt = PromptTemplate(
                input_variables=["input", "agent_scratchpad", "tool_names"],
                template=REACT_AGENT_PROMPT
            )
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
        
        return agent_executor
    
    def run(self, question: str) -> Dict[str, any]:
        """
        Run the agent on a legal question.
        
        Args:
            question: Legal question to answer.
            
        Returns:
            Dictionary with answer, sources used, and tool calls made.
        """
        console.print(f"\n[blue]Running ReAct Agent on question:[/blue]")
        console.print(f"[cyan]{question}[/cyan]\n")
        
        try:
            result = self.agent_executor.invoke({"input": question})
            
            answer = result.get("output", "")
            
            # Extract tool calls from intermediate steps
            tool_calls = []
            for step in result.get("intermediate_steps", []):
                if len(step) >= 2:
                    action = step[0]
                    tool_calls.append({
                        "tool": action.tool,
                        "input": action.tool_input
                    })
            
            response = {
                "answer": answer,
                "sources": self._extract_sources(answer),
                "tool_calls": tool_calls,
                "success": True
            }
            
            console.print(f"\n[green]Agent completed successfully.[/green]")
            return response
            
        except Exception as e:
            console.print(f"[red]Error running agent: {e}[/red]")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "tool_calls": [],
                "success": False
            }
    
    def _extract_sources(self, text: str) -> List[str]:
        """
        Extract source citations from the answer text.
        
        Args:
            text: Answer text to extract sources from.
            
        Returns:
            List of source citations.
        """
        import re
        pattern = r'\[Source: ([^,]+), Page: (\d+)\]'
        matches = re.findall(pattern, text)
        sources = [f"{source}, Page: {page}" for source, page in matches]
        return sources
