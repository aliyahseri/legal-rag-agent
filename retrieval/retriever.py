"""
Legal document retriever with parallel search.
Phase 4: Document retrieval from vector stores.
"""

from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console

console = Console()


class LegalRetriever:
    """Retrieve legal documents from vector stores."""
    
    def __init__(self, case_law_store: FAISS, contracts_store: FAISS, k: int = 5):
        """
        Initialize the retriever.
        
        Args:
            case_law_store: FAISS vector store for case law.
            contracts_store: FAISS vector store for contracts.
            k: Number of documents to retrieve per query.
        """
        self.case_law_store = case_law_store
        self.contracts_store = contracts_store
        self.k = k
    
    def retrieve_case_law(self, query: str) -> List[Document]:
        """
        Retrieve case law documents for a query.
        
        Args:
            query: Search query.
            
        Returns:
            List of retrieved Document objects.
        """
        if self.case_law_store is None:
            console.print("[yellow]Case law store not available.[/yellow]")
            return []
        
        docs = self.case_law_store.similarity_search(query, k=self.k)
        console.print(f"[blue]Retrieved {len(docs)} case law documents.[/blue]")
        return docs
    
    def retrieve_contracts(self, query: str) -> List[Document]:
        """
        Retrieve contract documents for a query.
        
        Args:
            query: Search query.
            
        Returns:
            List of retrieved Document objects.
        """
        if self.contracts_store is None:
            console.print("[yellow]Contracts store not available.[/yellow]")
            return []
        
        docs = self.contracts_store.similarity_search(query, k=self.k)
        console.print(f"[blue]Retrieved {len(docs)} contract documents.[/blue]")
        return docs
    
    def retrieve_parallel(self, query: str) -> Dict[str, List[Document]]:
        """
        Search both vector stores in parallel.
        
        Args:
            query: Search query.
            
        Returns:
            Dictionary with "case_law" and "contracts" keys containing retrieved documents.
        """
        console.print("[blue]Searching both stores in parallel...[/blue]")
        
        results = {"case_law": [], "contracts": []}
        
        def search_case_law():
            return self.retrieve_case_law(query)
        
        def search_contracts():
            return self.retrieve_contracts(query)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_store = {
                executor.submit(search_case_law): "case_law",
                executor.submit(search_contracts): "contracts"
            }
            
            for future in as_completed(future_to_store):
                store_name = future_to_store[future]
                try:
                    results[store_name] = future.result()
                except Exception as e:
                    console.print(f"[red]Error searching {store_name}: {e}[/red]")
                    results[store_name] = []
        
        return results
    
    def format_context(self, docs: List[Document]) -> str:
        """
        Format retrieved documents into a numbered, cited context string.
        
        Args:
            docs: List of retrieved Document objects.
            
        Returns:
            Formatted context string with citations.
        """
        if not docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            doc_type = doc.metadata.get("doc_type", "Unknown")
            
            context_part = f"[{i}] {doc.page_content}\n   [Source: {source}, Page: {page}, Type: {doc_type}]"
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
