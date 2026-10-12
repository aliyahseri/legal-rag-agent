"""
Main CLI entrypoint for Legal RAG Assistant.
Phase 11: Command-line interface for ingestion, querying, evaluation, and serving.
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from config.settings import get_settings
from ingestion.loader import LegalDocumentLoader
from ingestion.chunker import LegalTextChunker
from ingestion.embedder import EmbeddingFactory
from retrieval.vector_store import LegalVectorStoreManager
from retrieval.retriever import LegalRetriever
from agent.langgraph_agent import LegalLangGraphAgent
from evaluation.evaluator import LegalAgentEvaluator
from evaluation.test_cases import TEST_CASES

console = Console()


def ingest():
    """Run the document ingestion pipeline."""
    console.print(Panel.fit("[bold blue]Legal RAG Assistant - Document Ingestion[/bold blue]"))
    
    settings = get_settings()
    
    # Load documents
    console.print("\n[blue]Loading documents...[/blue]")
    loader = LegalDocumentLoader()
    
    case_law_docs = loader.load_case_law("data/case_law")
    contract_docs = loader.load_contracts("data/contracts")
    
    # If no PDFs, check for synthetic docs
    if not case_law_docs and not contract_docs:
        console.print("[yellow]No PDFs found. Checking for synthetic documents...[/yellow]")
        synthetic_case_law = loader.generate_synthetic_documents("data/case_law")
        synthetic_contracts = loader.generate_synthetic_documents("data/contracts")
        case_law_docs = synthetic_case_law
        contract_docs = synthetic_contracts
    
    doc_count = len(case_law_docs) + len(contract_docs)
    console.print(f"[green]Loaded {doc_count} documents total[/green]")
    
    if doc_count == 0:
        console.print("[red]No documents found. Please run: python scripts/generate_synthetic_docs.py[/red]")
        return
    
    # Chunk documents
    console.print("\n[blue]Chunking documents...[/blue]")
    chunker = LegalTextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    case_law_chunks = chunker.chunk_documents(case_law_docs)
    contract_chunks = chunker.chunk_documents(contract_docs)
    
    chunk_count = len(case_law_chunks) + len(contract_chunks)
    console.print(f"[green]Created {chunk_count} chunks[/green]")
    
    # Print chunk stats
    all_chunks = case_law_chunks + contract_chunks
    stats = chunker.get_chunk_stats(all_chunks)
    console.print(f"\n[blue]Chunk Statistics:[/blue]")
    console.print(f"  Count: {stats['count']}")
    console.print(f"  Average size: {stats['avg_size']:.2f} chars")
    console.print(f"  Min size: {stats['min_size']} chars")
    console.print(f"  Max size: {stats['max_size']} chars")
    
    # Build vector stores
    console.print("\n[blue]Building vector stores...[/blue]")
    vector_store_manager = LegalVectorStoreManager(
        embedding_model="huggingface",
        openai_api_key=settings.openai_api_key
    )
    
    case_law_store, contracts_store = vector_store_manager.build_or_load_stores(
        case_law_chunks,
        contract_chunks
    )
    
    # Print store sizes
    if case_law_store:
        console.print(f"[green]Case law store built with {len(case_law_chunks)} chunks[/green]")
    if contracts_store:
        console.print(f"[green]Contracts store built with {len(contract_chunks)} chunks[/green]")
    
    console.print("\n[green]Ingestion complete![/green]")


def query(question: str):
    """Run a single query through the agent."""
    console.print(Panel.fit(f"[bold blue]Legal RAG Assistant - Query[/bold blue]\n[cyan]{question}[/cyan]"))
    
    settings = get_settings()
    
    if not settings.xai_api_key and not settings.groq_api_key and not settings.google_api_key and not settings.openai_api_key:
        console.print("[red]Error: XAI_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY not set in .env file[/red]")
        return
    
    # Load vector stores
    console.print("\n[blue]Loading vector stores...[/blue]")
    vector_store_manager = LegalVectorStoreManager(
        embedding_model="huggingface",
        openai_api_key=settings.openai_api_key
    )
    
    case_law_store = vector_store_manager.load_case_law_store()
    contracts_store = vector_store_manager.load_contracts_store()
    
    if not case_law_store and not contracts_store:
        console.print("[red]Error: No vector stores found. Run: python main.py ingest[/red]")
        return
    
    # Initialize retriever
    retriever = LegalRetriever(
        case_law_store=case_law_store,
        contracts_store=contracts_store,
        k=5
    )
    
    # Initialize agent
    console.print("[blue]Initializing LangGraph agent...[/blue]")
    try:
        agent = LegalLangGraphAgent(retriever=retriever)
    except ValueError as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        return
    
    # Run query
    console.print("\n[blue]Processing query...[/blue]\n")
    result = agent.run(question)
    
    # Print results
    console.print("\n[bold green]ANSWER:[/bold green]")
    console.print(Panel.fit(result["answer"]))
    
    if result["sources"]:
        console.print(f"\n[bold blue]SOURCES ({len(result['sources'])}):[/bold blue]")
        for source in result["sources"]:
            console.print(f"  - {source}")
    
    console.print(f"\n[blue]Retrieval attempts: {result['retrieval_attempts']}[/blue]")
    console.print(f"[blue]Sub-queries: {', '.join(result['sub_queries'])}[/blue]")


def evaluate():
    """Run the full evaluation suite."""
    console.print(Panel.fit("[bold blue]Legal RAG Assistant - Evaluation Suite[/bold blue]"))
    
    settings = get_settings()
    
    if not settings.xai_api_key and not settings.groq_api_key and not settings.google_api_key and not settings.openai_api_key:
        console.print("[red]Error: XAI_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY not set in .env file[/red]")
        return
    
    # Load vector stores
    console.print("\n[blue]Loading vector stores...[/blue]")
    vector_store_manager = LegalVectorStoreManager(
        embedding_model="huggingface",
        openai_api_key=settings.openai_api_key
    )
    
    case_law_store = vector_store_manager.load_case_law_store()
    contracts_store = vector_store_manager.load_contracts_store()
    
    if not case_law_store and not contracts_store:
        console.print("[red]Error: No vector stores found. Run: python main.py ingest[/red]")
        return
    
    # Initialize retriever
    retriever = LegalRetriever(
        case_law_store=case_law_store,
        contracts_store=contracts_store,
        k=5
    )
    
    # Initialize agent
    console.print("[blue]Initializing LangGraph agent...[/blue]")
    try:
        agent = LegalLangGraphAgent(retriever=retriever)
    except ValueError as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        return
    
    # Run evaluation
    console.print("\n[blue]Running evaluation suite...[/blue]")
    evaluator = LegalAgentEvaluator()
    
    results_df = evaluator.run_full_evaluation(agent, TEST_CASES)
    
    console.print(f"\n[green]Evaluation complete![/green]")
    console.print(f"[blue]Results saved to evaluation_results.csv[/blue]")
    
    # Save to CSV
    results_df.to_csv("evaluation_results.csv", index=False)


def serve():
    """Start the FastAPI server."""
    console.print(Panel.fit("[bold blue]Legal RAG Assistant - FastAPI Server[/bold blue]"))
    
    import uvicorn
    from api.app import app
    
    console.print("\n[blue]Starting FastAPI server on http://0.0.0.0:8000[/blue]")
    console.print("[yellow]Press Ctrl+C to stop[/yellow]\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


def compare_models(question: str):
    """Compare different models on a query."""
    console.print(Panel.fit(f"[bold blue]Legal RAG Assistant - Model Comparison[/bold blue]\n[cyan]{question}[/cyan]"))
    
    settings = get_settings()
    
    if not settings.xai_api_key and not settings.groq_api_key and not settings.google_api_key and not settings.openai_api_key:
        console.print("[red]Error: XAI_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY not set in .env file[/red]")
        return
    
    console.print("\n[blue]Comparing models...[/blue]")
    evaluator = LegalAgentEvaluator()
    
    models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    results_df = evaluator.compare_models(question, models)
    
    console.print(f"\n[green]Model comparison complete![/green]")
    console.print(f"[blue]Results saved to model_comparison.csv[/blue]")
    
    # Save to CSV
    results_df.to_csv("model_comparison.csv", index=False)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Legal RAG Assistant - Agentic RAG System for Legal Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py ingest              # Run ingestion pipeline
  python main.py query "Does COVID-19 qualify as force majeure?"
  python main.py evaluate            # Run full evaluation suite
  python main.py serve               # Start FastAPI server
  python main.py compare-models "What are GDPR requirements?"
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    subparsers.add_parser("ingest", help="Run document ingestion pipeline")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("question", help="Legal question to answer")
    
    # Evaluate command
    subparsers.add_parser("evaluate", help="Run full evaluation suite")
    
    # Serve command
    subparsers.add_parser("serve", help="Start FastAPI server")
    
    # Compare models command
    compare_parser = subparsers.add_parser("compare-models", help="Compare models on a query")
    compare_parser.add_argument("question", help="Question to compare models on")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == "ingest":
        ingest()
    elif args.command == "query":
        query(args.question)
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "serve":
        serve()
    elif args.command == "compare-models":
        compare_models(args.question)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
