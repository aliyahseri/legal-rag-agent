"""
FastAPI application for Legal RAG Assistant.
Phase 9: REST API endpoints for querying, evaluation, and ingestion.
"""

import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from config.settings import get_settings
from ingestion.loader import LegalDocumentLoader
from ingestion.chunker import LegalTextChunker
from ingestion.embedder import EmbeddingFactory
from retrieval.vector_store import LegalVectorStoreManager
from retrieval.retriever import LegalRetriever
from agent.langgraph_agent import LegalLangGraphAgent
from evaluation.evaluator import LegalAgentEvaluator
from evaluation.test_cases import TEST_CASES
from llm.llm_factory import get_llm
from rich.console import Console

console = Console()

# Initialize settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title="Legal RAG Assistant API",
    description="Agentic RAG system for legal research",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
case_law_store = None
contracts_store = None
retriever = None
agent = None
vector_store_manager = None


# Request/Response Models
class QueryRequest(BaseModel):
    question: str = Field(..., description="Legal question to answer")
    doc_type: str = Field(default="all", description="Document type: 'case_law', 'contracts', or 'all'")
    use_agent: bool = Field(default=True, description="Use LangGraph agent if True, basic RAG if False")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float
    retrieval_count: int


class HealthResponse(BaseModel):
    status: str
    vector_stores_loaded: bool


class IngestResponse(BaseModel):
    status: str
    doc_count: int
    chunk_count: int
    message: str


# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add request timing to all responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load vector stores on startup."""
    global case_law_store, contracts_store, retriever, agent, vector_store_manager
    
    console.print("[blue]Starting FastAPI application...[/blue]")
    
    try:
        # Initialize vector store manager
        vector_store_manager = LegalVectorStoreManager(
            embedding_model="huggingface",
            openai_api_key=settings.openai_api_key
        )
        
        # Load existing vector stores
        case_law_store = vector_store_manager.load_case_law_store()
        contracts_store = vector_store_manager.load_contracts_store()
        
        # Initialize retriever
        if case_law_store or contracts_store:
            retriever = LegalRetriever(
                case_law_store=case_law_store,
                contracts_store=contracts_store,
                k=5
            )
        
        # Initialize agent if retriever is available
        if retriever:
            try:
                agent = LegalLangGraphAgent(retriever=retriever)
            except ValueError as e:
                console.print(f"[yellow]Agent initialization failed (missing API key): {e}[/yellow]")
        
        console.print(f"[green]Startup complete. Vector stores loaded: {case_law_store is not None and contracts_store is not None}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during startup: {e}[/red]")


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        vector_stores_loaded=(case_law_store is not None and contracts_store is not None)
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the legal RAG system.
    
    Args:
        request: QueryRequest with question, doc_type, and use_agent flags.
        
    Returns:
        QueryResponse with answer, sources, latency, and retrieval count.
    """
    start_time = time.time()
    
    if not retriever:
        raise HTTPException(status_code=503, detail="Vector stores not loaded. Run /ingest first.")
    
    if not agent and request.use_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized. Check API key configuration.")
    
    try:
        if request.use_agent and agent:
            # Use LangGraph agent
            result = agent.run(request.question)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            retrieval_count = result.get("retrieval_attempts", 0)
        else:
            # Basic RAG
            if request.doc_type == "case_law":
                docs = retriever.retrieve_case_law(request.question)
            elif request.doc_type == "contracts":
                docs = retriever.retrieve_contracts(request.question)
            else:
                results = retriever.retrieve_parallel(request.question)
                docs = results["case_law"] + results["contracts"]
            
            context = retriever.format_context(docs)
            sources = [f"{doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}" 
                      for doc in docs]
            
            # Simple answer generation
            from agent.prompts import RAG_PROMPT
            llm = get_llm()
            prompt = RAG_PROMPT.format(context=context, question=request.question)
            response = llm.invoke(prompt)
            answer = response.content
            retrieval_count = len(docs)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            latency_ms=round(latency_ms, 2),
            retrieval_count=retrieval_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/evaluate")
async def run_evaluation():
    """
    Run the full evaluation suite.
    
    Returns:
        JSON report with evaluation results.
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")
    
    try:
        evaluator = LegalAgentEvaluator()
        
        results_df = evaluator.run_full_evaluation(agent, TEST_CASES)
        
        return {
            "status": "complete",
            "results": results_df.to_dict(orient="records"),
            "summary": {
                "average_overall_score": float(results_df["overall_score"].mean()),
                "average_faithfulness": float(results_df["faithfulness"].mean()),
                "average_task_success": float(results_df["task_success"].mean()),
                "total_cases": len(TEST_CASES)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running evaluation: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents():
    """
    Trigger re-ingestion of documents from data/ directory.
    
    Returns:
        IngestResponse with ingestion statistics.
    """
    global case_law_store, contracts_store, retriever, agent, vector_store_manager
    
    try:
        console.print("[blue]Starting document ingestion...[/blue]")
        
        # Load documents
        loader = LegalDocumentLoader()
        case_law_docs = loader.load_case_law("data/case_law")
        contract_docs = loader.load_contracts("data/contracts")
        
        # If no PDFs, generate synthetic docs
        if not case_law_docs and not contract_docs:
            console.print("[yellow]No PDFs found, generating synthetic documents...[/yellow]")
            synthetic_case_law = loader.generate_synthetic_documents("data/case_law")
            synthetic_contracts = loader.generate_synthetic_documents("data/contracts")
            case_law_docs = synthetic_case_law
            contract_docs = synthetic_contracts
        
        doc_count = len(case_law_docs) + len(contract_docs)
        
        # Chunk documents
        chunker = LegalTextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        case_law_chunks = chunker.chunk_documents(case_law_docs)
        contract_chunks = chunker.chunk_documents(contract_docs)
        
        chunk_count = len(case_law_chunks) + len(contract_chunks)
        
        # Build vector stores
        if not vector_store_manager:
            vector_store_manager = LegalVectorStoreManager(
                embedding_model="huggingface",
                openai_api_key=settings.openai_api_key
            )
        
        case_law_store, contracts_store = vector_store_manager.build_or_load_stores(
            case_law_chunks,
            contract_chunks
        )
        
        # Update retriever and agent
        if case_law_store or contracts_store:
            retriever = LegalRetriever(
                case_law_store=case_law_store,
                contracts_store=contracts_store,
                k=5
            )
            
            try:
                agent = LegalLangGraphAgent(retriever=retriever)
            except ValueError as e:
                console.print(f"[yellow]Agent initialization failed (missing API key): {e}[/yellow]")
        
        console.print(f"[green]Ingestion complete: {doc_count} docs, {chunk_count} chunks[/green]")
        
        return IngestResponse(
            status="success",
            doc_count=doc_count,
            chunk_count=chunk_count,
            message="Documents ingested successfully"
        )
        
    except Exception as e:
        console.print(f"[red]Error during ingestion: {e}[/red]")
        raise HTTPException(status_code=500, detail=f"Error during ingestion: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
