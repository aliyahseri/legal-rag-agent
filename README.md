# Legal RAG Agent

An agentic RAG assistant for legal research that decomposes a question, searches separate case-law and contract indexes in parallel, and writes a cited, structured answer.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-agent-1C3C3C?logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-vector%20search-0467DF)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white)

## Overview

Legal questions often span more than one kind of source: a question about a non-compete touches both statutes/case law and the wording of the contract itself. This project keeps the two corpora in separate vector indexes and lets a LangGraph agent orchestrate retrieval over both before answering.

Key features (all present in the code):

- **LangGraph agent** (`agent/langgraph_agent.py`): a five-node state graph
  `decompose_question -> retrieve_documents -> check_context_sufficiency -> generate_answer -> format_response`,
  with a conditional edge that can loop back to retrieval (capped at 3 attempts).
- **Question decomposition**: the LLM rewrites the question into 2-3 searchable sub-queries.
- **Dual FAISS indexes with parallel search** (`retrieval/`): `case_law_index` and `contracts_index`, searched concurrently with a `ThreadPoolExecutor`, results de-duplicated by content.
- **Cited answers**: context is passed to the LLM with `[Source: file, Page: N]` markers, the answer is structured as Summary / Analysis / Relevant Precedents / Conclusion, and every prompt ends with a "research assistance only, consult a licensed attorney" disclaimer.
- **Pluggable LLM** (`llm/llm_factory.py`): Groq (default, `llama-3.3-70b-versatile`), Google Gemini (`gemini-2.0-flash`), xAI (`grok-beta`, via the OpenAI-compatible endpoint) or OpenAI, selected with `PRIMARY_LLM`.
- **Local embeddings**: `BAAI/bge-base-en-v1.5` via HuggingFace sentence-transformers on CPU, so indexing needs no API key.
- **Interfaces**: a CLI (`main.py`: `ingest`, `query`, `evaluate`, `serve`, `compare-models`) and a FastAPI service (`/health`, `/query`, `/evaluate`, `/ingest`). `/query` can run either the full agent or a single-pass RAG baseline (`use_agent: false`) filtered to one corpus.
- **Evaluation harness** (`evaluation/`): 10 hand-written test cases scored on task success, a faithfulness heuristic, latency and an estimated cost (details under [Results](#results)).
- **Synthetic corpus**: scripts that generate fictional legal documents so the pipeline runs end to end without any real data.

## Tech stack

| Area | Choice | Why (as evident in the code) |
|---|---|---|
| Orchestration | LangGraph `StateGraph` | Explicit nodes and a conditional retrieve-again loop instead of a free-form agent |
| LLM access | LangChain chat models (`langchain-groq`, `langchain-google-genai`, `langchain-openai`) | One `invoke()` interface across four providers |
| Embeddings | `BAAI/bge-base-en-v1.5` (`langchain-huggingface`, sentence-transformers) | Runs locally on CPU with normalized embeddings, no embedding API cost |
| Vector store | FAISS (`faiss-cpu`, `langchain-community`) | Lightweight, file-based indexes saved under `vector_stores/` |
| Chunking | `RecursiveCharacterTextSplitter` (800 chars, 100 overlap) | Splits on paragraph, line and sentence boundaries first |
| PDF parsing | PyMuPDF (`PyMuPDFLoader`) | Page-level documents, page numbers kept for citations |
| Config | `pydantic-settings` reading `.env` | Typed settings with defaults |
| API / CLI | FastAPI + Uvicorn, argparse + Rich | REST endpoints and readable console output |
| Synthetic data | ReportLab | Generates the demo PDFs |
| Tests | pytest | Unit tests for chunking, retrieval and prompt templates |

## Demo / Screenshots

<!-- TODO: add a screenshot at docs/screenshot.png -->
![Screenshot](docs/screenshot.png)

> Owner: capture a terminal screenshot of `python main.py query "Does COVID-19 qualify as a force majeure event?"` showing the agent nodes, the answer panel and the sources list, and save it as `docs/screenshot.png`.

## Dataset

**No data is included in this repository**, and no real legal documents are used. All documents are **synthetic**, written for this project.

There are two synthetic sources:

1. **Fallback `.txt` documents (used by default).** If `data/case_law/` and `data/contracts/` contain no PDFs, `ingest` writes 5 plain-text documents (force majeure, non-compete, GDPR, IP assignment, liquidated damages) into **both** folders and indexes them. Both indexes therefore contain the same 5 documents.
2. **10 synthetic PDFs (optional).** `python scripts/generate_synthetic_docs.py` writes 5 case-law PDFs to `./case_law/` and 5 contract PDFs to `./contracts/` **at the repository root**. Ingest does not read those folders, so move the files into `data/` yourself:

```
data/
├── case_law/
│   ├── case_1_force_majeure.pdf
│   ├── case_3_gdpr_compliance.pdf
│   ├── case_7_whistleblower_protection.pdf
│   ├── case_9_trade_secrets_dtsa.pdf
│   └── case_10_arbitration_enforceability.pdf
└── contracts/
    ├── contract_2_non_compete.pdf
    ├── contract_4_ip_assignment.pdf
    ├── contract_5_liquidated_damages.pdf
    ├── contract_6_auto_renewal.pdf
    └── contract_8_construction.pdf
```

You can also drop your own PDFs into `data/case_law/` and `data/contracts/`; each page becomes a document with `source`, `page` and `doc_type` metadata.

**Indexes / model weights.** `ingest` writes FAISS indexes to `vector_stores/case_law_index/` and `vector_stores/contracts_index/` (git-ignored). The embedding model `BAAI/bge-base-en-v1.5` is downloaded automatically from the Hugging Face Hub on first use. No model is trained in this project.

## Installation

```bash
git clone https://github.com/aliyahseri/legal-rag-agent.git
cd legal-rag-agent

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Configure environment variables:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Then edit `.env`. You need **one** LLM key, and `PRIMARY_LLM` must match it:

| Provider | Key | `PRIMARY_LLM` |
|---|---|---|
| Groq (default) | `GROQ_API_KEY` | `groq` (model set by `GROQ_MODEL`) |
| Google Gemini | `GOOGLE_API_KEY` | `gemini` |
| xAI | `XAI_API_KEY` | `xai` |
| OpenAI | `OPENAI_API_KEY` | an OpenAI model name, e.g. `gpt-4o-mini` (the value is passed as the model) |

An OpenAI key is **not** required. Embeddings run locally.

## Usage

All commands are run from the repository root.

```bash
# (optional) generate the 10 synthetic PDFs, then move them into data/ (see Dataset)
python scripts/generate_synthetic_docs.py

# build the FAISS indexes (falls back to synthetic .txt docs if data/ has no PDFs)
python main.py ingest

# ask a question through the LangGraph agent
python main.py query "Does COVID-19 qualify as a force majeure event?"

# run the 10-case evaluation suite, writes evaluation_results.csv
python main.py evaluate

# start the API on http://0.0.0.0:8000 (docs at /docs)
python main.py serve
```

To re-index after changing documents, delete `vector_stores/` first; an existing index is loaded, not rebuilt.

API examples:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Are non-compete agreements enforceable in California?", "use_agent": true}'

# single-pass RAG over one corpus only
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the GDPR data subject rights?", "use_agent": false, "doc_type": "contracts"}'

curl -X POST http://localhost:8000/ingest
curl -X POST http://localhost:8000/evaluate
```

`python main.py compare-models "<question>"` also exists, but see Known issues: it currently calls the same configured LLM for every listed model name.

Tests (use `python -m pytest` so the repository root is on `sys.path`; there is no `conftest.py`):

```bash
python -m pytest tests -v
```

The tests download the embedding model on first run and write to `vector_stores/` and `data/case_law/` (see Known issues).

## Results

No benchmark results are reported. Earlier versions of this README showed an evaluation table and a model-comparison table; those numbers were placeholders, not measured output, and have been removed.

What `python main.py evaluate` actually computes for each of the 10 cases in `evaluation/test_cases.py`:

- **Task success**: string checks on the answer: each expected topic and expected source substring present, at least `min_citations` `[Source: ...]` citations, the disclaimer present, length of at least 100 characters. Score = 1.0 minus 0.2 per failed check.
- **Faithfulness**: a keyword-overlap heuristic, not RAGAS or NLI. The answer is split into sentences; a sentence counts as supported if at least 30% of its words appear anywhere in the context. As currently wired it always returns 0 (see Known issues).
- **Latency**: wall-clock time of an additional agent run.
- **Cost**: a rough estimate from word counts, priced at GPT-4o rates ($2.50 / $10.00 per 1M input/output tokens), regardless of which provider actually ran.
- **Overall**: 0.4 x faithfulness + 0.4 x task success + 0.1 x latency score + 0.1 x cost score.

Results are printed as a table and saved to `evaluation_results.csv` (git-ignored).

## Project structure

```
legal-rag-agent/
├── main.py                      # CLI: ingest | query | evaluate | serve | compare-models
├── api/app.py                   # FastAPI service
├── agent/
│   ├── langgraph_agent.py       # LangGraph state graph (the agent used by CLI and API)
│   ├── prompts.py               # System, RAG, ReAct and chain-of-thought prompts
│   ├── tools.py                 # LangChain @tool wrappers around the retriever
│   └── react_agent.py           # Experimental ReAct agent (not wired in)
├── retrieval/
│   ├── vector_store.py          # Build / load FAISS indexes
│   └── retriever.py             # Per-corpus and parallel retrieval, context formatting
├── ingestion/
│   ├── loader.py                # PDF loading + synthetic .txt fallback
│   ├── chunker.py               # Recursive text splitting
│   └── embedder.py              # Embedding factory (not used by the main pipeline)
├── llm/llm_factory.py           # Groq / Gemini / xAI / OpenAI selection
├── evaluation/
│   ├── evaluator.py             # Metrics and model comparison
│   └── test_cases.py            # 10 evaluation cases
├── config/settings.py           # pydantic-settings configuration
├── scripts/generate_synthetic_docs.py   # 10 synthetic legal PDFs
├── tests/                       # pytest suite
├── data/README.md               # Where documents go (data not included)
├── .env.example
└── requirements.txt
```

## Challenges & what I learned

- **Separating corpora.** Keeping case law and contracts in different FAISS indexes and querying them in parallel makes the provenance of each retrieved chunk explicit and lets the API restrict a query to one corpus.
- **Making the agent loop explicit.** Modelling the workflow as a LangGraph state graph (rather than a free-form ReAct loop) made each step observable and bounded the number of retrieval rounds. It also showed how fragile LLM-output parsing is: the sufficiency check's substring test is the bug listed below.
- **Citations end to end.** Preserving `source` and `page` metadata from loading through chunking into the prompt is what makes `[Source: file, Page: N]` citations possible.
- **Provider independence.** A small LLM factory let the same graph run on Groq, Gemini, xAI or OpenAI, while local BGE embeddings removed the need for a paid embedding API.
- **Evaluating RAG honestly is hard.** The keyword-overlap faithfulness score and word-count cost estimate are cheap proxies; building them made the gap to proper LLM-judge or NLI-based evaluation clear. [TODO: confirm]

## Known issues & limitations

1. `agent/langgraph_agent.py` line 224: `"SUFFICIENT" in content` is also true for "INSUFFICIENT", so the context is always judged sufficient and the retrieve-more loop never runs.
2. Faithfulness is always 0: `main.py` line 195 and `api/app.py` line 214 call `run_full_evaluation` without `context_docs_map`, and `measure_faithfulness` returns 0.0 for empty context.
3. `scripts/generate_synthetic_docs.py` writes PDFs to `./case_law` and `./contracts`, but ingest reads `data/case_law` and `data/contracts`, so by default ingest uses the 5-document `.txt` fallback, written identically into both indexes.
4. `retrieval/vector_store.py` `build_or_load_stores` loads an existing index and never rebuilds it, so new documents (and `POST /ingest`) are ignored until `vector_stores/` is deleted.
5. `evaluation/evaluator.py` `measure_latency` runs the agent again instead of timing the first run, doubling LLM calls and cost during evaluation.
6. `compare-models` (`main.py` line 230, `evaluator.compare_models`) invokes the same default LLM for every "model" name; its "quality" is answer length divided by 500.
7. `agent/react_agent.py` is not used anywhere; `check_jurisdiction` takes two arguments, which a text ReAct agent cannot pass, and the fallback `REACT_AGENT_PROMPT` lacks the `{tools}` variable that `create_react_agent` requires.
8. Settings `EMBEDDING_MODEL`, `DEEPSEEK_API_KEY`, `HUGGINGFACE_API_TOKEN`, `LOG_LEVEL` and `VECTOR_STORE_PATH` are read but never used (index paths and the embedding model are hardcoded).
9. `api/app.py` endpoints are `async def` but make blocking LLM and embedding calls, which blocks the event loop under concurrent requests.
10. `FAISS.load_local(..., allow_dangerous_deserialization=True)` unpickles index files; only load indexes you built yourself.
11. `llm/llm_factory.py`: if `PRIMARY_LLM` does not match an available key, it falls through to OpenAI and passes `PRIMARY_LLM` (e.g. `groq`) as the OpenAI model name.
12. The tests write a 2-document index into `vector_stores/` (overwriting a real index) and `.txt` files into `data/case_law/`; `test_langgraph_agent_initialization` passes unsupported arguments, so its `try/except` swallows a `TypeError` and it asserts nothing.
13. Evaluation cases 6-10 (auto-renewal, whistleblower, construction, trade secrets, arbitration) have no matching document in the default 5-document fallback corpus.
14. The cost metric ignores retrieved context in its token estimate and always uses GPT-4o prices.
15. CORS allows all origins with credentials; there is no authentication or rate limiting on the API.

## Future improvements

- Fix the sufficiency check (exact-match parsing or structured output) so the retrieval loop can actually trigger.
- Return retrieved documents from the agent and pass them to the evaluator; replace the keyword heuristic with an LLM-judge or NLI faithfulness metric.
- Point the PDF generator and loader at the same directories and add a `--rebuild` flag to ingest.
- Make `compare-models` build a separate LLM per model and record real token usage from provider responses.
- Use `settings.vector_store_path` and `settings.embedding_model`, and isolate tests with `tmp_path`.
- Run blocking calls in a thread pool (or make the endpoints sync) and add API auth.

## Credits / Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph)
- [FAISS](https://github.com/facebookresearch/faiss) (Meta AI)
- [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) embedding model (MIT license)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF), [ReportLab](https://www.reportlab.com/), [FastAPI](https://fastapi.tiangolo.com/), [Rich](https://github.com/Textualize/rich)
- LLM providers: Groq (Llama 3.3 70B), Google Gemini, xAI, OpenAI

All legal documents in this project are synthetic. **This is research assistance only and not legal advice. Consult a licensed attorney.**

## License

MIT, see [LICENSE](LICENSE).

## Author

**Aliyah Seri**

- GitHub: [aliyahseri](https://github.com/aliyahseri)
