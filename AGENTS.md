# iris-vector-rag-private Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-01-03

## Active Technologies
- Python 3.12 + intersystems-irispython>=5.1.2, iris-vector-graph>=1.6.0 (required for HybridGraphRAG) (060-fix-users-tdyar)
- InterSystems IRIS (vector database with SQL interface) (060-fix-users-tdyar)
- Python 3.12 + intersystems-irispython>=5.1.2, iris-vector-graph>=1.6.0 (required for GraphRAG) (060-fix-users-tdyar)
- Python 3.12 + `iris_llm` (optional wheel), `dspy-ai` (existing), `langchain` (existing), `iris` (DBAPI, test-time) (065-iris-llm-substrate)
- IRIS SQL (`RAG.Entities`, `RAG.EntityRelationships`, `RAG.SourceDocuments`) (065-iris-llm-substrate)
- Python 3.12 (IRIS embedded), Python 3.12 (test/benchmark client via spike venv) + `iris.sql` (IRIS embedded Python), `numpy==1.26.4` (pre-installed in container), `iris.dbapi` (external client) (067-colbert-plaid-sp)
- `RAG.ColBERTCentroids`, `RAG.ColBERTDocCentroids`, `RAG.DocumentTokenEmbeddings` (from feature 066) (067-colbert-plaid-sp)

- Python 3.12, Docker, GitHub Actions (Ubuntu 24.04) + Checkov, Docker, GitHub Actions (001-fix-ci-security-failures)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style

Python 3.12, Docker, GitHub Actions (Ubuntu 24.04): Follow standard conventions

## Recent Changes
- 067-colbert-plaid-sp: Added Python 3.12 (IRIS embedded), Python 3.12 (test/benchmark client via spike venv) + `iris.sql` (IRIS embedded Python), `numpy==1.26.4` (pre-installed in container), `iris.dbapi` (external client)
- 065-iris-llm-substrate: Added Python 3.12 + `iris_llm` (optional wheel), `dspy-ai` (existing), `langchain` (existing), `iris` (DBAPI, test-time)
- 060-fix-users-tdyar: Added Python 3.12 + intersystems-irispython>=5.1.2, iris-vector-graph>=1.6.0 (required for GraphRAG)


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
