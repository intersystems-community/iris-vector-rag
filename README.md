# RAG Templates for InterSystems IRIS

This repository contains implementation templates for various Retrieval Augmented Generation (RAG) techniques using InterSystems IRIS.

## RAG Techniques Implemented

1. **BasicRAG**: Standard embedding-based retrieval
2. **HyDE**: Hypothetical Document Embeddings
3. **CRAG**: Corrective Retrieval Augmented Generation
4. **ColBERT**: Contextualized Late Interaction over BERT
5. **NodeRAG**: Heterogeneous graph-based retrieval
6. **GraphRAG**: Knowledge graph-based retrieval

## Features

- All techniques are implemented with Python and InterSystems IRIS
- Comprehensive Test-Driven Development (TDD) approach
- Tests with 1000+ real medical documents
- Performance benchmarking and comparison
- Scalable architecture for large document sets

## Requirements

- Python 3.11+
- Poetry for dependency management
- InterSystems IRIS 2025.1+
- Docker for test containers

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-templates.git
   cd rag-templates
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Set up IRIS:
   ```bash
   make start-iris
   ```

4. Load test data:
   ```bash
   make load-data
   ```

## Running Tests

### Running Tests with 1000+ Documents

As per our project requirements, all tests run with at least 1000 documents by default. This ensures testing in realistic conditions with substantial data volumes.

Basic test run:
```bash
make test-1000
```

Testing with real PMC documents:
```bash
make test-real-pmc-1000
```

Full test suite with reporting:
```bash
make test-all-1000-docs
```

Individual technique tests:
```bash
poetry run pytest -xvs tests/test_basic_1000.py
poetry run pytest -xvs tests/test_colbert_1000.py
poetry run pytest -xvs tests/test_noderag_1000.py
```

For more details on testing with 1000+ documents, see [1000_DOCUMENT_TESTING.md](1000_DOCUMENT_TESTING.md).

## Technique Documentation

Each RAG technique has detailed implementation documentation:

- [COLBERT_IMPLEMENTATION.md](COLBERT_IMPLEMENTATION.md)
- [NODERAG_IMPLEMENTATION.md](NODERAG_IMPLEMENTATION.md)
- [GRAPHRAG_IMPLEMENTATION.md](GRAPHRAG_IMPLEMENTATION.md)
- [CONTEXT_REDUCTION_STRATEGY.md](CONTEXT_REDUCTION_STRATEGY.md)

## Project Structure

```
rag-templates/
├── basic_rag/           # Standard RAG implementation
├── colbert/             # ColBERT implementation
├── common/              # Shared utilities
├── crag/                # Corrective RAG implementation
├── data/                # Data loading and processing
├── eval/                # Evaluation and benchmarking
├── graphrag/            # GraphRAG implementation
├── hyde/                # HyDE implementation
├── noderag/             # NodeRAG implementation
└── tests/               # Test suite
```

## Test-Driven Development

This project follows strict TDD principles:

1. **Test-First Development**: All features start with failing tests
2. **Red-Green-Refactor**: Write failing test, implement minimum code to pass, refactor
3. **Real End-to-End Tests**: Tests verify that RAG techniques actually work with real data
4. **Complete Pipeline Testing**: Test full pipeline from data ingestion to answer generation
5. **Assert Actual Results**: Tests make assertions on actual result properties

## Running the Demo

Each RAG technique has a demo script:

```bash
poetry run python demo_basic_rag.py
poetry run python demo_colbert.py
# etc.
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
