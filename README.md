# RAG-Templates

## 🎯 Project Status: Complete & Ready for Integration

RAG-Templates is now **complete as a reusable framework** with all core components delivered:
- ✅ 4 Production RAG Pipelines (BasicRAG, CRAG, GraphRAG, BasicRAGReranking)
- ✅ Enterprise IRIS Backend Integration
- ✅ Generic Memory Components Architecture
- ✅ Unified Bridge Adapter Interface
- ✅ Incremental Indexing Foundation

**Integration Documentation:**
- 📋 [Project Completion Report](docs/PROJECT_COMPLETION_REPORT_VALIDATED.md) - Validated achievements and honest metrics
- 🔗 [Integration Handoff Guide](docs/INTEGRATION_HANDOFF_GUIDE.md) - How to consume rag-templates in your application
- 🏗️ [Architecture Summary](docs/VALIDATED_ARCHITECTURE_SUMMARY.md) - Service boundaries and performance characteristics
- 🚀 [Production Readiness](docs/PRODUCTION_READINESS_ASSESSMENT.md) - Infrastructure requirements and deployment guidance

**Scope Transition:**
Application-specific features (PRefLexOR bridge, production monitoring, CI/CD) have been re-scoped to the [kg-ticket-resolver](https://github.com/your-org/kg-ticket-resolver) application project.

## Set up

```bash
# 1. Clone the repository
git clone <repository-url>
cd rag-templates

# 2. Set up the Python virtual environment and install dependencies
make setup-env  # This will create .venv and install core dependencies
make install    # This will install all dependencies from requirements.txt

# 3. Activate the virtual environment (if not already done by make setup-env/install)
#    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 4. Start the database
docker-compose up -d

# 5. Initialize and load sample data
make setup-db
make load-data

# 6. Test that you can access the iris_rag package
make validate-iris-rag

# 7. Create your .env file and add your API keys in

# 8. Try the various pipeline scripts! 
# Note: these scripts currently use OpenAI's ChatGPT as the LLM, so having an API key is essential.
# The first time you run any script after loading new data will take a long time due to overhead of initial chunking, embedding, and storage
cd scripts/basic
python try_basic_rag_pipeline.py
```

## 📚 References & Research

### RAG Technique Papers & Implementations

| Technique | Original Paper | Key Repository | Additional Resources |
|-----------|---------------|----------------|---------------------|
| **Basic RAG** | [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | [Facebook Research](https://github.com/facebookresearch/RAG) | [LangChain RAG](https://python.langchain.com/docs/tutorials/rag/) |
| **ColBERT** | [ColBERT: Efficient and Effective Passage Retrieval](https://arxiv.org/abs/2004.12832) | [Stanford ColBERT](https://github.com/stanford-futuredata/ColBERT) | [Pylate Integration](https://github.com/lightonai/pylate) |
| **CRAG** | [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) | [CRAG Implementation](https://github.com/HuskyInSalt/CRAG) | [LangGraph CRAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/) |
| **HyDE** | [Precise Zero-Shot Dense Retrieval](https://arxiv.org/abs/2212.10496) | [HyDE Official](https://github.com/texttron/hyde) | [LangChain HyDE](https://python.langchain.com/docs/how_to/hyde/) |
| **GraphRAG** | [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130) | [Microsoft GraphRAG](https://github.com/microsoft/graphrag) | [Neo4j GraphRAG](https://github.com/neo4j/neo4j-graphrag-python) |
| **NodeRAG** | [Hierarchical Text Retrieval](https://arxiv.org/abs/2310.20501) | [NodeRAG Implementation](https://github.com/microsoft/noderag) | [Hierarchical Retrieval](https://python.langchain.com/docs/how_to/parent_document_retriever/) |

### Core Technologies

- **Vector Databases**: [InterSystems IRIS Vector Search](https://docs.intersystems.com/iris20241/csp/docbook/DocBook.UI.Page.cls?KEY=GSQL_vecsearch)
- **Embeddings**: [Sentence Transformers](https://github.com/UKPLab/sentence-transformers), [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- **LLM Integration**: [LangChain](https://github.com/langchain-ai/langchain), [OpenAI API](https://platform.openai.com/docs/api-reference)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

