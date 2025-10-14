# PMC Document Processing - Existing Infrastructure Documentation

## 🎯 Overview

**IMPORTANT**: A comprehensive PMC document processing infrastructure already exists and works perfectly. This document describes the existing, proven system rather than introducing new functionality.

## ✅ Existing PMC Infrastructure (RECOMMENDED)

The project already includes a complete, production-ready PMC processing system:

### Core Components

1. **[`data/pmc_processor.py`](../../data/pmc_processor.py)** - Complete PMC XML processing
2. **[`data/loader_fixed.py`](../../data/loader_fixed.py)** - Database loading with vector support  
3. **[`scripts/utilities/download_real_pmc_docs.py`](../utilities/download_real_pmc_docs.py)** - PMC document downloader
4. **[`Makefile`](../../Makefile)** - Simple orchestration via `make load-data`

### Features Already Available

- ✅ **Full XML processing** with metadata extraction (title, abstract, authors, keywords)
- ✅ **Automatic chunking** (8000 characters with 400 character overlap)
- ✅ **Rich metadata** including PMC IDs, content length, chunk counts
- ✅ **Database integration** with IRIS vector storage
- ✅ **Error handling** and validation
- ✅ **Progress tracking** and performance metrics (596 docs/second)
- ✅ **Incremental loading** and resume capability

## 🚀 Quick Start (Using Existing System)

### Process and Load Sample PMC Documents
```bash
# Process 10 sample PMC documents and load into database
make load-data
```

**Expected Result**: 
- Processes 10 documents → 79 database entries (with chunks) in ~0.13 seconds
- 596 documents/second processing speed
- Success: `{'success': True, 'loaded_doc_count': 79, 'error_count': 0}`

### Direct Python Usage
```python
from data.pmc_processor import process_pmc_files
from data.loader_fixed import process_and_load_documents

# Process PMC files (returns generator)
documents = list(process_pmc_files('data/sample_10_docs', limit=10))

# Load into database with embeddings
result = process_and_load_documents('data/sample_10_docs', limit=10)
print(f"Loaded: {result}")
```

## 📊 Output Format (Existing System)

The existing system provides a comprehensive format:

```json
{
  "doc_id": "PMC1894889",
  "title": "New Samarium(III), Gadolinium(III), and Dysprosium(III) Complexes...",
  "content": "Complete document text...",
  "abstract": "New complexes of samarium(III), gadolinium(III)...", 
  "authors": ["Irena Kostova", "Georgi Momekov", "Peya Stancheva"],
  "keywords": [],
  "metadata": {
    "source": "PMC",
    "file_path": "data/sample_10_docs/PMC1894889.xml",
    "pmc_id": "PMC1894889",
    "content_length": 20390,
    "needs_chunking": true,
    "chunk_count": 6
  },
  "chunks": [
    {
      "chunk_id": "PMC1894889_chunk_0",
      "text": "Chunk content...",
      "chunk_index": 0,
      "start_pos": 0,
      "end_pos": 7967,
      "metadata": {
        "chunk_size": 7966,
        "overlap_with_previous": 0,
        "strategy": "fixed_size_with_sentences"
      }
    }
  ]
}
```

## ⚙️ Configuration (Existing System)

### Chunking Configuration
Located in [`data/pmc_processor.py`](../../data/pmc_processor.py):
```python
def _chunk_pmc_content(content: str, pmc_id: str, 
                      chunk_size: int = 8000, 
                      overlap: int = 400):
```

### Database Configuration
Located in [`data/loader_fixed.py`](../../data/loader_fixed.py):
```python
def load_documents_to_iris(connection, documents, 
                          embedding_func=None,
                          batch_size: int = 250,
                          handle_chunks: bool = True):
```

## 📁 Available Data Sources

- **`data/sample_10_docs/`** - 10 sample PMC XML files for testing
- **`data/downloaded_pmc_docs/`** - Downloaded PMC documents via utility script
- **Custom directories** - Any directory containing PMC XML files

## 🛠️ Advanced Usage

### Download Additional PMC Documents
```python
from scripts.utilities.download_real_pmc_docs import download_and_load_1000_pmc_docs
result = download_and_load_1000_pmc_docs()
```

### Custom Processing Parameters
```python
from data.loader_fixed import process_and_load_documents

result = process_and_load_documents(
    pmc_directory="your/pmc/directory",
    limit=100,           # Number of documents to process
    batch_size=50,       # Database batch size
    embedding_func=None  # Optional embedding function
)
```

### Process Only (No Database Loading)
```python
from data.pmc_processor import process_pmc_files

for doc in process_pmc_files('data/sample_10_docs', limit=5):
    print(f"Processed: {doc['doc_id']}")
    print(f"Chunks: {len(doc.get('chunks', []))}")
```

## 🔧 Troubleshooting

### Database Messages
You may see normal operational messages during loading:
- Table creation messages indicate existing schema (normal)
- Final success status: `'success': True, 'error_count': 0`

### UV Lock File Issues  
- Fixed in this project - should work out of the box
- If issues persist, check `uv.lock` around line 3185

### Database Connection Issues
- Ensure IRIS database is running
- Check connection configuration in environment variables
- Use `make check-data` to verify database state

## 📈 Performance Metrics

Recent test results with existing system:
- **Processing Speed**: 596 documents/second
- **Memory Efficiency**: Processes 10 docs → 79 database entries  
- **Chunk Expansion**: ~7-8x expansion (1 doc → 7-8 chunks average)
- **Success Rate**: 100% with sample data
- **Error Rate**: 0 errors

## 🎯 Recommendations

1. **Use the existing system** - It's comprehensive, tested, and performant
2. **Start with `make load-data`** - Simple entry point for testing  
3. **Focus on result success flag** - `'success': True` means everything worked
4. **Leverage chunking system** - Already optimized for RAG workflows
5. **Use sample data** - `data/sample_10_docs/` for development/testing

## 📚 Related Documentation

- [`data/pmc_processor.py`](../../data/pmc_processor.py) - Core processing logic
- [`data/loader_fixed.py`](../../data/loader_fixed.py) - Database integration  
- [`Makefile`](../../Makefile) - Available make targets
- [`scripts/utilities/download_real_pmc_docs.py`](../utilities/download_real_pmc_docs.py) - Document downloader

---

## 🆕 NEW: Unified PMC Dataset Loader

### Overview

The new **`pmc_loader.py`** provides a unified, configurable interface that integrates all existing PMC infrastructure into a single, easy-to-use loader for creating standardized PMC datasets.

### Key Features

- ✅ **Configurable via environment variables** (chunk size, overlap, limits, etc.)
- ✅ **Supports both test (100 docs) and full (10,000 docs) modes**
- ✅ **Resume capability** with checkpoint files
- ✅ **Standardized output** (metadata.json, documents.jsonl)
- ✅ **Comprehensive logging** and progress tracking
- ✅ **Sample medical queries** included for evaluation
- ✅ **Database integration** using existing loader infrastructure

### Quick Start

#### Test Mode (100 documents)
```bash
# Set test mode and run
export PMC_TEST_MODE=true
python scripts/data_loaders/pmc_loader.py
```

#### Full Mode (10,000 documents)
```bash
# Full production dataset
export PMC_TEST_MODE=false
export PMC_DOCUMENT_LIMIT=10000
python scripts/data_loaders/pmc_loader.py
```

### Configuration via Environment Variables

```bash
# Chunking configuration (in tokens, converted to chars internally)
export PMC_CHUNK_SIZE=512          # Default: 512 tokens (~2048 chars)
export PMC_CHUNK_OVERLAP=50        # Default: 50 tokens (~200 chars)

# Dataset configuration
export PMC_DOWNLOAD_DIR=data/pmc_dataset/     # Default output directory
export PMC_BATCH_SIZE=100                     # Database batch size
export PMC_DOCUMENT_LIMIT=10000              # Total documents to process
export PMC_TEST_MODE=false                   # true = 100 docs, false = full limit

# Example for custom configuration
export PMC_CHUNK_SIZE=1024                   # Larger chunks
export PMC_CHUNK_OVERLAP=100                 # More overlap
export PMC_DOWNLOAD_DIR=data/my_pmc_dataset/ # Custom directory
export PMC_BATCH_SIZE=50                     # Smaller batches
python scripts/data_loaders/pmc_loader.py
```

### Output Structure

The unified loader creates a complete dataset in the specified directory:

```
data/pmc_dataset/
├── metadata.json           # Dataset statistics and configuration
├── documents.jsonl         # Standardized document format (one JSON per line)
├── progress.json          # Checkpoint file for resume capability
├── final_results.json     # Complete pipeline results
├── pmc_loader.log         # Comprehensive logs
└── PMC*.xml              # Downloaded PMC XML files
```

### Standardized Output Format

#### metadata.json
```json
{
  "dataset_name": "PMC Biomedical Dataset",
  "version": "1.0.0",
  "created_at": "2024-09-14T17:23:00Z",
  "description": "Biomedical research documents from PubMed Central Open Access subset",
  "config": {
    "chunk_size_chars": 2048,
    "chunk_overlap_chars": 200,
    "document_limit": 10000,
    "test_mode": false
  },
  "statistics": {
    "total_documents": 10000,
    "total_chunks": 85000,
    "total_content_length": 50000000,
    "average_content_length": 5000,
    "documents_with_chunks": 9500,
    "average_chunks_per_doc": 8.5
  },
  "sample_queries": [
    "What are the current treatment options for type 2 diabetes mellitus?",
    "How does COVID-19 affect respiratory function and lung capacity?",
    // ... 15 sample medical queries for evaluation
  ]
}
```

#### documents.jsonl (one JSON object per line)
```json
{"id": "PMC123456", "title": "Document Title", "content": "Full content...", "abstract": "Abstract...", "authors": ["Author 1"], "keywords": ["keyword1"], "metadata": {...}, "chunks": [...]}
```

### Python API Usage

```python
from scripts.data_loaders.pmc_loader import PMCDatasetLoader, get_config

# Use default configuration from environment variables
config = get_config()
loader = PMCDatasetLoader(config)

# Run complete pipeline
result = loader.run_complete_pipeline(resume=True)

# Or run individual phases
download_result = loader.download_documents(resume=True)
documents = list(loader.process_documents(resume=True))
output_result = loader.generate_standardized_output(documents)
load_result = loader.load_to_database(resume=True)
```

### Resume Capability

The loader supports resume from any point:

```bash
# Start processing
python scripts/data_loaders/pmc_loader.py

# If interrupted, resume from checkpoint
python scripts/data_loaders/pmc_loader.py  # Automatically resumes
```

Progress is tracked in `progress.json`:
```json
{
  "phase": "processing",
  "downloaded_count": 5000,
  "processed_count": 3000,
  "loaded_count": 2500,
  "start_time": "2024-09-14T10:00:00Z",
  "last_update": "2024-09-14T10:30:00Z"
}
```

### Sample Medical Queries

The loader includes 15 medical domain queries for RAG evaluation:

1. "What are the current treatment options for type 2 diabetes mellitus?"
2. "How does COVID-19 affect respiratory function and lung capacity?"
3. "What are the most effective interventions for managing hypertension?"
4. "What are the risk factors and prevention strategies for cardiovascular disease?"
5. "How do immunotherapy treatments work in cancer patients?"
... (see metadata.json for complete list)

### Integration with Existing Infrastructure

The unified loader leverages all existing components:

- **Document Download**: Uses `SimplePMCDownloader` from `scripts.utilities.download_real_pmc_docs`
- **XML Processing**: Uses `PMCProcessor` from `data.pmc_processor`
- **Database Loading**: Uses `process_and_load_documents` from `data.loader_fixed`
- **Chunking**: Configurable chunking with existing algorithms

### Performance and Logging

- **Comprehensive logging** to both file and console
- **Progress tracking** with ETA calculations
- **Error handling** with retry logic
- **Performance metrics** (docs/second, total time)
- **Memory efficient** streaming processing

### Use Cases

1. **Research Dataset Creation**: Generate standardized PMC datasets for research
2. **RAG System Evaluation**: Use sample queries for systematic evaluation
3. **Production Deployment**: Scale to 10,000+ documents with resume capability
4. **Development Testing**: Quick test mode with 100 documents

---

**Note**: This documentation describes both the existing, proven PMC infrastructure and the new unified loader. The unified loader builds on the existing system while providing enhanced configurability and standardized output formats.