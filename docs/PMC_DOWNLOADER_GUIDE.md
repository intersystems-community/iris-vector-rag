# PMC Document Downloader System Guide

## Overview

The PMC Document Downloader System provides enterprise-scale downloading and loading of real medical/scientific documents from the PMC (PubMed Central) Open Access Subset. This system enables the RAG Templates project to obtain 10,000+ real documents for comprehensive testing and deployment.

## Features

- **Enterprise Scale**: Download 10,000+ real PMC documents
- **Batch Processing**: Efficient batch downloading with progress tracking
- **Document Validation**: Comprehensive validation of downloaded XML documents
- **Progress Monitoring**: Real-time progress tracking with ETA calculations
- **Checkpoint Support**: Resume interrupted downloads from checkpoints
- **Integration**: Seamless integration with UnifiedDocumentLoader
- **Command Line Interface**: Easy-to-use CLI for operations

## Architecture

### Core Components

1. **PMCAPIClient** (`data/pmc_downloader/api_client.py`)
   - Interface to PMC APIs and FTP services
   - Bulk file discovery and download
   - Document metadata retrieval
   - Rate limiting and error handling

2. **PMCBatchDownloader** (`data/pmc_downloader/batch_downloader.py`)
   - Batch downloading with progress tracking
   - Document validation and quality checks
   - Checkpoint save/restore functionality
   - Performance monitoring

3. **PMCEnterpriseLoader** (`data/pmc_downloader/integration.py`)
   - Integration with UnifiedDocumentLoader
   - End-to-end orchestration (download → validate → load)
   - Enterprise-scale configuration management

## Quick Start

### Command Line Usage

```bash
# Download 10,000 documents (default)
uv run python scripts/download_pmc_enterprise.py

# Download 50,000 documents to custom directory
uv run python scripts/download_pmc_enterprise.py --target 50000 --download-dir data/pmc_50k

# Resume interrupted download
uv run python scripts/download_pmc_enterprise.py --target 10000 --resume

# Download with custom settings
uv run python scripts/download_pmc_enterprise.py \
    --target 25000 \
    --batch-size 200 \
    --download-dir data/pmc_25k \
    --no-validation
```

### Programmatic Usage

```python
from data.pmc_downloader import load_enterprise_pmc_dataset

# Simple usage
result = load_enterprise_pmc_dataset(target_documents=10000)

# Advanced usage with custom configuration
def progress_callback(progress_info):
    print(f"Progress: {progress_info['phase_name']} - {progress_info['phase_progress']:.1f}%")

result = load_enterprise_pmc_dataset(
    target_documents=25000,
    progress_callback=progress_callback,
    config_overrides={
        'download_directory': 'data/pmc_custom',
        'batch_size': 150,
        'enable_validation': True
    }
)

if result['success']:
    print(f"Successfully loaded {result['final_document_count']} documents")
else:
    print(f"Failed: {result['error']}")
```

## Configuration

### Default Configuration

```python
{
    'target_document_count': 10000,
    'download_directory': 'data/pmc_enterprise',
    'enable_validation': True,
    'batch_size': 100,
    'use_checkpointing': True,
    'embedding_column_type': 'VECTOR',
    'downloader': {
        'batch_size': 100,
        'max_concurrent_downloads': 4,
        'checkpoint_interval': 500,
        'api_client': {
            'request_delay_seconds': 0.5,
            'max_retries': 3,
            'timeout_seconds': 30
        }
    },
    'loader': {
        'batch_size': 100,
        'token_batch_size': 1000,
        'embedding_column_type': 'VECTOR',
        'use_checkpointing': True,
        'refresh_connection': False,
        'gc_collect_interval': 50
    }
}
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `target_document_count` | Number of documents to download | 10000 |
| `download_directory` | Directory for downloads | `data/pmc_enterprise` |
| `enable_validation` | Validate downloaded documents | `True` |
| `batch_size` | Batch size for loading | 100 |
| `use_checkpointing` | Enable checkpoint support | `True` |
| `request_delay_seconds` | Delay between API requests | 0.5 |
| `max_retries` | Maximum retry attempts | 3 |
| `timeout_seconds` | Request timeout | 30 |

## Data Sources

### PMC Open Access Subset

The system downloads from the PMC Open Access Subset, which provides:

- **Real Medical Documents**: Peer-reviewed research papers
- **Structured XML**: Well-formatted XML with metadata
- **Bulk Downloads**: Efficient tar.gz archives
- **Regular Updates**: Fresh content from ongoing research

### Document Types

- Research articles
- Review papers
- Case studies
- Clinical trials
- Meta-analyses

## Validation Process

### Document Quality Checks

1. **File Integrity**: Verify file exists and has content
2. **XML Structure**: Parse and validate XML format
3. **Metadata Extraction**: Extract title, abstract, authors, keywords
4. **Content Quality**: Ensure sufficient abstract length (>50 chars)
5. **Medical Relevance**: Verify document contains medical/scientific content

### Validation Results

```python
ValidationResult(
    is_valid=True,
    file_path="PMC123456.xml",
    pmc_id="PMC123456",
    title="COVID-19 Vaccine Efficacy Study",
    abstract_length=245,
    file_size_bytes=15420,
    validation_time_seconds=0.12
)
```

## Progress Tracking

### Progress Information

```python
{
    'phase': 2,                           # Current phase (1=Download, 2=Validate, 3=Load)
    'phase_name': 'Load',                 # Human-readable phase name
    'phase_progress': 75.5,               # Progress within current phase (%)
    'overall_progress': 85.2,             # Overall progress (%)
    'current_operation': 'Loading batch 15/20',
    'documents_processed': 8520,          # Documents processed so far
    'documents_validated': 8450,          # Documents that passed validation
    'estimated_time_remaining': 180       # ETA in seconds
}
```

### Checkpoint System

- **Automatic Checkpoints**: Saved every 500 documents
- **Resume Capability**: Continue from last checkpoint
- **Progress Preservation**: Maintains download and validation state
- **Error Recovery**: Robust handling of interruptions

## Performance

### Typical Performance Metrics

| Scale | Download Time | Validation Time | Loading Time | Total Time |
|-------|---------------|-----------------|--------------|------------|
| 1,000 docs | 2-5 minutes | 30-60 seconds | 1-2 minutes | 4-8 minutes |
| 10,000 docs | 15-30 minutes | 3-5 minutes | 8-15 minutes | 25-50 minutes |
| 50,000 docs | 1-2 hours | 15-25 minutes | 40-75 minutes | 2-3.5 hours |

### Optimization Tips

1. **Batch Size**: Increase for faster loading (but more memory usage)
2. **Validation**: Disable for faster processing (but less reliability)
3. **Concurrent Downloads**: Increase for faster downloads (but more bandwidth)
4. **Disk Space**: Ensure sufficient space (estimate 100KB per document)

## Integration with RAG Templates

### Vector Store Integration

The system integrates seamlessly with the IRIS vector store:

```python
from iris_rag.storage.vector_store.iris_impl import IRISVectorStore
from iris_rag.config.manager import ConfigurationManager

# Documents are automatically loaded into vector store
config_manager = ConfigurationManager()
vector_store = IRISVectorStore(config_manager)

# Check document count
doc_count = vector_store.get_document_count()
print(f"Loaded {doc_count} documents")
```

### Pipeline Compatibility

Downloaded documents work with all RAG pipelines:

- BasicRAG
- CRAG (Corrective RAG)
- GraphRAG
- HybridIFind
- HyDE (Hypothetical Document Embeddings)
- NodeRAG
- ColBERT
- SQLRAG

## Testing

### Unit Tests

```bash
# Run PMC downloader tests
uv run pytest tests/test_pmc_downloader.py -v

# Run with integration tests (requires network)
uv run pytest tests/test_pmc_downloader.py::TestPMCDownloaderIntegration -v -s
```

### Enterprise Scale Tests

```bash
# Run 10K+ document enterprise test
uv run pytest tests/test_enterprise_10k_comprehensive.py -v -s
```

## Troubleshooting

### Common Issues

1. **Network Connectivity**
   - Ensure internet access to PMC FTP servers
   - Check firewall settings for FTP access
   - Verify DNS resolution for `ftp.ncbi.nlm.nih.gov`

2. **Disk Space**
   - Estimate 100KB per document for storage
   - Ensure 2x space for extraction (compressed + uncompressed)
   - Monitor disk usage during large downloads

3. **Memory Usage**
   - Reduce batch size if experiencing memory issues
   - Enable garbage collection intervals
   - Monitor system resources during processing

4. **IRIS Database**
   - Ensure IRIS Enterprise edition for large datasets
   - Verify database connection and credentials
   - Check available database storage

### Error Recovery

1. **Interrupted Downloads**
   ```bash
   # Resume from checkpoint
   uv run python scripts/download_pmc_enterprise.py --target 10000 --resume
   ```

2. **Validation Failures**
   ```bash
   # Skip validation for speed
   uv run python scripts/download_pmc_enterprise.py --target 10000 --no-validation
   ```

3. **Loading Errors**
   - Check IRIS database connectivity
   - Verify schema initialization
   - Review error logs in `pmc_enterprise_download.log`

### Log Files

- **Download Log**: `pmc_enterprise_download.log`
- **Checkpoint File**: `data/pmc_download_checkpoint.json`
- **Test Logs**: `test_output/test_pmc_downloader.log`

## API Reference

### PMCAPIClient

```python
client = PMCAPIClient(config)
files = client.get_available_bulk_files()
result = client.download_bulk_file(filename, target_dir)
extract_result = client.extract_bulk_file(archive_path, extract_dir)
```

### PMCBatchDownloader

```python
downloader = PMCBatchDownloader(config)
result = downloader.download_enterprise_dataset(progress_callback)
```

### PMCEnterpriseLoader

```python
loader = PMCEnterpriseLoader(config)
result = loader.load_enterprise_dataset(embedding_func, colbert_func, progress_callback)
```

## Best Practices

1. **Start Small**: Test with 1,000 documents before scaling up
2. **Monitor Progress**: Use progress callbacks for long-running operations
3. **Enable Checkpoints**: Always use checkpointing for large downloads
4. **Validate Documents**: Keep validation enabled for production use
5. **Plan Storage**: Ensure adequate disk space before starting
6. **Network Stability**: Use stable internet connection for large downloads
7. **Resource Monitoring**: Monitor CPU, memory, and disk usage

## Support

For issues and questions:

1. Check this documentation
2. Review log files for error details
3. Run tests to verify system functionality
4. Check network connectivity and system resources
5. Consult the RAG Templates project documentation

## License

This system is part of the RAG Templates project and follows the same licensing terms.