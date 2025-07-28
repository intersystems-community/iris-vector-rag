# Sample Data Manager Specification

## 1. Overview

The Sample Data Manager is a core component of the Quick Start system responsible for automated management of sample PMC documents. It provides a clean abstraction layer for downloading, validating, storing, and ingesting sample datasets optimized for quick start scenarios.

## 2. Architecture

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sample Data Manager                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Data Source    │  │  Download       │  │  Validation     │  │
│  │  Registry       │  │  Orchestrator   │  │  Engine         │  │
│  │                 │  │                 │  │                 │  │
│  │ • PMC API       │  │ • Parallel DL   │  │ • Schema Check  │  │
│  │ • Local Cache   │  │ • Progress      │  │ • Content Val   │  │
│  │ • Custom Sets   │  │ • Retry Logic   │  │ • Integrity     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│           │                     │                     │         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Storage        │  │  Ingestion      │  │  Health         │  │
│  │  Manager        │  │  Pipeline       │  │  Monitor        │  │
│  │                 │  │                 │  │                 │  │
│  │ • File System   │  │ • Batch Proc    │  │ • Status Track  │  │
│  │ • Compression   │  │ • Vector Gen    │  │ • Error Report  │  │
│  │ • Cleanup       │  │ • IRIS Insert   │  │ • Metrics       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Service Boundaries

Each component operates within well-defined boundaries:

- **Data Source Registry**: Manages available data sources and their configurations
- **Download Orchestrator**: Handles parallel downloads with progress tracking
- **Validation Engine**: Ensures data integrity and format compliance
- **Storage Manager**: Manages local file system operations
- **Ingestion Pipeline**: Processes documents into IRIS database
- **Health Monitor**: Tracks system health and performance metrics

## 3. Interface Specifications

### 3.1 Core Interfaces

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

class DataSourceType(Enum):
    PMC_API = "pmc_api"
    LOCAL_CACHE = "local_cache"
    CUSTOM_SET = "custom_set"

@dataclass
class SampleDataConfig:
    """Configuration for sample data operations."""
    source_type: DataSourceType
    document_count: int
    categories: List[str]
    storage_path: Path
    cache_enabled: bool = True
    parallel_downloads: int = 4
    batch_size: int = 10
    cleanup_on_success: bool = False
    iris_edition: str = "community"

@dataclass
class DocumentMetadata:
    """Metadata for a sample document."""
    pmc_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    file_size: int
    download_url: str
    local_path: Optional[Path] = None

@dataclass
class DownloadProgress:
    """Progress tracking for download operations."""
    total_documents: int
    downloaded: int
    failed: int
    current_document: Optional[str] = None
    bytes_downloaded: int = 0
    total_bytes: Optional[int] = None
    estimated_time_remaining: Optional[float] = None

@dataclass
class ValidationResult:
    """Result of document validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    document_count: int
    total_size: int

@dataclass
class IngestionResult:
    """Result of database ingestion."""
    success: bool
    documents_processed: int
    documents_ingested: int
    errors: List[str]
    processing_time: float
    database_size: int
```

### 3.2 Primary Interface

```python
class ISampleDataManager(ABC):
    """Primary interface for sample data management."""
    
    @abstractmethod
    async def download_samples(
        self, 
        config: SampleDataConfig,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None
    ) -> List[DocumentMetadata]:
        """
        Download sample documents according to configuration.
        
        Args:
            config: Download configuration
            progress_callback: Optional progress tracking callback
            
        Returns:
            List of downloaded document metadata
            
        Raises:
            DownloadError: If download fails
            ConfigurationError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    async def validate_samples(
        self, 
        storage_path: Path,
        strict_mode: bool = False
    ) -> ValidationResult:
        """
        Validate downloaded sample documents.
        
        Args:
            storage_path: Path to downloaded documents
            strict_mode: Enable strict validation rules
            
        Returns:
            Validation result with details
            
        Raises:
            ValidationError: If validation fails critically
        """
        pass
    
    @abstractmethod
    async def ingest_samples(
        self, 
        storage_path: Path,
        config: SampleDataConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> IngestionResult:
        """
        Ingest samples into IRIS database.
        
        Args:
            storage_path: Path to validated documents
            config: Ingestion configuration
            progress_callback: Optional progress tracking callback
            
        Returns:
            Ingestion result with statistics
            
        Raises:
            IngestionError: If ingestion fails
            DatabaseError: If database operations fail
        """
        pass
    
    @abstractmethod
    async def cleanup_samples(
        self, 
        storage_path: Path,
        keep_cache: bool = True
    ) -> None:
        """
        Clean up temporary sample files.
        
        Args:
            storage_path: Path to clean up
            keep_cache: Whether to preserve cache files
            
        Raises:
            CleanupError: If cleanup fails
        """
        pass
    
    @abstractmethod
    async def get_available_sources(self) -> List[Dict[str, Any]]:
        """
        Get list of available data sources.
        
        Returns:
            List of available data source configurations
        """
        pass
    
    @abstractmethod
    async def estimate_requirements(
        self, 
        config: SampleDataConfig
    ) -> Dict[str, Any]:
        """
        Estimate resource requirements for configuration.
        
        Args:
            config: Sample data configuration
            
        Returns:
            Dictionary with estimated disk space, memory, time requirements
        """
        pass
```

### 3.3 Data Source Interface

```python
class IDataSource(ABC):
    """Interface for data source implementations."""
    
    @abstractmethod
    async def list_available_documents(
        self, 
        categories: List[str],
        limit: Optional[int] = None
    ) -> List[DocumentMetadata]:
        """List available documents for download."""
        pass
    
    @abstractmethod
    async def download_document(
        self, 
        metadata: DocumentMetadata,
        storage_path: Path
    ) -> Path:
        """Download a single document."""
        pass
    
    @abstractmethod
    async def verify_document(
        self, 
        metadata: DocumentMetadata,
        local_path: Path
    ) -> bool:
        """Verify downloaded document integrity."""
        pass
```

## 4. Implementation Architecture

### 4.1 Class Structure

```python
# quick_start/data/sample_manager.py
class SampleDataManager(ISampleDataManager):
    """Main implementation of sample data management."""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.data_sources: Dict[DataSourceType, IDataSource] = {}
        self.download_orchestrator = DownloadOrchestrator()
        self.validation_engine = ValidationEngine()
        self.storage_manager = StorageManager()
        self.ingestion_pipeline = IngestionPipeline(config_manager)
        self.health_monitor = HealthMonitor()
        
        self._register_data_sources()
    
    def _register_data_sources(self):
        """Register available data sources."""
        self.data_sources[DataSourceType.PMC_API] = PMCAPIDataSource()
        self.data_sources[DataSourceType.LOCAL_CACHE] = LocalCacheDataSource()
        self.data_sources[DataSourceType.CUSTOM_SET] = CustomSetDataSource()

# quick_start/data/sources/pmc_api.py
class PMCAPIDataSource(IDataSource):
    """PMC API data source implementation."""
    
    def __init__(self):
        self.api_client = PMCAPIClient()
        self.rate_limiter = RateLimiter(requests_per_second=2)
    
    async def list_available_documents(
        self, 
        categories: List[str],
        limit: Optional[int] = None
    ) -> List[DocumentMetadata]:
        """Implementation for PMC API document listing."""
        pass

# quick_start/data/orchestrator.py
class DownloadOrchestrator:
    """Manages parallel downloads with progress tracking."""
    
    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.progress_tracker = ProgressTracker()
    
    async def download_batch(
        self,
        documents: List[DocumentMetadata],
        data_source: IDataSource,
        storage_path: Path,
        progress_callback: Optional[Callable] = None
    ) -> List[Path]:
        """Download documents in parallel batches."""
        pass
```

### 4.2 Configuration Schema

```yaml
# quick_start/config/templates/sample_data.yaml
sample_data:
  profiles:
    minimal:
      document_count: 10
      categories: ["medical"]
      parallel_downloads: 2
      batch_size: 5
    
    standard:
      document_count: 50
      categories: ["medical", "research"]
      parallel_downloads: 4
      batch_size: 10
    
    extended:
      document_count: 100
      categories: ["medical", "research", "clinical"]
      parallel_downloads: 6
      batch_size: 15
  
  sources:
    pmc_api:
      base_url: "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
      rate_limit: 2  # requests per second
      timeout: 30
      retry_attempts: 3
      retry_delay: 5
    
    local_cache:
      cache_directory: "data/sample_cache"
      max_cache_size: "1GB"
      cache_expiry: "7d"
  
  storage:
    base_path: "data/quick_start_samples"
    compression: true
    checksum_validation: true
    cleanup_policy: "retain_on_success"
  
  ingestion:
    batch_size: 10
    parallel_workers: 2
    embedding_model: "all-MiniLM-L6-v2"
    chunk_size: 512
    chunk_overlap: 50
  
  validation:
    strict_mode: false
    required_fields: ["title", "abstract", "content"]
    min_content_length: 100
    max_file_size: "10MB"
```

## 5. Error Handling

### 5.1 Exception Hierarchy

```python
class SampleDataError(Exception):
    """Base exception for sample data operations."""
    pass

class ConfigurationError(SampleDataError):
    """Configuration validation errors."""
    pass

class DownloadError(SampleDataError):
    """Download operation errors."""
    
    def __init__(self, message: str, failed_documents: List[str] = None):
        super().__init__(message)
        self.failed_documents = failed_documents or []

class ValidationError(SampleDataError):
    """Document validation errors."""
    
    def __init__(self, message: str, validation_details: Dict[str, Any] = None):
        super().__init__(message)
        self.validation_details = validation_details or {}

class IngestionError(SampleDataError):
    """Database ingestion errors."""
    
    def __init__(self, message: str, processed_count: int = 0):
        super().__init__(message)
        self.processed_count = processed_count

class StorageError(SampleDataError):
    """File system storage errors."""
    pass

class CleanupError(SampleDataError):
    """Cleanup operation errors."""
    pass
```

### 5.2 Error Recovery Strategies

```python
class ErrorRecoveryManager:
    """Manages error recovery strategies."""
    
    async def handle_download_failure(
        self, 
        error: DownloadError,
        config: SampleDataConfig
    ) -> bool:
        """
        Handle download failures with retry logic.
        
        Returns:
            True if recovery successful, False otherwise
        """
        if len(error.failed_documents) < config.document_count * 0.1:
            # Less than 10% failure rate - retry failed documents
            return await self._retry_failed_downloads(error.failed_documents)
        else:
            # High failure rate - switch to alternative source
            return await self._switch_data_source(config)
    
    async def handle_validation_failure(
        self, 
        error: ValidationError,
        storage_path: Path
    ) -> bool:
        """Handle validation failures with cleanup and re-download."""
        # Clean up invalid documents
        await self._cleanup_invalid_documents(error.validation_details)
        
        # Re-download if possible
        return await self._redownload_invalid_documents(storage_path)
    
    async def handle_ingestion_failure(
        self, 
        error: IngestionError,
        config: SampleDataConfig
    ) -> bool:
        """Handle ingestion failures with partial recovery."""
        if error.processed_count > 0:
            # Partial success - continue with processed documents
            logger.warning(f"Partial ingestion: {error.processed_count} documents processed")
            return True
        else:
            # Complete failure - reset and retry
            return await self._reset_and_retry_ingestion(config)
```

## 6. Performance Optimization

### 6.1 Caching Strategy

```python
class CacheManager:
    """Manages local caching for sample data."""
    
    def __init__(self, cache_dir: Path, max_size: int):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.cache_index = CacheIndex()
    
    async def get_cached_document(self, pmc_id: str) -> Optional[Path]:
        """Retrieve document from cache if available."""
        cache_entry = self.cache_index.get(pmc_id)
        if cache_entry and cache_entry.is_valid():
            return cache_entry.path
        return None
    
    async def cache_document(self, pmc_id: str, content: bytes) -> Path:
        """Cache document with compression and indexing."""
        # Ensure cache size limits
        await self._enforce_cache_limits()
        
        # Compress and store
        cache_path = self.cache_dir / f"{pmc_id}.xml.gz"
        with gzip.open(cache_path, 'wb') as f:
            f.write(content)
        
        # Update index
        self.cache_index.add(pmc_id, cache_path, len(content))
        
        return cache_path
```

### 6.2 Parallel Processing

```python
class ParallelProcessor:
    """Manages parallel processing for sample data operations."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_documents_parallel(
        self,
        documents: List[DocumentMetadata],
        processor_func: Callable,
        batch_size: int = 10
    ) -> List[Any]:
        """Process documents in parallel batches."""
        results = []
        
        for batch in self._create_batches(documents, batch_size):
            batch_tasks = [
                asyncio.get_event_loop().run_in_executor(
                    self.executor, processor_func, doc
                )
                for doc in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
        
        return results
```

## 7. Monitoring and Observability

### 7.1 Health Monitoring

```python
class HealthMonitor:
    """Monitors health of sample data operations."""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()
    
    async def check_download_health(self) -> Dict[str, Any]:
        """Check health of download operations."""
        return {
            "download_success_rate": self.metrics.get_success_rate("download"),
            "average_download_time": self.metrics.get_average_time("download"),
            "active_downloads": self.metrics.get_active_count("download"),
            "cache_hit_rate": self.metrics.get_cache_hit_rate(),
            "disk_usage": self.metrics.get_disk_usage(),
            "memory_usage": self.metrics.get_memory_usage()
        }
    
    async def check_ingestion_health(self) -> Dict[str, Any]:
        """Check health of ingestion operations."""
        return {
            "ingestion_success_rate": self.metrics.get_success_rate("ingestion"),
            "documents_per_second": self.metrics.get_throughput("ingestion"),
            "database_size": self.metrics.get_database_size(),
            "vector_index_health": self.metrics.get_index_health(),
            "connection_pool_status": self.metrics.get_connection_status()
        }
```

### 7.2 Performance Metrics

```python
@dataclass
class PerformanceMetrics:
    """Performance metrics for sample data operations."""
    
    # Download metrics
    total_download_time: float
    average_download_speed: float  # MB/s
    download_success_rate: float
    cache_hit_rate: float
    
    # Validation metrics
    validation_time: float
    validation_success_rate: float
    
    # Ingestion metrics
    ingestion_time: float
    documents_per_second: float
    vector_generation_time: float
    database_insert_time: float
    
    # Resource metrics
    peak_memory_usage: int  # bytes
    disk_usage: int  # bytes
    cpu_usage_percent: float
    
    # Quality metrics
    document_quality_score: float
    content_diversity_score: float
```

## 8. Testing Strategy

### 8.1 Unit Tests

```python
# tests/quick_start/test_sample_manager.py
class TestSampleDataManager:
    """Unit tests for SampleDataManager."""
    
    @pytest.fixture
    def sample_manager(self):
        config_manager = Mock(spec=ConfigurationManager)
        return SampleDataManager(config_manager)
    
    @pytest.mark.asyncio
    async def test_download_samples_success(self, sample_manager):
        """Test successful sample download."""
        config = SampleDataConfig(
            source_type=DataSourceType.PMC_API,
            document_count=5,
            categories=["medical"],
            storage_path=Path("/tmp/test_samples")
        )
        
        result = await sample_manager.download_samples(config)
        
        assert len(result) == 5
        assert all(doc.pmc_id for doc in result)
    
    @pytest.mark.asyncio
    async def test_download_samples_partial_failure(self, sample_manager):
        """Test download with partial failures."""
        # Test implementation
        pass
    
    @pytest.mark.asyncio
    async def test_validate_samples_success(self, sample_manager):
        """Test successful sample validation."""
        # Test implementation
        pass
```

### 8.2 Integration Tests

```python
# tests/quick_start/test_sample_integration.py
class TestSampleDataIntegration:
    """Integration tests for complete sample data workflow."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete download -> validate -> ingest workflow."""
        # Test implementation
        pass
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery mechanisms."""
        # Test implementation
        pass
```

## 9. Security Considerations

### 9.1 Data Security

- **Download Verification**: All downloads verified with checksums
- **Content Sanitization**: XML content sanitized before processing
- **Access Control**: File system permissions properly configured
- **Secure Cleanup**: Sensitive data securely deleted

### 9.2 Network Security

- **HTTPS Only**: All network communications over HTTPS
- **Rate Limiting**: Respect API rate limits and implement backoff
- **Timeout Handling**: Proper timeout handling for network operations
- **Certificate Validation**: SSL certificate validation enabled

This specification provides a comprehensive foundation for implementing the Sample Data Manager component with proper separation of concerns, error handling, and extensibility.