# Test Database Volumes

Mountable IRIS database volumes pre-populated with different testing scenarios. This enables rapid context switching between testing environments while maintaining constitutional compliance with live database requirements.

## Available Test Databases

### `basic-rag-testdb/`
Pre-populated with standard medical documents for basic RAG pipeline testing.

**Contents:**
- 50+ medical documents (diabetes, COVID-19, cancer, heart disease)
- Vector embeddings for OpenAI text-embedding-3-small
- Basic RAG schema with DocumentChunks table
- Sample entities for basic knowledge graph testing

**Use Cases:**
- Basic RAG pipeline validation
- Vector similarity search testing
- Document retrieval functionality
- Performance baseline establishment

### `graphrag-testdb/`
Enhanced database with knowledge graph entities and relationships.

**Contents:**
- 100+ medical documents with complex relationships
- Entity extraction results (diseases, treatments, symptoms)
- Knowledge graph with 500+ entities and relationships
- GraphRAG schema with Entities and EntityRelationships tables
- Community detection results for graph clustering

**Use Cases:**
- GraphRAG pipeline testing
- Knowledge graph traversal validation
- Entity relationship queries
- Graph-based reasoning testing

### `crag-testdb/`
Database optimized for Corrective RAG testing with relevance examples.

**Contents:**
- High-quality and low-quality document pairs
- Relevance evaluation examples
- Documents with known answer patterns
- Quality scoring baselines
- Corrective retrieval test cases

**Use Cases:**
- CRAG pipeline validation
- Relevance evaluation testing
- Quality correction workflows
- Answer refinement testing

### `enterprise-testdb/`
Large-scale database for enterprise testing scenarios.

**Contents:**
- 10,000+ documents from multiple domains
- Complex entity relationships (2000+ entities)
- Multi-hop reasoning examples
- Performance testing baselines
- Concurrent access patterns

**Use Cases:**
- Enterprise scale testing
- Performance benchmarking
- Concurrent access validation
- Large-scale retrieval testing

## Usage Patterns

### Quick Test Database Switching

```bash
# Switch to basic RAG testing environment
make test-db-basic

# Switch to GraphRAG testing environment
make test-db-graphrag

# Switch to enterprise scale testing
make test-db-enterprise

# Create fresh empty database
make test-db-clean
```

### Docker Compose Integration

```yaml
# docker-compose.test.yml
services:
  iris-test:
    image: docker.iscinternal.com/intersystems/iris:2025.3.0EHAT.127.0-linux-arm64v8
    volumes:
      - ./docker/test-databases/basic-rag-testdb:/opt/irisapp/mgr/user:rw
    ports:
      - "31972:1972"
      - "352773:52773"
    environment:
      - ISC_DATA_DIRECTORY=/opt/irisapp/mgr/user
```

### Environment-Specific Testing

```bash
# Test with specific database context
IRIS_TEST_DB=graphrag make test-examples-basic

# Override default test database
TEST_DATABASE_VOLUME=./docker/test-databases/crag-testdb make test-examples-advanced
```

## Database Volume Structure

Each test database volume follows this structure:

```
testdb-name/
├── IRIS.DAT              # Main database file
├── IRIS.WIJ              # Write image journal
├── iris.key              # License key (if needed)
├── schema/
│   ├── tables.sql        # Schema definition
│   ├── indexes.sql       # Index creation
│   └── procedures.sql    # Stored procedures
├── data/
│   ├── documents/        # Source documents
│   ├── embeddings/       # Pre-computed embeddings
│   └── entities/         # Knowledge graph data
├── metadata/
│   ├── statistics.json   # Database statistics
│   ├── performance.json  # Performance baselines
│   └── manifest.json     # Database manifest
└── README.md            # Database-specific documentation
```

## Creating New Test Databases

### 1. Bootstrap Empty Database

```bash
# Create new test database volume
make create-test-db NAME=my-testdb

# Start with empty IRIS instance
docker-compose -f docker-compose.test.yml up -d iris-test

# Initialize schema
python scripts/test-db/initialize_schema.py --database my-testdb
```

### 2. Populate with Test Data

```bash
# Load documents and generate embeddings
python scripts/test-db/populate_database.py \
    --database my-testdb \
    --documents ./test-data/documents/ \
    --generate-embeddings

# Extract entities and relationships (for GraphRAG)
python scripts/test-db/extract_entities.py \
    --database my-testdb \
    --llm-provider openai
```

### 3. Create Database Snapshot

```bash
# Stop IRIS and create volume snapshot
docker-compose -f docker-compose.test.yml stop iris-test

# Copy database files to persistent volume
cp -r /var/lib/docker/volumes/test-iris-data/_data ./docker/test-databases/my-testdb/

# Document the database
python scripts/test-db/generate_manifest.py --database my-testdb
```

## Performance and Optimization

### Database Size Management

Test databases are optimized for size and loading speed:

- **Basic RAG**: ~100MB (loads in 5-10 seconds)
- **GraphRAG**: ~500MB (loads in 15-30 seconds)
- **CRAG**: ~200MB (loads in 10-15 seconds)
- **Enterprise**: ~2GB (loads in 1-2 minutes)

### Memory Configuration

Each test database includes optimized IRIS memory settings:

```ini
# iris.cpf (database-specific)
[ConfigFile]
globals=128
routines=64
gmheap=67108864
locksiz=16777216
```

### Indexing Strategy

Pre-built indexes for common query patterns:

- Vector similarity indexes (IRIS Vector Search)
- Full-text search indexes (Entity names, Document content)
- Foreign key indexes (Document → Chunks, Entities → Relationships)
- Composite indexes (Optimized for RAG query patterns)

## CI/CD Integration

### Automated Test Database Selection

The testing framework automatically selects appropriate test databases:

```python
def select_test_database(pipeline_type: str, test_category: str) -> str:
    """Select optimal test database for testing scenario."""
    database_map = {
        ("basic", "basic"): "basic-rag-testdb",
        ("graphrag", "advanced"): "graphrag-testdb",
        ("crag", "advanced"): "crag-testdb",
        ("enterprise", "scale"): "enterprise-testdb"
    }
    return database_map.get((pipeline_type, test_category), "basic-rag-testdb")
```

### GitHub Actions Integration

```yaml
# .github/workflows/ci.yml
- name: Setup test database
  run: |
    export TEST_DB=$(python scripts/test-db/select_database.py --pipeline ${{ matrix.pipeline }})
    docker-compose -f docker-compose.test.yml up -d iris-test
    scripts/test-db/wait_for_ready.sh

- name: Run pipeline tests
  run: |
    make test-examples-${{ matrix.category }}
  env:
    IRIS_HOST: localhost
    IRIS_PORT: 31972
```

## Security and Data Management

### Data Privacy

Test databases contain only synthetic or public domain content:

- Medical information from public health sources
- Synthetic patient data (no real PHI)
- Academic research abstracts
- Public domain educational content

### Version Control

Test databases are version controlled separately:

- **Git LFS**: For smaller test databases (<100MB)
- **External Storage**: For enterprise databases (>1GB)
- **Checksums**: SHA256 verification for data integrity
- **Manifests**: Metadata tracking for reproducibility

### Cleanup and Maintenance

```bash
# Clean up test databases
make clean-test-dbs

# Update test database from source
make update-test-db NAME=basic-rag-testdb

# Verify test database integrity
make verify-test-db NAME=graphrag-testdb
```

## Troubleshooting

### Common Issues

**Database Won't Mount**
```bash
# Check file permissions
sudo chown -R 51773:51773 ./docker/test-databases/my-testdb/
chmod -R 755 ./docker/test-databases/my-testdb/
```

**Schema Mismatch**
```bash
# Verify schema version
python scripts/test-db/check_schema.py --database my-testdb
```

**Performance Issues**
```bash
# Check database statistics
python scripts/test-db/analyze_performance.py --database my-testdb
```

### Validation Scripts

```bash
# Validate database consistency
python scripts/test-db/validate_database.py --database my-testdb

# Check test data quality
python scripts/test-db/verify_test_data.py --database my-testdb

# Performance benchmarking
python scripts/test-db/benchmark_database.py --database my-testdb
```

This mountable volume approach provides the perfect balance between constitutional compliance (live IRIS database) and testing efficiency (pre-populated, consistent test scenarios).