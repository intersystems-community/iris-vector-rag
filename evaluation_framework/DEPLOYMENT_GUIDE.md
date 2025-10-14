# Biomedical RAG Evaluation Framework - Deployment Guide

## Quick Start

### 1. Installation

```bash
# Clone repository and navigate to evaluation framework
cd /path/to/rag-templates/evaluation_framework

# Install dependencies 
uv pip install -r requirements.txt

# Verify installation
python simple_test.py
```

### 2. Basic Usage

```bash
# Demo evaluation (small scale)
python -m evaluation_orchestrator \
    --experiment-name "demo" \
    --max-documents 50 \
    --max-questions 20 \
    --output-dir "outputs/demo"

# Production evaluation (full scale)
python -m evaluation_orchestrator \
    --experiment-name "production_eval" \
    --max-documents 10000 \
    --max-questions 1000 \
    --output-dir "outputs/production"
```

## Dependencies and Models

### Required Models for Full Functionality

The framework requires several language models that must be downloaded:

1. **Question Generation Models**:
   ```bash
   # These will be downloaded automatically on first use
   - facebook/bart-large-cnn
   - microsoft/DialoGPT-medium
   ```

2. **Biomedical NLP Models**:
   ```bash
   # BioBERT for domain-specific processing
   - dmis-lab/biobert-base-cased-v1.1
   - dmis-lab/biobert-base-cased-v1.1-squad
   
   # ScispaCy models
   python -m spacy download en_core_sci_sm
   python -m spacy download en_ner_bc5cdr_md
   ```

3. **Embedding Models**:
   ```bash
   # Sentence transformers for similarity
   - sentence-transformers/all-MiniLM-L6-v2
   - sentence-transformers/all-mpnet-base-v2
   ```

### Installation Commands

```bash
# Install all biomedical models
python -c "
import transformers
import sentence_transformers
import spacy

# Download required models
transformers.AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
"

# Install spaCy models
python -m spacy download en_core_sci_sm
python -m spacy download en_ner_bc5cdr_md
```

## Configuration

### Environment Variables

Create a `.env` file in the evaluation_framework directory:

```bash
# Database connections
IRIS_CONNECTION_STRING="iris://user:pass@host:port/namespace"
IRIS_USERNAME="your_username"
IRIS_PASSWORD="your_password"

# Model configurations
HUGGINGFACE_TOKEN="your_hf_token"  # Optional, for private models
OPENAI_API_KEY="your_openai_key"   # Optional, for GPT-based metrics

# Evaluation settings
MAX_WORKERS=8
DEFAULT_BATCH_SIZE=32
ENABLE_GPU=true

# Logging
LOG_LEVEL=INFO
LOG_FILE="evaluation.log"
```

### Configuration File

Create `evaluation_config.yaml`:

```yaml
# Experiment configuration
experiment:
  name: "biomedical_rag_evaluation"
  description: "Comprehensive evaluation of biomedical RAG pipelines"
  version: "1.0.0"

# Data processing
data:
  max_documents: 5000
  max_questions: 500
  min_document_quality: 0.3
  min_question_quality: 0.5
  batch_size: 32

# Pipeline configuration
pipelines:
  - BasicRAG
  - CRAG  
  - GraphRAG
  - BasicRAGReranking

# Evaluation settings
evaluation:
  metrics:
    - faithfulness
    - answer_relevancy
    - context_precision
    - context_recall
    - context_utilization
    - answer_correctness
    - answer_similarity
  
  statistical_tests:
    significance_threshold: 0.05
    power_threshold: 0.8
    effect_size_threshold: 0.2
    
  quality_controls:
    enable_validation: true
    enable_checkpointing: true
    enable_caching: true

# Output configuration
output:
  base_dir: "outputs/evaluations"
  generate_visualizations: true
  generate_dashboard: true
  generate_reports: true
  
  formats:
    - json
    - html
    - markdown
    - pdf

# Performance settings
performance:
  max_workers: 8
  enable_parallel: true
  memory_limit_gb: 16
  timeout_minutes: 120

# Reproducibility
reproducibility:
  random_seed: 42
  save_intermediate: true
  version_control: true
```

## Integration with Existing Infrastructure

### IRIS Database Connection

The framework integrates with existing IRIS database infrastructure:

```python
# Example integration
from evaluation_framework.pmc_data_pipeline import create_pmc_data_pipeline, PMCProcessingConfig

config = PMCProcessingConfig(
    max_documents=1000,
    connection_string=os.getenv('IRIS_CONNECTION_STRING'),
    batch_size=50
)

pipeline = create_pmc_data_pipeline(config)
documents = pipeline.load_documents_from_iris()
```

### Pipeline Registration

The framework automatically discovers available RAG pipelines:

```python
# Pipelines should implement this interface
class CustomRAGPipeline:
    def query(self, question: str, top_k: int = 5, generate_answer: bool = True) -> Dict[str, Any]:
        return {
            'question': question,
            'retrieved_contexts': [...],
            'answer': "Generated answer",
            'metadata': {...}
        }
        
    def get_name(self) -> str:
        return "CustomRAG"
```

## Troubleshooting

### Common Issues

1. **Model Download Failures**
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/huggingface/
   python -c "import transformers; transformers.AutoModel.from_pretrained('model-name')"
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size in config
   batch_size: 16  # Instead of 32
   max_workers: 4  # Instead of 8
   ```

3. **IRIS Connection Issues**
   ```bash
   # Test connection
   python -c "
   import iris
   conn = iris.connect('iris://user:pass@host:port/namespace')
   print('Connection successful')
   "
   ```

4. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

### Performance Optimization

1. **Use GPU Acceleration** (if available):
   ```python
   # Set in environment
   export CUDA_VISIBLE_DEVICES=0
   ```

2. **Optimize Batch Sizes**:
   ```yaml
   # Start small and increase
   batch_size: 16  # For limited memory
   batch_size: 64  # For high-memory systems
   ```

3. **Enable Caching**:
   ```yaml
   enable_caching: true
   cache_dir: "/path/to/fast/storage"
   ```

### Monitoring and Logging

The framework provides comprehensive logging:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
```

## Production Deployment

### Recommended System Requirements

- **CPU**: 8+ cores
- **RAM**: 32+ GB 
- **Storage**: 100+ GB SSD
- **GPU**: Optional (NVIDIA with 8+ GB VRAM)
- **Python**: 3.9+

### Deployment Steps

1. **Environment Setup**:
   ```bash
   # Create dedicated environment
   python -m venv evaluation_env
   source evaluation_env/bin/activate
   
   # Install framework
   pip install -r requirements.txt
   ```

2. **Model Preparation**:
   ```bash
   # Pre-download models
   python scripts/download_models.py
   ```

3. **Configuration**:
   ```bash
   # Set production config
   cp evaluation_config.prod.yaml evaluation_config.yaml
   ```

4. **Validation**:
   ```bash
   # Run validation tests
   python simple_test.py
   python test_framework.py
   ```

5. **Monitoring Setup**:
   ```bash
   # Setup monitoring
   python scripts/setup_monitoring.py
   ```

### Production Considerations

- **Resource Management**: Monitor memory and CPU usage
- **Model Caching**: Pre-load models to reduce startup time
- **Error Handling**: Implement robust error recovery
- **Backup Strategy**: Regular backup of evaluation results
- **Security**: Secure API keys and database credentials

## Support and Maintenance

### Regular Maintenance Tasks

1. **Model Updates**:
   ```bash
   # Check for model updates monthly
   python scripts/check_model_updates.py
   ```

2. **Performance Monitoring**:
   ```bash
   # Generate performance reports
   python scripts/performance_report.py
   ```

3. **Data Cleanup**:
   ```bash
   # Clean old evaluation results
   python scripts/cleanup_old_results.py --days 30
   ```

### Getting Help

- **Framework Issues**: Check logs in `evaluation.log`
- **Model Issues**: Verify model downloads and GPU availability
- **Performance Issues**: Monitor system resources and adjust batch sizes
- **Integration Issues**: Verify IRIS database connectivity