# Biomedical RAG Pipeline Evaluation Framework

A comprehensive, statistically rigorous evaluation system for comparing biomedical RAG (Retrieval-Augmented Generation) pipelines with empirical evidence and actionable insights.

## Overview

This framework provides:

1. **Biomedical Question Generation** - Domain-specific question generation from PMC documents
2. **RAGAS Metrics Framework** - Comprehensive evaluation using adapted RAGAS metrics  
3. **Statistical Analysis** - Rigorous statistical testing with power analysis and effect sizes
4. **Comparative Analysis** - Multi-pipeline comparison with statistical significance testing
5. **PMC Data Pipeline** - Scalable processing of 10K+ biomedical documents
6. **Evaluation Orchestrator** - End-to-end reproducible evaluation workflows
7. **Empirical Reporting** - Publication-ready reports with actionable recommendations

## Framework Components

### Core Modules

- `biomedical_question_generator.py` - Generate domain-specific evaluation questions
- `ragas_metrics_framework.py` - RAGAS metrics adapted for biomedical domain
- `statistical_evaluation_methodology.py` - Statistical analysis with power calculations
- `comparative_analysis_system.py` - Multi-pipeline comparison orchestration
- `pmc_data_pipeline.py` - Large-scale PMC document processing
- `evaluation_orchestrator.py` - End-to-end evaluation coordination
- `empirical_reporting.py` - Comprehensive report generation

### Key Features

- **Statistical Rigor**: Power analysis, effect size calculations, multiple comparison corrections
- **Biomedical Optimization**: ScispaCy, BioBERT, medical entity recognition
- **Scalability**: Handles 10K+ documents with parallel processing
- **Reproducibility**: Checkpointing, configuration management, random seed control
- **Comprehensive Reporting**: Executive summaries, technical reports, academic papers

## Installation

The framework integrates with the existing uv-based dependency management:

```bash
# Install evaluation dependencies
uv pip install -r evaluation_framework/requirements.txt

# Verify installation
uv run python -c "from evaluation_framework import evaluation_orchestrator; print('✓ Framework ready')"
```

## Quick Start

### 1. Demo Evaluation (2 pipelines, 20 questions)

```bash
uv run python -m evaluation_framework.evaluation_orchestrator \
    --experiment-name "demo_biomedical_rag" \
    --max-documents 50 \
    --max-questions 20 \
    --pipelines BasicRAG GraphRAG \
    --output-dir "outputs/evaluations"
```

### 2. Quick Evaluation (all pipelines, 100 questions)

```bash
uv run python -m evaluation_framework.evaluation_orchestrator \
    --experiment-name "quick_biomedical_rag" \
    --max-documents 500 \
    --max-questions 100 \
    --pipelines BasicRAG CRAG GraphRAG BasicRAGReranking \
    --output-dir "outputs/evaluations"
```

### 3. Full Evaluation (10K+ documents, 1000 questions)

```bash
uv run python -m evaluation_framework.evaluation_orchestrator \
    --experiment-name "full_biomedical_rag" \
    --max-documents 10000 \
    --max-questions 1000 \
    --pipelines BasicRAG CRAG GraphRAG BasicRAGReranking \
    --output-dir "outputs/evaluations"
```

## Configuration

### Evaluation Configuration

Create a YAML configuration file for reproducible experiments:

```yaml
# evaluation_config.yaml
experiment_name: "production_biomedical_rag_evaluation"
experiment_description: "Production-ready biomedical RAG evaluation"

# Data configuration
max_documents: 5000
max_questions: 500
test_questions_per_pipeline: 100

# Pipeline configuration
pipelines_to_evaluate:
  - BasicRAG
  - CRAG
  - GraphRAG
  - BasicRAGReranking

# Quality thresholds
min_document_quality: 0.3
min_question_quality: 0.5
statistical_significance_threshold: 0.05

# Processing configuration
max_workers: 8
batch_size: 32
enable_caching: true
enable_checkpointing: true

# Output configuration
output_base_dir: "outputs/evaluations"
generate_visualizations: true
generate_interactive_dashboard: true
generate_final_report: true

# Reproducibility
random_seed: 42
```

### Usage with Configuration

```bash
uv run python -m evaluation_framework.evaluation_orchestrator \
    --config evaluation_config.yaml
```

## Component Usage

### 1. Biomedical Question Generation

```python
from evaluation_framework.biomedical_question_generator import (
    create_biomedical_question_generator, 
    QuestionGenerationConfig
)

config = QuestionGenerationConfig(
    total_questions=200,
    min_confidence_score=0.7
)

generator = create_biomedical_question_generator(config)
questions = generator.generate_questions_from_documents(documents)
```

### 2. RAGAS Metrics Evaluation

```python
from evaluation_framework.ragas_metrics_framework import create_biomedical_ragas_framework

framework = create_biomedical_ragas_framework()
results = framework.evaluate_pipeline(responses, ground_truth, "pipeline_name")
```

### 3. Statistical Analysis

```python
from evaluation_framework.statistical_evaluation_methodology import create_statistical_framework

stats_framework = create_statistical_framework()
power_result = stats_framework.conduct_power_analysis(metric_data, "faithfulness")
comparison_results = stats_framework.compare_pipelines_statistical(metric_data, "faithfulness")
```

### 4. Report Generation

```python
from evaluation_framework.empirical_reporting import create_empirical_reporting_framework

reporting = create_empirical_reporting_framework()
reports = reporting.generate_comprehensive_report(
    evaluation_results, 
    experiment_config, 
    "evaluation_name"
)
```

## Output Structure

```
outputs/evaluations/
├── biomedical_rag_evaluation_20231201_100000/
│   ├── config.yaml                    # Experiment configuration
│   ├── evaluation.log                 # Execution logs
│   ├── results.json                   # Complete results
│   ├── evaluation_questions.json      # Generated questions
│   ├── comparative_analysis/          # Analysis outputs
│   │   ├── ragas_radar_chart.html
│   │   ├── statistical_significance_heatmap.png
│   │   └── performance_distributions.png
│   └── reports/                       # Generated reports
│       ├── executive_summary.html
│       ├── technical_report.html
│       ├── academic_paper.md
│       └── interactive_dashboard.html
```

## Report Types

### 1. Executive Summary
- Key findings and recommendations
- Strategic implications
- 5-minute reading time
- Decision-maker focused

### 2. Technical Report
- Comprehensive methodology
- Statistical analysis details
- Implementation guidance
- Developer/researcher focused

### 3. Academic Paper
- Peer-review ready format
- Complete methodology section
- Statistical evidence presentation
- Publication ready

### 4. Interactive Dashboard
- Real-time exploration
- Performance comparisons
- Statistical evidence visualization
- Stakeholder presentations

## Advanced Usage

### Resume Evaluation

```bash
uv run python -m evaluation_framework.evaluation_orchestrator \
    --resume biomedical_rag_evaluation_20231201_100000
```

### Partial Evaluation

```bash
uv run python -m evaluation_framework.evaluation_orchestrator \
    --partial data_preparation question_generation pipeline_evaluation
```

### Custom Pipeline Integration

The framework automatically detects and evaluates any pipeline that implements the standard interface:

```python
class CustomRAGPipeline:
    def query(self, question: str, top_k: int = 5, generate_answer: bool = True) -> Dict[str, Any]:
        # Implementation
        return {
            'question': question,
            'retrieved_contexts': contexts,
            'answer': generated_answer,
            'metadata': {...}
        }
```

## Statistical Methodology

### Power Analysis
- Target power ≥ 0.8
- Effect size sensitivity analysis
- Sample size validation

### Multiple Comparisons
- Benjamini-Hochberg FDR correction
- Family-wise error rate control
- Adjusted p-values reported

### Effect Sizes
- Cohen's d for practical significance
- Confidence intervals for all estimates
- Bootstrap resampling for robust estimates

### Evidence Levels
- **Strong**: p < 0.001, |d| > 0.8, power > 0.8
- **Moderate**: p < 0.01, |d| > 0.5, power > 0.7  
- **Weak**: p < 0.05, |d| > 0.2
- **Insufficient**: p ≥ 0.05 or small effect size

## Validation and Quality Assurance

### Data Quality
- Document validation with biomedical relevance scoring
- Question quality assessment with confidence thresholds
- Statistical assumption checking with diagnostic tests

### Reproducibility
- Configuration-driven experiments
- Random seed control across all components
- Checkpointing for long-running evaluations
- Complete audit trail with timestamped logs

### Performance Monitoring
- Execution time tracking per component
- Memory usage monitoring for large datasets
- Progress reporting with estimated completion times
- Error handling with graceful degradation

## Integration with Existing Infrastructure

The framework integrates seamlessly with the existing RAG infrastructure:

- **Pipeline Registry**: Automatically discovers available pipelines
- **IRIS Database**: Uses existing document storage and vector indexing
- **Configuration Management**: Leverages existing config infrastructure
- **Logging**: Integrates with project logging standards

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Documents**
   ```bash
   # Reduce batch size
   --batch-size 16
   ```

2. **Long Evaluation Times**
   ```bash
   # Use demo mode for testing
   --max-documents 100 --max-questions 50
   ```

3. **Pipeline Import Errors**
   ```bash
   # Verify pipeline availability
   uv run python -c "from iris_rag.pipelines import registry; print(registry.list_pipeline_names())"
   ```

4. **Statistical Analysis Failures**
   ```bash
   # Check sample sizes
   --max-questions 100  # Minimum for robust statistics
   ```

### Performance Optimization

- Use caching for repeated evaluations: `--enable-caching`
- Parallel processing: `--max-workers 8`
- Progressive evaluation: Start with demo, scale to full
- Checkpoint usage: `--enable-checkpointing` for long runs

## Contributing

When extending the framework:

1. Follow existing patterns for configuration and logging
2. Implement comprehensive error handling
3. Add statistical validation for new metrics
4. Include reproducibility measures (random seeds, versioning)
5. Provide clear documentation and examples

## License

This evaluation framework is part of the RAG Templates project and follows the same licensing terms.