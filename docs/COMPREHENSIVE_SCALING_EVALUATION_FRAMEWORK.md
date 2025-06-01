# Comprehensive Scaling and Evaluation Framework

## Overview

This framework provides a systematic approach to testing all **7 RAG techniques** across increasing dataset sizes with comprehensive **RAGAS metrics** evaluation. It addresses the need to understand how RAG techniques perform and scale with real-world data volumes.

## 🎯 Objectives

1. **Methodical Dataset Scaling**: Test techniques at 1K, 2.5K, 5K, 10K, 25K, and 50K documents
2. **Comprehensive Quality Assessment**: Use all RAGAS metrics for objective evaluation
3. **Performance Benchmarking**: Track response times, memory usage, and scalability
4. **Technique Comparison**: Provide data-driven recommendations for technique selection
5. **Production Readiness**: Validate techniques at enterprise scale

## 🔬 RAG Techniques Evaluated

| Technique | Description | Key Characteristics |
|-----------|-------------|-------------------|
| **BasicRAG** | Traditional vector similarity search | Reliable baseline, consistent performance |
| **HyDE** | Hypothetical Document Embeddings | Quality-focused, generates hypothetical documents |
| **CRAG** | Corrective Retrieval Augmented Generation | Enhanced coverage with corrective mechanisms |
| **ColBERT** | Contextualized Late Interaction over BERT | Token-level semantic matching, advanced retrieval |
| **NodeRAG** | Node-based retrieval with maximum coverage | Comprehensive document retrieval specialist |
| **GraphRAG** | Graph-based knowledge retrieval | Ultra-fast performance, structured knowledge |
| **HybridIFindRAG** | Multi-modal fusion approach | Complex analysis, multiple retrieval strategies |

## 📊 Evaluation Metrics

### RAGAS Quality Metrics
- **Answer Relevancy**: How relevant the answer is to the question
- **Context Precision**: Precision of retrieved context
- **Context Recall**: Recall of retrieved context  
- **Faithfulness**: How faithful the answer is to the context
- **Answer Similarity**: Similarity to ground truth answers
- **Answer Correctness**: Correctness of the generated answer
- **Context Relevancy**: Relevance of retrieved context

### Performance Metrics
- **Response Time**: End-to-end query processing time
- **Documents Retrieved**: Number of documents returned
- **Similarity Scores**: Average similarity scores of retrieved documents
- **Answer Length**: Length of generated answers
- **Memory Usage**: System memory consumption
- **Success Rate**: Percentage of successful query completions

## 🏗️ Framework Architecture

### Core Components

1. **[`ScalingEvaluationFramework`](../eval/scaling_evaluation_framework.py)**
   - Comprehensive evaluation at specific dataset sizes
   - RAGAS metrics integration
   - Performance monitoring
   - System resource tracking

2. **[`AutomatedDatasetScaling`](../scripts/automated_dataset_scaling.py)**
   - Systematic dataset size scaling
   - Ingestion performance monitoring
   - Data integrity validation
   - Database metrics tracking

3. **[`ComprehensiveScalingOrchestrator`](../eval/comprehensive_scaling_orchestrator.py)**
   - Coordinates scaling and evaluation
   - Generates visualizations
   - Creates comprehensive reports
   - Manages the complete pipeline

4. **[`run_comprehensive_scaling_evaluation.py`](../scripts/adhoc_utils/run_comprehensive_scaling_evaluation.py)**
   - Main execution script
   - Prerequisite checking
   - Multiple execution modes
   - Results management

### Execution Modes

#### 1. Current Size Mode (`--mode current_size`)
- Evaluates all techniques at current database size
- Quick assessment without scaling
- Ideal for initial testing

#### 2. Comprehensive Mode (`--mode comprehensive`)
- Full scaling and evaluation pipeline
- Tests at multiple dataset sizes
- Complete RAGAS evaluation
- Most thorough assessment

#### 3. Scaling Only Mode (`--mode scaling_only`)
- Dataset scaling without evaluation
- Performance monitoring during ingestion
- Data integrity validation
- Preparation for later evaluation

## 🚀 Usage Instructions

### Prerequisites

1. **Database Setup**: IRIS database with RAG schema
2. **Document Data**: PMC articles loaded and processed
3. **Dependencies**: Install required packages
   ```bash
   pip install ragas datasets matplotlib seaborn plotly psutil
   ```
4. **API Keys**: OpenAI API key for real RAGAS evaluation (optional)

### Quick Start

```bash
# Check current status and run evaluation at current size
python ../scripts/adhoc_utils/run_comprehensive_scaling_evaluation.py --mode current_size

# Run comprehensive scaling and evaluation (recommended)
python ../scripts/adhoc_utils/run_comprehensive_scaling_evaluation.py --mode comprehensive

# Scale dataset only (for preparation)
python ../scripts/adhoc_utils/run_comprehensive_scaling_evaluation.py --mode scaling_only
```

### Advanced Usage

```bash
# Skip prerequisite checks
python ../scripts/adhoc_utils/run_comprehensive_scaling_evaluation.py --mode current_size --skip-checks

# Run with specific configuration
export OPENAI_API_KEY="your-api-key"
python ../scripts/adhoc_utils/run_comprehensive_scaling_evaluation.py --mode comprehensive
```

## 📈 Dataset Scaling Strategy

### Target Sizes
- **1,000 documents**: Baseline testing
- **2,500 documents**: Small-scale validation
- **5,000 documents**: Medium-scale assessment
- **10,000 documents**: Large-scale testing
- **25,000 documents**: Enterprise-scale validation
- **50,000 documents**: Maximum scale testing

### Scaling Process
1. **Current Size Assessment**: Determine starting point
2. **Incremental Scaling**: Add documents to reach target size
3. **Performance Monitoring**: Track ingestion metrics
4. **Data Integrity**: Validate data consistency
5. **Evaluation Execution**: Run comprehensive assessment

## 📊 Output and Results

### Generated Files

#### JSON Results
- `comprehensive_scaling_pipeline_YYYYMMDD_HHMMSS.json`: Complete results
- `pipeline_intermediate_SIZE_YYYYMMDD_HHMMSS.json`: Intermediate results
- `scaling_evaluation_results_YYYYMMDD_HHMMSS.json`: Evaluation-only results

#### Reports
- `comprehensive_scaling_report_YYYYMMDD_HHMMSS.md`: Executive summary
- `scaling_evaluation_report_YYYYMMDD_HHMMSS.md`: Detailed analysis

#### Visualizations
- `performance_scaling_analysis_YYYYMMDD_HHMMSS.png`: Performance trends
- `quality_scaling_analysis_YYYYMMDD_HHMMSS.png`: Quality trends
- `scaling_dashboard_YYYYMMDD_HHMMSS.png`: Comprehensive dashboard

#### Logs
- `scaling_evaluation_YYYYMMDD_HHMMSS.log`: Detailed execution log

### Result Structure

```json
{
  "evaluation_plan": {
    "dataset_sizes": [1000, 2500, 5000, 10000, 25000, 50000],
    "techniques": ["BasicRAG", "HyDE", "CRAG", "ColBERT", "NodeRAG", "GraphRAG", "HybridIFindRAG"],
    "ragas_metrics": ["answer_relevancy", "context_precision", ...],
    "performance_metrics": ["response_time", "documents_retrieved", ...]
  },
  "results_by_size": {
    "1000": {
      "database_stats": {...},
      "techniques": {
        "BasicRAG": {
          "success_rate": 1.0,
          "avg_response_time": 7.95,
          "ragas_scores": {...},
          "individual_results": [...]
        }
      }
    }
  }
}
```

## 📋 Evaluation Protocol

### Standardized Test Queries
- **10 medical/scientific queries** covering different domains
- **Ground truth answers** for quality assessment
- **Keyword categories** for analysis
- **Consistent across all techniques** for fair comparison

### Quality Assessment Process
1. **Query Execution**: Run each query against each technique
2. **Context Extraction**: Extract retrieved documents and contexts
3. **RAGAS Evaluation**: Calculate all quality metrics
4. **Performance Measurement**: Track response times and resource usage
5. **Aggregation**: Compute technique-level statistics

### Performance Monitoring
- **System Metrics**: CPU, memory, disk usage
- **Database Metrics**: Query performance, index usage
- **Application Metrics**: Response times, success rates
- **Resource Tracking**: Memory deltas, processing overhead

## 🎯 Use Cases and Applications

### Research and Development
- **Technique Comparison**: Objective assessment of RAG approaches
- **Scalability Analysis**: Understanding performance at scale
- **Quality Benchmarking**: RAGAS-based quality assessment
- **Optimization Guidance**: Data-driven improvement recommendations

### Production Deployment
- **Technique Selection**: Choose optimal approach for use case
- **Capacity Planning**: Understand resource requirements
- **Performance Expectations**: Set realistic SLAs
- **Monitoring Strategy**: Establish quality and performance baselines

### Academic Research
- **Reproducible Benchmarks**: Standardized evaluation methodology
- **Comparative Studies**: Fair comparison across techniques
- **Scalability Research**: Understanding scaling characteristics
- **Quality Assessment**: Objective quality measurement

## 🔧 Customization and Extension

### Adding New Techniques
1. **Implement Pipeline**: Create technique-specific pipeline class
2. **Update Framework**: Add to technique list in evaluation framework
3. **Test Integration**: Verify compatibility with evaluation protocol
4. **Document Characteristics**: Add to technique comparison table

### Custom Metrics
1. **Extend RAGAS**: Add custom quality metrics
2. **Performance Metrics**: Add domain-specific measurements
3. **Visualization**: Update charts and dashboards
4. **Reporting**: Include in generated reports

### Dataset Customization
1. **Target Sizes**: Modify scaling strategy
2. **Test Queries**: Add domain-specific questions
3. **Ground Truth**: Provide custom reference answers
4. **Data Sources**: Integrate different document types

## 🚨 Troubleshooting

### Common Issues

#### Database Connection
```bash
# Check database status
python -c "from common.iris_connector_jdbc import get_iris_connection; print('✅ Connected')"
```

#### Missing Dependencies
```bash
# Install RAGAS
pip install ragas datasets

# Install visualization libraries
pip install matplotlib seaborn plotly
```

#### Memory Issues
- **Reduce batch sizes** in evaluation framework
- **Limit concurrent evaluations**
- **Monitor system resources** during execution

#### Performance Issues
- **Check database indexes** for optimal query performance
- **Verify HNSW indexes** are properly configured
- **Monitor disk I/O** during large-scale operations

### Debug Mode
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
python ../scripts/adhoc_utils/run_comprehensive_scaling_evaluation.py --mode current_size
```

## 📚 References and Related Work

### RAGAS Framework
- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS Metrics Guide](https://docs.ragas.io/en/stable/concepts/metrics/)

### RAG Techniques
- **HyDE**: [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
- **CRAG**: [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)
- **ColBERT**: [Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)

### Evaluation Methodologies
- [Evaluating Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2309.15217)
- [A Comprehensive Survey on RAG Evaluation](https://arxiv.org/abs/2401.05826)

## 🤝 Contributing

### Adding Techniques
1. **Fork repository** and create feature branch
2. **Implement technique** following existing patterns
3. **Add tests** and documentation
4. **Submit pull request** with evaluation results

### Improving Framework
1. **Identify enhancement opportunities**
2. **Implement improvements** with backward compatibility
3. **Test thoroughly** across all techniques
4. **Document changes** and update examples

### Reporting Issues
1. **Check existing issues** for duplicates
2. **Provide detailed reproduction steps**
3. **Include system information** and logs
4. **Suggest potential solutions** if possible

---

**Framework Version**: 1.0  
**Last Updated**: May 30, 2025  
**Compatibility**: All 7 RAG techniques, IRIS database, RAGAS metrics