# RAG Benchmarking Datasets and Reference Results

This document outlines key datasets and published results for benchmarking Retrieval-Augmented Generation (RAG) systems. These can be used to compare our project's performance against established benchmarks.

## MultiHopQA

[MultiHopQA](https://allenai.org/data/hotpotqa) (also known as HotPotQA) is a popular dataset for evaluating multi-hop question answering capabilities, requiring systems to retrieve and reason over multiple documents.

### Dataset Characteristics
- **Size**: 113k question-answer pairs (90k training, 7.4k dev, 7.4k test)
- **Type**: Multi-hop reasoning questions that require retrieving and connecting facts from multiple documents
- **Domains**: Wikipedia-based, covering diverse topics
- **Question Types**: Comparison questions, multi-bridge questions requiring 2+ evidence pieces

### Reference Results (F1 Scores)

| System | Answer F1 | Supporting Facts F1 | Joint F1 |
|--------|-----------|---------------------|----------|
| GRR (2021) | 81.16 | 90.51 | 74.91 |
| HGN (2020) | 82.22 | 88.58 | 74.21 |
| GraphRAG (2023) | 79.64 | 84.93 | 70.28 |
| ColBERT (2020) | 68.70 | 72.80 | 57.80 |
| Basic Dense Retrieval | 63.10 | 66.70 | 49.20 |

## BioASQ

[BioASQ](http://bioasq.org/) is a competition focusing on biomedical semantic indexing and question answering, highly relevant for our PMC-focused RAG system.

### Dataset Characteristics
- **Size**: Thousands of questions across multiple years of competitions
- **Type**: Biomedical questions requiring domain expertise
- **Question Categories**: Yes/No, Factoid, List, and Summary

### Reference Results (From BioASQ-10)

| System | Yes/No Accuracy | Factoid MRR | List F1 | Summary ROUGE-2 |
|--------|----------------|------------|---------|-----------------|
| SOTA (2022) | 87.2% | 0.564 | 0.479 | 0.497 |
| BM25 + T5 | 81.4% | 0.423 | 0.385 | 0.412 |
| ColBERT + T5 | 84.1% | 0.481 | 0.436 | 0.449 |

## MedMCQA

[MedMCQA](https://medmcqa.github.io/) is a large-scale, multiple-choice dataset for medical question answering with over 194k questions from medical entrance exams.

### Dataset Characteristics
- **Size**: 194k multiple-choice questions (182k train, 4.2k dev, 6.5k test)
- **Type**: Medical entrance exam questions (AIIMS and NEET-PG)
- **Specialties**: 18 medical subjects including Anatomy, Physiology, Biochemistry, etc.
- **Question Format**: Multiple-choice questions with 4 options

### Reference Results (Accuracy)

| System | Dev Set | Test Set |
|--------|---------|----------|
| Human Expert | - | 90.0% |
| Med-PaLM (2023) | 76.5% | 75.0% |
| BioLinkBERT | 52.9% | 51.3% |
| Basic RAG (BM25) | 46.2% | 45.1% |

## MMLU (Medical Subset)

[Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) includes medical subset tasks that can be used for evaluating medical knowledge.

### Dataset Characteristics
- Medical subtasks: Clinical Knowledge, Medical Genetics, Anatomy, etc.
- Multiple-choice format

### Reference Results (Medical Subtasks Accuracy)

| System | Clinical Knowledge | Anatomy | Medical Genetics | Average |
|--------|-------------------|---------|-----------------|---------|
| GPT-4 | 91.2% | 84.7% | 85.0% | 87.0% |
| GPT-4 + RAG | 93.8% | 88.5% | 89.2% | 90.5% |
| Claude 2 | 85.3% | 77.8% | 79.1% | 80.7% |
| Med-PaLM 2 | 90.5% | 83.2% | 86.0% | 86.6% |

## PubMedQA

[PubMedQA](https://pubmedqa.github.io/) is a biomedical question answering dataset based on PubMed abstracts.

### Dataset Characteristics
- **Size**: 1k expert-annotated QA pairs (for dev/test), 211.3k unlabeled instances
- **Type**: Yes/No/Maybe questions based on PubMed abstracts
- **Format**: Each instance includes a question, a context from a PubMed abstract, and an answer

### Reference Results (Accuracy)

| System | Test Set |
|--------|----------|
| Human performance | 78.0% |
| BioBERT + Retrieval | 60.2% |
| BM25 + PubMedBERT | 55.8% |
| SciFive + RAG | 68.5% |

## Implementing Benchmarks in Our System

To compare our RAG techniques against these published benchmarks:

1. **Dataset Adaptation**:
   - Create query sets based on real examples from these datasets
   - For MultiHopQA, use our MultiHop query template in the benchmarking script

2. **Metric Alignment**:
   - Implement F1 score calculation for answer quality
   - Add ROUGE-N metrics for summary-based answers
   - Calculate precision/recall for fact retrieval

3. **Reporting Format**:
   - Include comparison to published benchmarks in reports
   - Generate delta charts showing improvement/regression

4. **Custom Evaluation**:
   - For biomedical domain, emphasize BioASQ and PubMedQA comparisons
   - For multi-hop reasoning, focus on MultiHopQA metrics

## Using This Information

When running benchmarks with the `run_benchmark_demo.py` script:

```bash
# For standard medical queries
python run_benchmark_demo.py --dataset medical

# For multi-hop reasoning evaluation (comparable to MultiHopQA)
python run_benchmark_demo.py --dataset multihop
```

The generated reports will include comparisons to the reference results where applicable.
