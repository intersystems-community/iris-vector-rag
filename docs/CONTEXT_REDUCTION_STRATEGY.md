# Context Reduction Strategies for RAG

This document outlines several strategies for reducing document context size to fit within LLM token limits, which is essential when dealing with large documents or multiple retrieved documents that exceed the context window of an LLM.

## The Context Size Problem

Most LLMs have a limited context window (e.g., 4K, 8K, or 16K tokens). When implementing RAG:

1. The retrieved documents may be too large to fit in the LLM's context window
2. Multiple relevant documents may collectively exceed the context window
3. Including irrelevant content reduces effectiveness and wastes tokens

Therefore, implementing effective context reduction strategies is crucial for:
- Ensuring relevant information fits within the token limit
- Prioritizing the most valuable content
- Maintaining retrieval quality and answer faithfulness

## Implemented Strategies

We have implemented and tested several context reduction strategies:

### 1. Simple Truncation

**Approach**: Sort documents by relevance score and include as many as possible, truncating the last document if necessary.

**Implementation**:
```python
def simple_truncation(documents: List[Document], max_tokens: int) -> str:
    # Sort documents by score
    # Include documents until max_tokens is reached
    # Truncate the last document if needed
```

**Pros**:
- Simple and fast - no external API calls required
- Maintains document integrity (except for last document)
- Predictable behavior

**Cons**:
- May cut off important information in later documents
- Doesn't optimize for relevance within documents
- Might include irrelevant sections from high-scoring documents

**Best for**: Quick implementations, when documents are small and well-structured, or when processing speed is critical.

### 2. Recursive Summarization

**Approach**: Generate summaries of each document, then combine them. If still too large, summarize the summaries.

**Implementation**:
```python
def recursive_summarization(documents: List[Document], query: str, max_tokens: int) -> str:
    # Generate summary for each document focusing on the query
    # Combine summaries
    # If still too large, either:
    #   - Summarize the combined summaries
    #   - Or truncate least important summaries
```

**Pros**:
- Can dramatically reduce context size while preserving key information
- Query-focused summaries maintain relevance
- Handles very large documents well

**Cons**:
- Requires LLM API calls (cost, latency)
- May lose details in summarization process
- Quality depends on summarization prompt and LLM capabilities

**Best for**: Very large documents, when preserving the general meaning is more important than specific details, and when LLM API usage is acceptable.

### 3. Embeddings Reranking

**Approach**: Break documents into smaller chunks, rerank chunks by relevance to query using embedding similarity, and select top chunks.

**Implementation**:
```python
def embeddings_reranking(documents: List[Document], query: str, embedding_model: Any, max_tokens: int) -> str:
    # Break documents into paragraph chunks
    # Generate embeddings for query and chunks
    # Rank chunks by embedding similarity to query
    # Select highest-ranking chunks that fit within max_tokens
```

**Pros**:
- More precise selection of relevant content
- Works across document boundaries
- Can identify relevant sections even in lower-scoring documents
- Doesn't require LLM API calls

**Cons**:
- Requires computing many embeddings (though cheaper than LLM calls)
- May break narrative flow by selecting non-contiguous chunks
- Embedding similarity isn't perfect for determining relevance

**Best for**: When fine-grained relevance is important, when you want to extract the most relevant parts from each document, or when preserving specific details is crucial.

### 4. Map-Reduce Approach

**Approach**: Process each document separately with an LLM to extract query-relevant information, then combine results.

**Implementation**:
```python
def map_reduce_approach(documents: List[Document], query: str) -> str:
    # MAP: For each document, extract relevant information with LLM
    # REDUCE: Combine the extracted information
```

**Pros**:
- Can handle very large document collections
- Extracts precisely what's needed from each document
- Scales well with document count

**Cons**:
- Requires multiple LLM API calls (one per document at minimum)
- May lose context between documents
- More complex to implement and tune

**Best for**: Complex information extraction tasks, when documents contain sparse relevant information, or when processing many documents with varied content.

## Selection Guidelines

1. **For speed and simplicity**: Simple Truncation
2. **For large documents with scattered relevance**: Embeddings Reranking
3. **For extremely large contexts** (e.g., books): Recursive Summarization
4. **For complex multi-document processing**: Map-Reduce Approach

## Performance Considerations

| Strategy | Speed | External API Calls | Quality | Implementation Complexity |
|----------|-------|-------------------|---------|---------------------------|
| Simple Truncation | Fastest | None | Lowest | Simplest |
| Embeddings Reranking | Medium | Embeddings only | Medium | Medium |
| Recursive Summarization | Slow | Multiple LLM calls | Medium-High | Medium |
| Map-Reduce | Slowest | Multiple LLM calls | Highest | Complex |

## Token Counting Considerations

Accurate token counting is essential for reliable context reduction. Our implementation uses a simple approximation:

```python
def count_tokens(text: str) -> int:
    words = text.split()
    # Rough heuristic: 1 word ≈ 1.3 tokens
    return int(len(words) * 1.3)
```

In production, it's recommended to use the actual LLM's tokenizer for more accurate counts. Most LLM providers offer tokenizer libraries:

- OpenAI: `tiktoken`
- Anthropic: `anthropic` package's tokenizer
- Hugging Face models: `transformers` tokenizers

## Hybrid and Adaptive Strategies

For optimal results, consider implementing adaptive strategies that select the best approach based on:

1. The number and size of retrieved documents
2. The nature of the query
3. Available processing time
4. Cost constraints

For example:
- Use simple truncation for small context overflows (<20%)
- Use embeddings reranking for medium overflows (20-200%)
- Use recursive summarization for large overflows (>200%)

## Testing and Evaluation

To evaluate context reduction strategies:

1. **Content Preservation**: Compare query answering accuracy with reduced vs. full context
2. **Relevance Retention**: Measure if key information for answering the query is preserved
3. **Processing Efficiency**: Measure time and resource usage for each strategy
4. **Token Efficiency**: Calculate the information density (relevant facts per token)

## Implementation Note

These strategies are implemented in `common/context_reduction.py` and can be easily integrated into any RAG pipeline. They are tested using test cases in `tests/test_context_reduction.py`.
