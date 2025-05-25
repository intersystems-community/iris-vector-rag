"""
Document Chunking Service for PMC RAG System

This module provides a comprehensive document chunking service that implements
multiple chunking strategies for improved vector retrieval effectiveness.
"""

import logging
import json
import re
import sys
import os
from typing import List, Dict, Any, Optional, Tuple, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
import statistics

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.iris_connector import get_iris_connection

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    text: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]
    chunk_type: str = "unknown"

class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str, doc_id: str = None) -> List[Chunk]:
        """Chunk the given text into a list of chunks."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        pass

class FixedSizeChunkingStrategy(ChunkingStrategy):
    """Fixed-size chunking with configurable overlap and sentence preservation."""
    
    def __init__(self, chunk_size: int = 512, overlap_size: int = 50, 
                 preserve_sentences: bool = True, min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.preserve_sentences = preserve_sentences
        self.min_chunk_size = min_chunk_size
    
    def get_strategy_name(self) -> str:
        return "fixed_size"
    
    def chunk(self, text: str, doc_id: str = None) -> List[Chunk]:
        """Chunk text into fixed-size pieces with overlap."""
        if len(text) <= self.chunk_size:
            return [Chunk(
                text=text,
                start_pos=0,
                end_pos=len(text),
                metadata={"chunk_size": len(text), "is_complete_doc": True},
                chunk_type=self.get_strategy_name()
            )]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to preserve sentence boundaries
            if self.preserve_sentences and end < len(text):
                # Look for sentence endings within the last 20% of the chunk
                search_start = max(start + int(self.chunk_size * 0.8), start + self.min_chunk_size)
                sentence_end = self._find_sentence_boundary(text, search_start, end)
                if sentence_end > search_start:
                    end = sentence_end
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    start_pos=start,
                    end_pos=end,
                    metadata={
                        "chunk_index": chunk_index,
                        "chunk_size": len(chunk_text),
                        "overlap_with_previous": min(self.overlap_size, start) if start > 0 else 0,
                        "strategy_params": {
                            "target_chunk_size": self.chunk_size,
                            "overlap_size": self.overlap_size
                        }
                    },
                    chunk_type=self.get_strategy_name()
                ))
                chunk_index += 1
            
            # Move start position with overlap
            start = max(end - self.overlap_size, start + 1)
            
            # Prevent infinite loop
            if start >= end:
                break
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best sentence boundary within the given range."""
        # Look for sentence endings (., !, ?) followed by space or end of text
        sentence_pattern = r'[.!?]\s+'
        
        # Search backwards from end to start
        search_text = text[start:end]
        matches = list(re.finditer(sentence_pattern, search_text))
        
        if matches:
            # Return position after the last sentence ending
            last_match = matches[-1]
            return start + last_match.end()
        
        return end

class SemanticChunkingStrategy(ChunkingStrategy):
    """Semantic chunking based on sentence similarity (simplified version)."""
    
    def __init__(self, similarity_threshold: float = 0.7, min_chunk_size: int = 200, 
                 max_chunk_size: int = 1000):
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def get_strategy_name(self) -> str:
        return "semantic"
    
    def chunk(self, text: str, doc_id: str = None) -> List[Chunk]:
        """Chunk text based on semantic boundaries (simplified implementation)."""
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [Chunk(
                text=text,
                start_pos=0,
                end_pos=len(text),
                metadata={"sentence_count": len(sentences), "is_single_sentence": True},
                chunk_type=self.get_strategy_name()
            )]
        
        chunks = []
        current_chunk_sentences = []
        current_start_pos = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            current_chunk_sentences.append(sentence)
            current_text = ' '.join(current_chunk_sentences)
            
            # Check if we should split here
            should_split = (
                len(current_text) >= self.max_chunk_size or
                (len(current_text) >= self.min_chunk_size and 
                 self._should_split_here(sentences, i)) or
                i == len(sentences) - 1  # Last sentence
            )
            
            if should_split and len(current_text) >= self.min_chunk_size:
                chunk_end_pos = current_start_pos + len(current_text)
                
                chunks.append(Chunk(
                    text=current_text,
                    start_pos=current_start_pos,
                    end_pos=chunk_end_pos,
                    metadata={
                        "chunk_index": chunk_index,
                        "sentence_count": len(current_chunk_sentences),
                        "semantic_boundary": i < len(sentences) - 1,
                        "strategy_params": {
                            "similarity_threshold": self.similarity_threshold,
                            "min_chunk_size": self.min_chunk_size,
                            "max_chunk_size": self.max_chunk_size
                        }
                    },
                    chunk_type=self.get_strategy_name()
                ))
                
                current_start_pos = chunk_end_pos + 1
                current_chunk_sentences = []
                chunk_index += 1
        
        # Handle any remaining sentences
        if current_chunk_sentences:
            current_text = ' '.join(current_chunk_sentences)
            chunks.append(Chunk(
                text=current_text,
                start_pos=current_start_pos,
                end_pos=current_start_pos + len(current_text),
                metadata={
                    "chunk_index": chunk_index,
                    "sentence_count": len(current_chunk_sentences),
                    "is_final_chunk": True
                },
                chunk_type=self.get_strategy_name()
            ))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex."""
        # Simple sentence splitting - could be improved with spaCy or NLTK
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _should_split_here(self, sentences: List[str], current_index: int) -> bool:
        """Determine if we should split at the current sentence (simplified)."""
        # Simplified semantic boundary detection
        # In a real implementation, this would use sentence embeddings
        
        if current_index >= len(sentences) - 1:
            return True
        
        current_sentence = sentences[current_index].lower()
        next_sentence = sentences[current_index + 1].lower()
        
        # Simple heuristics for topic boundaries
        topic_indicators = [
            'however', 'furthermore', 'moreover', 'in addition', 'on the other hand',
            'in contrast', 'meanwhile', 'subsequently', 'therefore', 'thus',
            'in conclusion', 'finally', 'first', 'second', 'third'
        ]
        
        # Check if next sentence starts with a topic indicator
        for indicator in topic_indicators:
            if next_sentence.startswith(indicator):
                return True
        
        # Check for significant vocabulary change (simplified)
        current_words = set(re.findall(r'\w+', current_sentence))
        next_words = set(re.findall(r'\w+', next_sentence))
        
        if len(current_words) > 0 and len(next_words) > 0:
            overlap = len(current_words.intersection(next_words))
            similarity = overlap / min(len(current_words), len(next_words))
            return similarity < self.similarity_threshold
        
        return False

class HybridChunkingStrategy(ChunkingStrategy):
    """Hybrid chunking that combines semantic and fixed-size strategies."""
    
    def __init__(self, primary_strategy: ChunkingStrategy, 
                 fallback_strategy: ChunkingStrategy,
                 max_chunk_size: int = 800):
        self.primary_strategy = primary_strategy
        self.fallback_strategy = fallback_strategy
        self.max_chunk_size = max_chunk_size
    
    def get_strategy_name(self) -> str:
        return "hybrid"
    
    def chunk(self, text: str, doc_id: str = None) -> List[Chunk]:
        """Use primary strategy, fall back to secondary for oversized chunks."""
        primary_chunks = self.primary_strategy.chunk(text, doc_id)
        
        final_chunks = []
        for chunk in primary_chunks:
            if len(chunk.text) <= self.max_chunk_size:
                # Chunk is acceptable size, keep as-is but update type
                chunk.chunk_type = self.get_strategy_name()
                chunk.metadata["primary_strategy"] = self.primary_strategy.get_strategy_name()
                final_chunks.append(chunk)
            else:
                # Chunk is too large, re-chunk with fallback strategy
                sub_chunks = self.fallback_strategy.chunk(chunk.text, doc_id)
                for i, sub_chunk in enumerate(sub_chunks):
                    # Adjust positions relative to original text
                    sub_chunk.start_pos += chunk.start_pos
                    sub_chunk.end_pos += chunk.start_pos
                    sub_chunk.chunk_type = self.get_strategy_name()
                    sub_chunk.metadata.update({
                        "primary_strategy": self.primary_strategy.get_strategy_name(),
                        "fallback_strategy": self.fallback_strategy.get_strategy_name(),
                        "was_re_chunked": True,
                        "original_chunk_size": len(chunk.text)
                    })
                    final_chunks.append(sub_chunk)
        
        return final_chunks

class DocumentChunkingService:
    """Main service for document chunking operations."""
    
    def __init__(self, embedding_func=None):
        self.strategies = self._initialize_strategies()
        self.embedding_func = embedding_func or self._get_mock_embedding_func()
    
    def _initialize_strategies(self) -> Dict[str, ChunkingStrategy]:
        """Initialize available chunking strategies."""
        fixed_strategy = FixedSizeChunkingStrategy(
            chunk_size=512, overlap_size=50, preserve_sentences=True
        )
        
        semantic_strategy = SemanticChunkingStrategy(
            similarity_threshold=0.6, min_chunk_size=200, max_chunk_size=1000
        )
        
        hybrid_strategy = HybridChunkingStrategy(
            primary_strategy=semantic_strategy,
            fallback_strategy=fixed_strategy,
            max_chunk_size=800
        )
        
        return {
            "fixed_size": fixed_strategy,
            "semantic": semantic_strategy,
            "hybrid": hybrid_strategy
        }
    
    def _get_mock_embedding_func(self):
        """Mock embedding function for testing."""
        def mock_embed(texts: List[str]) -> List[List[float]]:
            # Return random embeddings for testing
            import random
            return [[random.random() for _ in range(768)] for _ in texts]
        return mock_embed
    
    def chunk_document(self, doc_id: str, text: str, strategy_name: str) -> List[Dict[str, Any]]:
        """Chunk a document using the specified strategy."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown chunking strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        chunks = strategy.chunk(text, doc_id)
        
        chunk_records = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{strategy_name}_{i}"
            
            # Generate embedding
            try:
                embedding = self.embedding_func([chunk.text])[0]
                embedding_str = ','.join(map(str, embedding))
            except Exception as e:
                logger.warning(f"Failed to generate embedding for chunk {chunk_id}: {e}")
                embedding_str = None
            
            chunk_records.append({
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'chunk_index': i,
                'chunk_type': chunk.chunk_type,
                'chunk_text': chunk.text,
                'start_position': chunk.start_pos,
                'end_position': chunk.end_pos,
                'embedding_str': embedding_str,
                'chunk_metadata': json.dumps(chunk.metadata)
            })
        
        return chunk_records
    
    def analyze_chunking_effectiveness(self, doc_id: str, text: str) -> Dict[str, Any]:
        """Analyze how different chunking strategies would perform on a document."""
        analysis = {
            "document_info": {
                "doc_id": doc_id,
                "original_length": len(text),
                "word_count": len(text.split()),
                "sentence_count": len(re.split(r'[.!?]+', text))
            },
            "strategy_analysis": {}
        }
        
        for strategy_name, strategy in self.strategies.items():
            try:
                chunks = strategy.chunk(text, doc_id)
                
                chunk_lengths = [len(chunk.text) for chunk in chunks]
                
                strategy_analysis = {
                    "chunk_count": len(chunks),
                    "avg_chunk_length": statistics.mean(chunk_lengths) if chunk_lengths else 0,
                    "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
                    "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
                    "std_dev_length": statistics.stdev(chunk_lengths) if len(chunk_lengths) > 1 else 0,
                    "chunks_exceeding_512_tokens": sum(1 for length in chunk_lengths if length > 1920),  # ~512 tokens
                    "chunks_exceeding_384_tokens": sum(1 for length in chunk_lengths if length > 1440),  # ~384 tokens
                    "total_chunked_length": sum(chunk_lengths),
                    "coverage_ratio": sum(chunk_lengths) / len(text) if len(text) > 0 else 0
                }
                
                analysis["strategy_analysis"][strategy_name] = strategy_analysis
                
            except Exception as e:
                logger.error(f"Error analyzing strategy {strategy_name}: {e}")
                analysis["strategy_analysis"][strategy_name] = {"error": str(e)}
        
        return analysis
    
    def store_chunks(self, chunk_records: List[Dict[str, Any]], connection=None) -> bool:
        """Store chunk records in the database."""
        conn_provided = connection is not None
        if not connection:
            connection = get_iris_connection()
        
        try:
            cursor = connection.cursor()
            
            # Insert chunks
            sql = """
            INSERT INTO RAG.DocumentChunks
            (chunk_id, doc_id, chunk_index, chunk_type, chunk_text, 
             start_position, end_position, embedding_str, chunk_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            chunk_params = []
            for record in chunk_records:
                chunk_params.append((
                    record['chunk_id'],
                    record['doc_id'],
                    record['chunk_index'],
                    record['chunk_type'],
                    record['chunk_text'],
                    record['start_position'],
                    record['end_position'],
                    record['embedding_str'],
                    record['chunk_metadata']
                ))
            
            cursor.executemany(sql, chunk_params)
            connection.commit()
            cursor.close()
            
            logger.info(f"Successfully stored {len(chunk_records)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            if connection:
                connection.rollback()
            return False
        finally:
            if not conn_provided and connection:
                connection.close()
    
    def process_sample_documents(self, limit: int = 10, strategy_names: List[str] = None) -> Dict[str, Any]:
        """Process a sample of documents with chunking for testing."""
        if strategy_names is None:
            strategy_names = ["hybrid"]
        
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            # Get sample documents (avoid LONGVARCHAR comparison)
            cursor.execute("""
                SELECT doc_id, title, text_content
                FROM RAG.SourceDocuments
                WHERE text_content IS NOT NULL
                LIMIT ?
            """, (limit,))
            
            documents = cursor.fetchall()
            
            results = {
                "processed_documents": 0,
                "total_chunks_created": 0,
                "strategy_results": {name: {"chunks": 0, "documents": 0} for name in strategy_names},
                "analysis_results": [],
                "errors": []
            }
            
            for doc_id, title, text_content in documents:
                # Skip documents with empty content
                if not text_content or text_content.strip() == "":
                    continue
                    
                try:
                    # Analyze chunking effectiveness
                    analysis = self.analyze_chunking_effectiveness(doc_id, text_content)
                    analysis["title"] = title
                    results["analysis_results"].append(analysis)
                    
                    # Process with each strategy
                    for strategy_name in strategy_names:
                        try:
                            chunks = self.chunk_document(doc_id, text_content, strategy_name)
                            
                            # Store chunks (commented out to avoid modifying database in demo)
                            # success = self.store_chunks(chunks, connection)
                            
                            results["strategy_results"][strategy_name]["chunks"] += len(chunks)
                            results["strategy_results"][strategy_name]["documents"] += 1
                            results["total_chunks_created"] += len(chunks)
                            
                        except Exception as e:
                            error_msg = f"Error processing {doc_id} with {strategy_name}: {e}"
                            logger.error(error_msg)
                            results["errors"].append(error_msg)
                    
                    results["processed_documents"] += 1
                    
                except Exception as e:
                    error_msg = f"Error processing document {doc_id}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in process_sample_documents: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            connection.close()

def main():
    """Demo the chunking service."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("="*60)
    print("DOCUMENT CHUNKING SERVICE DEMO")
    print("="*60)
    
    # Initialize service
    service = DocumentChunkingService()
    
    # Process sample documents
    print("\n🔄 Processing sample documents with chunking...")
    results = service.process_sample_documents(limit=5, strategy_names=["fixed_size", "semantic", "hybrid"])
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"\n📊 PROCESSING RESULTS:")
    print(f"Documents processed: {results['processed_documents']}")
    print(f"Total chunks created: {results['total_chunks_created']}")
    
    print(f"\n📈 STRATEGY PERFORMANCE:")
    for strategy, stats in results["strategy_results"].items():
        avg_chunks = stats["chunks"] / stats["documents"] if stats["documents"] > 0 else 0
        print(f"  {strategy}: {stats['chunks']} chunks from {stats['documents']} docs (avg: {avg_chunks:.1f} chunks/doc)")
    
    if results["errors"]:
        print(f"\n⚠️  ERRORS ({len(results['errors'])}):")
        for error in results["errors"][:3]:  # Show first 3 errors
            print(f"  - {error}")
    
    print(f"\n📋 DETAILED ANALYSIS (first 3 documents):")
    for i, analysis in enumerate(results["analysis_results"][:3]):
        doc_info = analysis["document_info"]
        print(f"\n  Document {i+1}: {doc_info['doc_id']}")
        print(f"    Title: {analysis.get('title', 'N/A')[:80]}...")
        print(f"    Length: {doc_info['original_length']:,} chars, {doc_info['word_count']} words")
        
        for strategy, stats in analysis["strategy_analysis"].items():
            if "error" not in stats:
                print(f"    {strategy}: {stats['chunk_count']} chunks, avg {stats['avg_chunk_length']:.0f} chars")
    
    print(f"\n✅ Demo completed successfully!")

if __name__ == "__main__":
    main()