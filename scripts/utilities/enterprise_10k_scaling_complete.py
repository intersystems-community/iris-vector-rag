#!/usr/bin/env python3
"""
Complete Enterprise 10K Document Scaling Pipeline
Scales the RAG system to 10,000 documents with all 7 techniques operational and comprehensive evaluation
"""

import sys
import time
import logging
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import traceback

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.embedding_utils import get_embedding_model # Updated import
from data.loader_optimized_performance import process_and_load_documents_optimized # Path remains correct
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'enterprise_10k_complete_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Enterprise10KCompleteScaling:
    """Complete 10K scaling with real document ingestion and all RAG components"""
    
    def __init__(self):
        self.connection = get_iris_connection()
        self.target_size = 10000
        self.batch_size = 50  # Memory-efficient batch size
        self.scaling_metrics = {}
        
        # Initialize embedding models
        self.embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
        
    def get_current_state(self) -> Dict[str, Any]:
        """Get comprehensive current database state"""
        try:
            cursor = self.connection.cursor()
            
            # Core document counts
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            doc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
            chunk_count = cursor.fetchone()[0]
            
            # Knowledge Graph components
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphEntities")
                entity_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphRelationships")
                rel_count = cursor.fetchone()[0]
            except:
                entity_count = 0
                rel_count = 0
            
            # ColBERT token embeddings
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
                token_count = cursor.fetchone()[0]
            except:
                token_count = 0
            
            cursor.close()
            
            return {
                'document_count': doc_count,
                'chunk_count': chunk_count,
                'entity_count': entity_count,
                'relationship_count': rel_count,
                'token_embedding_count': token_count,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get current state: {e}")
            return {}
    
    def get_memory_metrics(self) -> Dict[str, Any]:
        """Get system memory metrics"""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            
            return {
                'system_memory_total_gb': memory.total / (1024**3),
                'system_memory_used_gb': memory.used / (1024**3),
                'system_memory_percent': memory.percent,
                'process_memory_mb': process.memory_info().rss / (1024**2),
                'process_memory_percent': process.memory_percent(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Failed to get memory metrics: {e}")
            return {}
    
    def check_available_data_files(self) -> List[str]:
        """Check for available PMC data files for scaling"""
        data_dir = Path("data")
        
        # Look for PMC XML files
        xml_files = list(data_dir.glob("*.xml"))
        nxml_files = list(data_dir.glob("*.nxml"))
        
        # Look for compressed files
        gz_files = list(data_dir.glob("*.xml.gz"))
        tar_files = list(data_dir.glob("*.tar.gz"))
        
        all_files = xml_files + nxml_files + gz_files + tar_files
        
        logger.info(f"📁 Found {len(all_files)} potential data files")
        for file in all_files[:5]:  # Show first 5
            logger.info(f"   {file.name}") # Assuming this was the intended content for the loop
        
        return [str(f) for f in all_files] # Ensuring this is part of check_available_data_files

    def scale_documents_to_10k(self, current_docs: int) -> Dict[str, Any]: # Corrected indentation
        """Scale documents to 10K using real PMC data"""
        docs_needed = self.target_size - current_docs
        
        if docs_needed <= 0:
            return {
                'success': True,
                'documents_added': 0,
                'already_at_target': True,
                'message': f'Already at target size: {current_docs:,} >= {self.target_size:,}'
            }
        
        logger.info(f"🎯 Scaling from {current_docs:,} to {self.target_size:,} documents")
        logger.info(f"📈 Need to add {docs_needed:,} documents")
        
        start_time = time.time()
        memory_before = self.get_memory_metrics()
        
        try:
            # Use the optimized data loader to add more documents
            data_dir = "data"
            
            # Process and load documents with memory-efficient approach
            load_result = process_and_load_documents_optimized(
                pmc_directory=data_dir,
                connection=self.connection,
                embedding_func=self.embedding_func,
                colbert_doc_encoder_func=self.colbert_doc_encoder_func,
                limit=docs_needed,
                batch_size=self.batch_size,
                token_batch_size=1000,
                use_mock=False
            )
            
            processing_time = time.time() - start_time
            memory_after = self.get_memory_metrics()
            
            if load_result.get('success'):
                docs_added = load_result.get('loaded_doc_count', 0)
                tokens_added = load_result.get('loaded_token_count', 0)
                
                return {
                    'success': True,
                    'documents_added': docs_added,
                    'tokens_added': tokens_added,
                    'processing_time_seconds': processing_time,
                    'documents_per_second': docs_added / processing_time if processing_time > 0 else 0,
                    'memory_before': memory_before,
                    'memory_after': memory_after,
                    'load_result': load_result
                }
            else:
                return {
                    'success': False,
                    'error': load_result.get('error', 'Unknown error'),
                    'processing_time_seconds': processing_time
                }
                
        except Exception as e:
            logger.error(f"❌ Error scaling documents: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time_seconds': time.time() - start_time
            }
    
    def embedding_func(self, texts: List[str]) -> List[List[float]]: # Corrected indentation
        """Generate embeddings for texts"""
        return self.embedding_model.encode(texts).tolist()
    
    def colbert_doc_encoder_func(self, text: str) -> Tuple[List[str], List[List[float]]]: # Corrected indentation
        """Generate ColBERT token embeddings for document"""
        try:
            # Simple tokenization and embedding for ColBERT simulation
            tokens = text.split()[:100]  # Limit tokens for performance
            if not tokens:
                return [], []
            
            # Generate embeddings for each token
            embeddings = self.embedding_model.encode(tokens).tolist()
            return tokens, embeddings
            
        except Exception as e:
            logger.error(f"Error in ColBERT encoding: {e}")
            return [], []

def main():
    """Main execution function"""
    try:
        logger.info("🚀 Starting Enterprise 10K Complete Scaling Pipeline")
        logger.info("="*80)
        
        # Initialize scaling pipeline
        scaler = Enterprise10KCompleteScaling()
        
        # Run complete scaling
        results = scaler.run_complete_10k_scaling()
        
        logger.info("\n🎉 Enterprise 10K Scaling Pipeline Complete!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()