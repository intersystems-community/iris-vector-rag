"""
PMC Downloader Integration with UnifiedDocumentLoader

This module provides seamless integration between the PMC downloader system
and the existing UnifiedDocumentLoader infrastructure for enterprise-scale
document loading into IRIS.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from .batch_downloader import PMCBatchDownloader, DownloadProgress
from data.unified_loader import UnifiedDocumentLoader, process_and_load_documents_unified
from data.pmc_processor import process_pmc_files
from iris_rag.config.manager import ConfigurationManager

logger = logging.getLogger(__name__)

class PMCEnterpriseLoader:
    """
    Enterprise-scale PMC document loader that combines downloading and loading.
    
    This class orchestrates the entire process:
    1. Download 10,000+ real PMC documents
    2. Validate document quality
    3. Load into IRIS using UnifiedDocumentLoader
    4. Provide comprehensive progress tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize enterprise PMC loader.
        
        Args:
            config: Configuration dictionary with downloader and loader settings
        """
        self.config = config
        
        # Initialize downloader
        downloader_config = config.get('downloader', {})
        downloader_config.update({
            'target_document_count': config.get('target_document_count', 10000),
            'download_directory': config.get('download_directory', 'data/pmc_enterprise'),
            'enable_validation': config.get('enable_validation', True)
        })
        self.downloader = PMCBatchDownloader(downloader_config)
        
        # Initialize loader configuration
        self.loader_config = config.get('loader', {})
        self.loader_config.update({
            'batch_size': config.get('batch_size', 100),
            'use_checkpointing': config.get('use_checkpointing', True),
            'embedding_column_type': config.get('embedding_column_type', 'VECTOR')
        })
        
        # Progress tracking
        self.total_phases = 3  # Download, Validate, Load
        self.current_phase = 0
        self.phase_names = ["Download", "Validate", "Load"]
        
    def load_enterprise_dataset(self, 
                              embedding_func: Optional[Callable] = None,
                              colbert_doc_encoder_func: Optional[Callable] = None,
                              progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Load enterprise-scale PMC dataset (10,000+ documents) into IRIS.
        
        Args:
            embedding_func: Optional function to generate document embeddings
            colbert_doc_encoder_func: Optional function to generate ColBERT token embeddings
            progress_callback: Optional callback for overall progress updates
            
        Returns:
            Dictionary with comprehensive loading results
        """
        start_time = time.time()
        
        logger.info("🚀 Starting enterprise PMC dataset loading...")
        logger.info(f"   Target documents: {self.config.get('target_document_count', 10000):,}")
        logger.info(f"   Download directory: {self.config.get('download_directory', 'data/pmc_enterprise')}")
        
        results = {
            'success': False,
            'total_time_seconds': 0,
            'phases': {},
            'final_document_count': 0,
            'download_results': {},
            'loading_results': {}
        }
        
        try:
            # Phase 1: Download and validate documents
            self.current_phase = 1
            logger.info("📥 Phase 1: Downloading and validating PMC documents...")
            
            def download_progress(progress: DownloadProgress):
                if progress_callback:
                    overall_progress = {
                        'phase': self.current_phase,
                        'phase_name': self.phase_names[self.current_phase - 1],
                        'phase_progress': progress.progress_percentage,
                        'overall_progress': ((self.current_phase - 1) / self.total_phases) * 100 + 
                                          (progress.progress_percentage / self.total_phases),
                        'current_operation': progress.current_file,
                        'documents_processed': progress.downloaded_documents,
                        'documents_validated': progress.validated_documents,
                        'estimated_time_remaining': progress.estimated_time_remaining_seconds
                    }
                    progress_callback(overall_progress)
            
            download_results = self.downloader.download_enterprise_dataset(download_progress)
            results['download_results'] = download_results
            results['phases']['download'] = download_results
            
            if not download_results['success']:
                raise RuntimeError(f"Download phase failed: {download_results.get('error')}")
            
            logger.info(f"✅ Download complete: {download_results['downloaded_documents']:,} documents")
            
            # Phase 2: Load documents into IRIS
            self.current_phase = 2
            logger.info("📊 Phase 2: Loading documents into IRIS...")
            
            extract_dir = Path(download_results['extract_directory'])
            if not extract_dir.exists():
                raise RuntimeError(f"Extract directory not found: {extract_dir}")
            
            # Count available XML files
            xml_files = list(extract_dir.glob("*.xml"))
            logger.info(f"Found {len(xml_files)} XML files to load")
            
            if len(xml_files) == 0:
                raise RuntimeError("No XML files found in extract directory")
            
            # Load documents using UnifiedDocumentLoader
            loading_start_time = time.time()
            
            loading_results = process_and_load_documents_unified(
                config=self.loader_config,
                pmc_directory=str(extract_dir),
                embedding_func=embedding_func,
                colbert_doc_encoder_func=colbert_doc_encoder_func,
                limit=download_results['downloaded_documents']
            )
            
            loading_time = time.time() - loading_start_time
            loading_results['loading_time_seconds'] = loading_time
            
            results['loading_results'] = loading_results
            results['phases']['loading'] = loading_results
            
            if not loading_results.get('success'):
                raise RuntimeError(f"Loading phase failed: {loading_results.get('error')}")
            
            logger.info(f"✅ Loading complete: {loading_results.get('loaded_doc_count', 0):,} documents loaded")
            
            # Phase 3: Final validation and reporting
            self.current_phase = 3
            logger.info("📋 Phase 3: Final validation and reporting...")
            
            final_document_count = loading_results.get('loaded_doc_count', 0)
            target_count = self.config.get('target_document_count', 10000)
            
            # Validate enterprise scale achievement
            if final_document_count < target_count:
                logger.warning(f"Target not fully achieved: {final_document_count} < {target_count}")
            else:
                logger.info(f"🎯 Enterprise scale achieved: {final_document_count:,} >= {target_count:,}")
            
            # Calculate final statistics
            total_time = time.time() - start_time
            
            results.update({
                'success': True,
                'total_time_seconds': total_time,
                'final_document_count': final_document_count,
                'target_achieved': final_document_count >= target_count,
                'download_time_seconds': download_results.get('download_time_seconds', 0),
                'loading_time_seconds': loading_time,
                'overall_throughput_docs_per_second': final_document_count / total_time if total_time > 0 else 0,
                'validation_summary': {
                    'total_downloaded': download_results.get('downloaded_documents', 0),
                    'total_validated': download_results.get('validated_documents', 0),
                    'total_loaded': final_document_count,
                    'validation_success_rate': (download_results.get('validated_documents', 0) / 
                                              max(download_results.get('downloaded_documents', 1), 1)) * 100
                }
            })
            
            # Final progress callback
            if progress_callback:
                final_progress = {
                    'phase': 3,
                    'phase_name': 'Complete',
                    'phase_progress': 100.0,
                    'overall_progress': 100.0,
                    'current_operation': 'Enterprise loading complete',
                    'documents_processed': final_document_count,
                    'documents_validated': download_results.get('validated_documents', 0),
                    'estimated_time_remaining': 0
                }
                progress_callback(final_progress)
            
            logger.info("🎉 Enterprise PMC dataset loading complete!")
            logger.info(f"   📊 Final count: {final_document_count:,} documents")
            logger.info(f"   ⏱️ Total time: {total_time:.2f} seconds")
            logger.info(f"   🚀 Throughput: {results['overall_throughput_docs_per_second']:.2f} docs/sec")
            logger.info(f"   ✅ Target achieved: {results['target_achieved']}")
            
            return results
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Enterprise loading failed in phase {self.current_phase}: {str(e)}"
            logger.error(error_msg)
            
            results.update({
                'success': False,
                'error': error_msg,
                'failed_phase': self.current_phase,
                'failed_phase_name': self.phase_names[self.current_phase - 1] if self.current_phase <= len(self.phase_names) else 'Unknown',
                'total_time_seconds': total_time
            })
            
            return results

def create_enterprise_loader_config(target_documents: int = 10000,
                                  download_dir: str = "data/pmc_enterprise",
                                  enable_validation: bool = True,
                                  batch_size: int = 100) -> Dict[str, Any]:
    """
    Create a standard configuration for enterprise PMC loading.
    
    Args:
        target_documents: Target number of documents to download
        download_dir: Directory for downloads
        enable_validation: Whether to validate documents
        batch_size: Batch size for loading
        
    Returns:
        Configuration dictionary
    """
    return {
        'target_document_count': target_documents,
        'download_directory': download_dir,
        'enable_validation': enable_validation,
        'batch_size': batch_size,
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
            'batch_size': batch_size,
            'token_batch_size': 1000,
            'embedding_column_type': 'VECTOR',
            'use_checkpointing': True,
            'refresh_connection': False,
            'gc_collect_interval': 50
        }
    }

def load_enterprise_pmc_dataset(target_documents: int = 10000,
                               embedding_func: Optional[Callable] = None,
                               colbert_doc_encoder_func: Optional[Callable] = None,
                               progress_callback: Optional[Callable] = None,
                               config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to load enterprise PMC dataset with default configuration.
    
    Args:
        target_documents: Target number of documents to download and load
        embedding_func: Optional function to generate document embeddings
        colbert_doc_encoder_func: Optional function to generate ColBERT token embeddings
        progress_callback: Optional callback for progress updates
        config_overrides: Optional configuration overrides
        
    Returns:
        Dictionary with loading results
    """
    # Create base configuration
    config = create_enterprise_loader_config(target_documents)
    
    # Apply overrides
    if config_overrides:
        config.update(config_overrides)
    
    # Create and run loader
    loader = PMCEnterpriseLoader(config)
    return loader.load_enterprise_dataset(
        embedding_func=embedding_func,
        colbert_doc_encoder_func=colbert_doc_encoder_func,
        progress_callback=progress_callback
    )