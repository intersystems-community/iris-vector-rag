#!/usr/bin/env python3
"""
Unified PMC Document Loader

This script integrates existing PMC infrastructure to provide a complete solution for:
- Downloading and processing 10,000 PMC documents
- Configurable chunking and processing
- Database integration
- Standardized output generation
- Progress tracking and resume capability

Uses existing modules:
- data.pmc_processor for XML processing and chunking
- data.loader_fixed for database integration  
- scripts.utilities.download_real_pmc_docs for document downloading
"""

import logging
import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.pmc_processor import PMCProcessor, process_pmc_files
from data.loader_fixed import process_and_load_documents
from scripts.utilities.download_real_pmc_docs import SimplePMCDownloader

logger = logging.getLogger(__name__)

@dataclass
class PMCLoaderConfig:
    """Configuration for PMC loader."""
    chunk_size: int
    chunk_overlap: int
    download_dir: Path
    batch_size: int
    document_limit: int
    test_mode: bool
    output_dir: Path

def get_config() -> PMCLoaderConfig:
    """Get configuration from environment variables with defaults."""
    # Convert token-based sizes to character approximations (1 token ≈ 4 chars)
    chunk_size_tokens = int(os.getenv('PMC_CHUNK_SIZE', '512'))
    chunk_overlap_tokens = int(os.getenv('PMC_CHUNK_OVERLAP', '50'))
    
    chunk_size_chars = chunk_size_tokens * 4  # ~2048 chars for 512 tokens
    chunk_overlap_chars = chunk_overlap_tokens * 4  # ~200 chars for 50 tokens
    
    test_mode = os.getenv('PMC_TEST_MODE', 'False').lower() == 'true'
    document_limit = 100 if test_mode else int(os.getenv('PMC_DOCUMENT_LIMIT', '10000'))
    
    download_dir = Path(os.getenv('PMC_DOWNLOAD_DIR', 'data/pmc_dataset'))
    output_dir = download_dir  # Use same directory for output
    
    return PMCLoaderConfig(
        chunk_size=chunk_size_chars,
        chunk_overlap=chunk_overlap_chars,
        download_dir=download_dir,
        batch_size=int(os.getenv('PMC_BATCH_SIZE', '100')),
        document_limit=document_limit,
        test_mode=test_mode,
        output_dir=output_dir
    )

# Sample medical domain queries for evaluation
SAMPLE_QUERIES = [
    "What are the current treatment options for type 2 diabetes mellitus?",
    "How does COVID-19 affect respiratory function and lung capacity?",
    "What are the most effective interventions for managing hypertension?",
    "What are the risk factors and prevention strategies for cardiovascular disease?",
    "How do immunotherapy treatments work in cancer patients?",
    "What are the latest developments in Alzheimer's disease research?",
    "What are the symptoms and diagnostic criteria for depression?",
    "How effective are different pain management strategies for chronic conditions?",
    "What are the mechanisms of antibiotic resistance in bacterial infections?",
    "What nutritional interventions are recommended for metabolic syndrome?",
    "How do vaccines stimulate immune system responses?",
    "What are the genetic factors associated with breast cancer risk?",
    "How does inflammation contribute to autoimmune disease progression?",
    "What are the biomarkers used for early detection of kidney disease?",
    "What surgical techniques are most effective for treating heart valve disorders?"
]

class PMCDatasetLoader:
    """
    Unified PMC dataset loader that integrates existing infrastructure.
    
    Features:
    - Downloads real PMC documents using existing downloader
    - Processes XML using existing PMC processor with configurable chunking
    - Loads to database using existing loader
    - Generates standardized output files
    - Supports incremental loading and resume capability
    """
    
    def __init__(self, config: PMCLoaderConfig):
        self.config = config
        self.downloader = SimplePMCDownloader(str(config.download_dir))
        self.processor = PMCProcessor(
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap
        )
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking files
        self.progress_file = self.config.output_dir / "progress.json"
        self.metadata_file = self.config.output_dir / "metadata.json"
        self.documents_file = self.config.output_dir / "documents.jsonl"
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_file = self.config.output_dir / "pmc_loader.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Configure logger
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load progress from checkpoint file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load progress file: {e}")
        
        return {
            "phase": "starting",
            "downloaded_count": 0,
            "processed_count": 0,
            "loaded_count": 0,
            "start_time": None,
            "last_update": None
        }
    
    def _save_progress(self, progress: Dict[str, Any]):
        """Save progress to checkpoint file."""
        progress["last_update"] = datetime.now().isoformat()
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def download_documents(self, resume: bool = True) -> Dict[str, Any]:
        """
        Download PMC documents with resume capability.
        
        Args:
            resume: Whether to resume from previous progress
            
        Returns:
            Dictionary with download statistics
        """
        progress = self._load_progress() if resume else {}
        
        if progress.get("phase") == "download_complete":
            logger.info("Download phase already completed, skipping...")
            return {"success": True, "downloaded_count": progress.get("downloaded_count", 0)}
        
        logger.info(f"🚀 Starting PMC document download (limit: {self.config.document_limit})")
        
        # Check existing downloads
        existing_files = list(self.config.download_dir.glob("*.xml"))
        existing_count = len(existing_files)
        logger.info(f"Found {existing_count} existing downloaded files")
        
        progress["downloaded_count"] = existing_count
        progress["phase"] = "downloading"
        if not progress.get("start_time"):
            progress["start_time"] = datetime.now().isoformat()
        
        # Determine how many more we need
        if existing_count < self.config.document_limit:
            needed = self.config.document_limit - existing_count
            logger.info(f"Need to download {needed} more documents")
            
            # Get PMC IDs (get extra in case some fail)
            pmc_ids = self.downloader.get_pmc_ids(needed + min(200, needed // 2))
            
            if pmc_ids:
                # Download documents with progress tracking
                batch_size = min(self.config.batch_size // 10, 20)  # Smaller batches for downloading
                
                for i in range(0, min(len(pmc_ids), needed), batch_size):
                    batch = pmc_ids[i:i + batch_size]
                    
                    logger.info(f"Downloading batch {i//batch_size + 1}: {len(batch)} documents")
                    
                    for pmc_id in batch:
                        xml_path = self.downloader.download_single_pmc(pmc_id)
                        if xml_path:
                            progress["downloaded_count"] += 1
                        
                        if progress["downloaded_count"] % 50 == 0:
                            logger.info(f"Downloaded {progress['downloaded_count']} documents")
                            self._save_progress(progress)
                        
                        # Check if we've reached our limit
                        if progress["downloaded_count"] >= self.config.document_limit:
                            break
                    
                    if progress["downloaded_count"] >= self.config.document_limit:
                        break
                    
                    time.sleep(1)  # Rate limiting between batches
            else:
                logger.error("Failed to get PMC IDs")
                return {"success": False, "error": "Failed to get PMC IDs"}
        
        progress["phase"] = "download_complete"
        self._save_progress(progress)
        
        logger.info(f"✅ Download complete: {progress['downloaded_count']} documents")
        return {"success": True, "downloaded_count": progress["downloaded_count"]}
    
    def process_documents(self, resume: bool = True) -> Generator[Dict[str, Any], None, None]:
        """
        Process downloaded PMC documents with chunking.
        
        Args:
            resume: Whether to resume from previous progress
            
        Yields:
            Processed document dictionaries
        """
        progress = self._load_progress() if resume else {}
        
        if progress.get("phase") == "processing_complete":
            logger.info("Processing phase already completed")
            return
        
        logger.info(f"📄 Starting document processing with chunk_size={self.config.chunk_size}")
        
        progress["phase"] = "processing"
        processed_count = progress.get("processed_count", 0)
        
        # Process documents using existing infrastructure
        for i, doc in enumerate(process_pmc_files(
            str(self.config.download_dir), 
            limit=self.config.document_limit
        )):
            # Skip already processed documents if resuming
            if i < processed_count:
                continue
            
            # Add processing timestamp
            doc["metadata"]["processed_at"] = datetime.now().isoformat()
            doc["metadata"]["loader_version"] = "1.0.0"
            
            yield doc
            
            processed_count += 1
            progress["processed_count"] = processed_count
            
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count} documents")
                self._save_progress(progress)
        
        progress["phase"] = "processing_complete"
        self._save_progress(progress)
        logger.info(f"✅ Processing complete: {processed_count} documents")
    
    def generate_standardized_output(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate standardized output files (metadata.json, documents.jsonl).
        
        Args:
            documents: List of processed documents
            
        Returns:
            Dictionary with output statistics
        """
        logger.info("📝 Generating standardized output files...")
        
        # Calculate statistics
        total_docs = len(documents)
        total_chunks = sum(len(doc.get("chunks", [])) for doc in documents)
        total_content_length = sum(len(doc.get("content", "")) for doc in documents)
        
        # Create metadata
        metadata = {
            "dataset_name": "PMC Biomedical Dataset",
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "description": "Biomedical research documents from PubMed Central Open Access subset",
            "source": "PMC Open Access",
            "config": {
                "chunk_size_chars": self.config.chunk_size,
                "chunk_overlap_chars": self.config.chunk_overlap,
                "document_limit": self.config.document_limit,
                "test_mode": self.config.test_mode
            },
            "statistics": {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_content_length": total_content_length,
                "average_content_length": total_content_length / total_docs if total_docs > 0 else 0,
                "documents_with_chunks": sum(1 for doc in documents if doc.get("chunks")),
                "average_chunks_per_doc": total_chunks / total_docs if total_docs > 0 else 0
            },
            "sample_queries": SAMPLE_QUERIES,
            "files": {
                "metadata": "metadata.json",
                "documents": "documents.jsonl",
                "progress": "progress.json",
                "logs": "pmc_loader.log"
            }
        }
        
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save documents in JSONL format
        with open(self.documents_file, 'w') as f:
            for doc in documents:
                # Create standardized document format
                standardized_doc = {
                    "id": doc["doc_id"],
                    "title": doc["title"],
                    "content": doc["content"],
                    "abstract": doc["abstract"],
                    "authors": doc["authors"],
                    "keywords": doc["keywords"],
                    "metadata": doc["metadata"],
                    "chunks": doc.get("chunks", [])
                }
                f.write(json.dumps(standardized_doc, default=str) + '\n')
        
        logger.info(f"✅ Generated standardized output:")
        logger.info(f"  📄 {self.metadata_file.name}: {self.metadata_file.stat().st_size / 1024:.1f} KB")
        logger.info(f"  📄 {self.documents_file.name}: {self.documents_file.stat().st_size / (1024*1024):.1f} MB")
        
        return {
            "success": True,
            "metadata_file": str(self.metadata_file),
            "documents_file": str(self.documents_file),
            "statistics": metadata["statistics"]
        }
    
    def load_to_database(self, resume: bool = True) -> Dict[str, Any]:
        """
        Load processed documents to database using existing infrastructure.
        
        Args:
            resume: Whether to resume from previous progress
            
        Returns:
            Dictionary with loading statistics
        """
        progress = self._load_progress() if resume else {}
        
        if progress.get("phase") == "loading_complete":
            logger.info("Database loading phase already completed")
            return {"success": True, "loaded_count": progress.get("loaded_count", 0)}
        
        logger.info("💾 Loading documents to database...")
        
        progress["phase"] = "loading"
        
        # Use existing infrastructure for database loading
        load_result = process_and_load_documents(
            str(self.config.download_dir),
            limit=self.config.document_limit,
            batch_size=self.config.batch_size
        )
        
        progress["loaded_count"] = load_result.get("loaded_doc_count", 0)
        progress["phase"] = "loading_complete"
        self._save_progress(progress)
        
        logger.info(f"✅ Database loading complete: {progress['loaded_count']} documents")
        return load_result
    
    def run_complete_pipeline(self, resume: bool = True) -> Dict[str, Any]:
        """
        Run the complete PMC dataset creation pipeline.
        
        Args:
            resume: Whether to resume from previous progress
            
        Returns:
            Dictionary with complete pipeline statistics
        """
        start_time = time.time()
        
        mode_str = "TEST MODE (100 docs)" if self.config.test_mode else f"FULL MODE ({self.config.document_limit} docs)"
        logger.info(f"🚀 Starting PMC Dataset Loader - {mode_str}")
        logger.info(f"📁 Output directory: {self.config.output_dir}")
        
        try:
            # Phase 1: Download documents
            download_result = self.download_documents(resume=resume)
            if not download_result.get("success"):
                return download_result
            
            # Phase 2: Process documents
            logger.info("📄 Processing documents...")
            documents = list(self.process_documents(resume=resume))
            
            # Phase 3: Generate standardized output
            output_result = self.generate_standardized_output(documents)
            
            # Phase 4: Load to database
            load_result = self.load_to_database(resume=resume)
            
            # Final statistics
            duration = time.time() - start_time
            
            final_result = {
                "success": True,
                "mode": "test" if self.config.test_mode else "full",
                "duration_minutes": duration / 60,
                "download_result": download_result,
                "processing_result": {
                    "processed_count": len(documents),
                    "total_chunks": sum(len(doc.get("chunks", [])) for doc in documents)
                },
                "output_result": output_result,
                "load_result": load_result,
                "config": {
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                    "document_limit": self.config.document_limit,
                    "output_dir": str(self.config.output_dir)
                }
            }
            
            # Save final results
            results_file = self.config.output_dir / "final_results.json"
            with open(results_file, 'w') as f:
                json.dump(final_result, f, indent=2, default=str)
            
            logger.info("=" * 80)
            logger.info("🎉 PMC DATASET LOADER COMPLETE!")
            logger.info(f"📁 Output Directory: {self.config.output_dir}")
            logger.info(f"📊 Documents Downloaded: {download_result.get('downloaded_count', 0)}")
            logger.info(f"📄 Documents Processed: {len(documents)}")
            logger.info(f"📦 Total Chunks: {final_result['processing_result']['total_chunks']}")
            logger.info(f"💾 Documents Loaded: {load_result.get('loaded_doc_count', 0)}")
            logger.info(f"⏱️ Total Time: {duration/60:.1f} minutes")
            logger.info("=" * 80)
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration_minutes": (time.time() - start_time) / 60
            }

def main():
    """Main entry point for PMC loader."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Get configuration
    config = get_config()
    
    # Create and run loader
    loader = PMCDatasetLoader(config)
    result = loader.run_complete_pipeline(resume=True)
    
    if not result.get("success"):
        print(f"❌ Error: {result.get('error')}")
        sys.exit(1)
    
    print("✅ PMC Dataset Loader completed successfully!")
    return result

if __name__ == "__main__":
    main()