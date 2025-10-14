#!/usr/bin/env python3
"""
Robust PMC Batch Ingestion System
Processes all 998 PMC documents with proper error handling, progress tracking, and resumption capabilities.
"""

import glob
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.config import get_config
from common.iris_client import IRISClient
from iris_rag.config.manager import ConfigManager
from iris_rag.pipelines.graphrag import GraphRAGPipeline


class RobustPMCBatchProcessor:
    """Robust batch processor for PMC documents with full error handling and progress tracking."""

    def __init__(self, data_dir: str = "data/downloaded_pmc_docs"):
        self.data_dir = Path(data_dir)
        self.config = get_config()
        self.iris_client = IRISClient()

        # Setup logging
        self.setup_logging()

        # Progress tracking
        self.progress_file = Path("outputs/logs/batch_ingestion_progress.json")
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)

        # Load or initialize progress
        self.progress = self.load_progress()

        # Initialize GraphRAG pipeline
        self.pipeline = None
        self.setup_pipeline()

    def setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = Path("outputs/logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"robust_batch_ingestion_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 Starting robust PMC batch ingestion - Log: {log_file}")

    def setup_pipeline(self):
        """Initialize GraphRAG pipeline with proper configuration."""
        try:
            self.logger.info("🔧 Initializing GraphRAG pipeline...")
            config_manager = ConfigManager()
            config = config_manager.get_pipeline_config("graphrag")
            self.pipeline = GraphRAGPipeline(config)
            self.logger.info("✅ GraphRAG pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize GraphRAG pipeline: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def load_progress(self) -> Dict[str, Any]:
        """Load processing progress from checkpoint file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r") as f:
                    progress = json.load(f)
                self.logger.info(
                    f"📂 Loaded progress: {progress['processed_count']}/{progress['total_files']} files processed"
                )
                return progress
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load progress file: {e}")

        # Initialize new progress tracking
        xml_files = list(self.data_dir.glob("*.xml"))
        progress = {
            "start_time": datetime.now().isoformat(),
            "total_files": len(xml_files),
            "processed_count": 0,
            "successful_count": 0,
            "failed_count": 0,
            "processed_files": [],
            "failed_files": [],
            "last_checkpoint": None,
            "total_entities_extracted": 0,
            "total_relationships_extracted": 0,
        }
        self.save_progress(progress)
        return progress

    def save_progress(self, progress: Dict[str, Any]):
        """Save current progress to checkpoint file."""
        progress["last_checkpoint"] = datetime.now().isoformat()
        with open(self.progress_file, "w") as f:
            json.dump(progress, f, indent=2)

    def get_pending_files(self) -> List[Path]:
        """Get list of files that haven't been processed yet."""
        xml_files = list(self.data_dir.glob("*.xml"))
        processed_files = set(self.progress.get("processed_files", []))

        pending_files = [f for f in xml_files if f.name not in processed_files]
        self.logger.info(f"📋 Found {len(pending_files)} pending files to process")
        return pending_files

    def process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single PMC XML file with comprehensive error handling."""
        start_time = time.time()
        result = {
            "file": file_path.name,
            "success": False,
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "processing_time": 0,
            "error": None,
        }

        try:
            self.logger.info(f"🔄 Processing: {file_path.name}")

            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Process through GraphRAG pipeline
            # Create a simple document structure
            document = {
                "content": content,
                "source": file_path.name,
                "metadata": {"file_path": str(file_path)},
            }

            # Process the document
            pipeline_result = self.pipeline.process(
                document["content"], document["metadata"]
            )

            # Extract metrics from result
            if pipeline_result and hasattr(pipeline_result, "metadata"):
                result["entities_extracted"] = pipeline_result.metadata.get(
                    "entities_count", 0
                )
                result["relationships_extracted"] = pipeline_result.metadata.get(
                    "relationships_count", 0
                )

            result["success"] = True
            result["processing_time"] = time.time() - start_time

            self.logger.info(
                f"✅ Processed {file_path.name}: "
                f"{result['entities_extracted']} entities, "
                f"{result['relationships_extracted']} relationships "
                f"({result['processing_time']:.2f}s)"
            )

        except Exception as e:
            result["error"] = str(e)
            result["processing_time"] = time.time() - start_time
            self.logger.error(f"❌ Failed to process {file_path.name}: {e}")

        return result

    def process_batch(self, batch_size: int = 10) -> Dict[str, Any]:
        """Process files in batches with progress reporting."""
        pending_files = self.get_pending_files()

        if not pending_files:
            self.logger.info("🎉 All files already processed!")
            return self.progress

        self.logger.info(
            f"🚀 Starting batch processing of {len(pending_files)} files (batch size: {batch_size})"
        )

        batch_start_time = time.time()

        for i, file_path in enumerate(pending_files):
            try:
                # Process single file
                result = self.process_single_file(file_path)

                # Update progress
                self.progress["processed_count"] += 1
                self.progress["processed_files"].append(file_path.name)

                if result["success"]:
                    self.progress["successful_count"] += 1
                    self.progress["total_entities_extracted"] += result[
                        "entities_extracted"
                    ]
                    self.progress["total_relationships_extracted"] += result[
                        "relationships_extracted"
                    ]
                else:
                    self.progress["failed_count"] += 1
                    self.progress["failed_files"].append(
                        {"file": file_path.name, "error": result["error"]}
                    )

                # Save progress every 10 files
                if (i + 1) % 10 == 0:
                    self.save_progress(self.progress)
                    self.logger.info(
                        f"📊 Progress checkpoint: {self.progress['processed_count']}/{self.progress['total_files']} "
                        f"({(self.progress['processed_count']/self.progress['total_files']*100):.1f}%)"
                    )

                # Report progress every batch
                if (i + 1) % batch_size == 0:
                    elapsed = time.time() - batch_start_time
                    rate = (i + 1) / elapsed
                    eta_seconds = (len(pending_files) - i - 1) / rate
                    eta_minutes = eta_seconds / 60

                    self.logger.info(
                        f"📈 Batch progress: {i+1}/{len(pending_files)} files "
                        f"({rate:.2f} files/sec, ETA: {eta_minutes:.1f} minutes)"
                    )

            except KeyboardInterrupt:
                self.logger.info("⏹️ Processing interrupted by user")
                self.save_progress(self.progress)
                return self.progress

            except Exception as e:
                self.logger.error(
                    f"💥 Unexpected error processing {file_path.name}: {e}"
                )
                self.progress["failed_count"] += 1
                self.progress["failed_files"].append(
                    {"file": file_path.name, "error": str(e)}
                )

        # Final save
        self.save_progress(self.progress)

        # Final summary
        total_time = time.time() - batch_start_time
        self.logger.info(f"🎯 Batch processing complete!")
        self.logger.info(
            f"   📊 Total processed: {self.progress['processed_count']}/{self.progress['total_files']}"
        )
        self.logger.info(f"   ✅ Successful: {self.progress['successful_count']}")
        self.logger.info(f"   ❌ Failed: {self.progress['failed_count']}")
        self.logger.info(
            f"   🏷️ Total entities: {self.progress['total_entities_extracted']}"
        )
        self.logger.info(
            f"   🔗 Total relationships: {self.progress['total_relationships_extracted']}"
        )
        self.logger.info(f"   ⏱️ Total time: {total_time/60:.2f} minutes")

        return self.progress


def main():
    """Main execution function."""
    print("🚀 ROBUST PMC BATCH INGESTION SYSTEM")
    print("=" * 50)

    processor = RobustPMCBatchProcessor()

    # Check if we should resume or start fresh
    if processor.progress["processed_count"] > 0:
        print(
            f"📂 Resuming from checkpoint: {processor.progress['processed_count']}/{processor.progress['total_files']} files already processed"
        )
        response = input("Continue from checkpoint? (y/n): ").lower()
        if response == "n":
            processor.progress = (
                processor.load_progress()
            )  # This will reset if file is deleted

    try:
        # Start batch processing
        final_progress = processor.process_batch(batch_size=10)

        # Generate final report
        print("\n" + "=" * 50)
        print("🎉 FINAL PROCESSING REPORT")
        print("=" * 50)
        print(
            f"📊 Files processed: {final_progress['processed_count']}/{final_progress['total_files']}"
        )
        print(f"✅ Successful: {final_progress['successful_count']}")
        print(f"❌ Failed: {final_progress['failed_count']}")
        print(
            f"🏷️ Total entities extracted: {final_progress['total_entities_extracted']}"
        )
        print(
            f"🔗 Total relationships extracted: {final_progress['total_relationships_extracted']}"
        )

        if final_progress["failed_files"]:
            print(f"\n❌ Failed files ({len(final_progress['failed_files'])}):")
            for failed in final_progress["failed_files"][:10]:  # Show first 10
                print(f"   - {failed['file']}: {failed['error'][:100]}...")

    except Exception as e:
        print(f"💥 Fatal error: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
