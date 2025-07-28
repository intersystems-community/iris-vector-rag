#!/usr/bin/env python3
"""
PMC Enterprise Document Downloader Script

Command-line interface for downloading and loading enterprise-scale
PMC document datasets for RAG Templates system testing and deployment.

Usage:
    uv run python scripts/download_pmc_enterprise.py --target 10000
    uv run python scripts/download_pmc_enterprise.py --target 50000 --download-dir data/pmc_50k
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.pmc_downloader import load_enterprise_pmc_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pmc_enterprise_download.log')
    ]
)
logger = logging.getLogger(__name__)

def create_progress_display():
    """Create a progress display function."""
    last_update = [0]  # Use list for mutable reference
    
    def progress_callback(progress_info: Dict[str, Any]):
        current_time = time.time()
        
        # Update every 5 seconds to avoid spam
        if current_time - last_update[0] < 5:
            return
        
        last_update[0] = current_time
        
        phase = progress_info.get('phase_name', 'Unknown')
        phase_progress = progress_info.get('phase_progress', 0)
        overall_progress = progress_info.get('overall_progress', 0)
        operation = progress_info.get('current_operation', '')
        docs_processed = progress_info.get('documents_processed', 0)
        docs_validated = progress_info.get('documents_validated', 0)
        eta_seconds = progress_info.get('estimated_time_remaining', 0)
        
        # Format ETA
        if eta_seconds > 0:
            eta_minutes = int(eta_seconds / 60)
            eta_str = f"ETA: {eta_minutes}m" if eta_minutes > 0 else f"ETA: {int(eta_seconds)}s"
        else:
            eta_str = "ETA: calculating..."
        
        print(f"\n📊 Progress Update:")
        print(f"   Phase: {phase} ({phase_progress:.1f}%)")
        print(f"   Overall: {overall_progress:.1f}%")
        print(f"   Operation: {operation}")
        print(f"   Documents: {docs_processed:,} processed, {docs_validated:,} validated")
        print(f"   {eta_str}")
        print("-" * 60)
    
    return progress_callback

def main():
    """Main entry point for PMC enterprise downloader."""
    parser = argparse.ArgumentParser(
        description="Download and load enterprise-scale PMC document datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 10,000 documents (default)
  uv run python scripts/download_pmc_enterprise.py
  
  # Download 50,000 documents to custom directory
  uv run python scripts/download_pmc_enterprise.py --target 50000 --download-dir data/pmc_50k
  
  # Download with custom batch size and validation disabled
  uv run python scripts/download_pmc_enterprise.py --target 25000 --batch-size 200 --no-validation
  
  # Resume from checkpoint
  uv run python scripts/download_pmc_enterprise.py --target 10000 --resume
        """
    )
    
    parser.add_argument(
        '--target', '-t',
        type=int,
        default=10000,
        help='Target number of documents to download (default: 10000)'
    )
    
    parser.add_argument(
        '--download-dir', '-d',
        type=str,
        default='data/pmc_enterprise',
        help='Directory for downloads (default: data/pmc_enterprise)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=100,
        help='Batch size for loading (default: 100)'
    )
    
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable document validation (faster but less reliable)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing checkpoint if available'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually downloading'
    )
    
    args = parser.parse_args()
    
    # Adjust logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Display configuration
    print("🚀 PMC Enterprise Document Downloader")
    print("=" * 50)
    print(f"Target Documents: {args.target:,}")
    print(f"Download Directory: {args.download_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Validation: {'Disabled' if args.no_validation else 'Enabled'}")
    print(f"Resume Mode: {'Enabled' if args.resume else 'Disabled'}")
    print(f"Dry Run: {'Yes' if args.dry_run else 'No'}")
    print("=" * 50)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No actual downloads will be performed")
        print(f"Would download {args.target:,} documents to {args.download_dir}")
        return 0
    
    # Confirm for large downloads
    if args.target > 25000:
        response = input(f"\n⚠️  You're about to download {args.target:,} documents. This may take several hours and use significant disk space. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            return 0
    
    try:
        # Create progress callback
        progress_callback = create_progress_display()
        
        # Configure overrides
        config_overrides = {
            'download_directory': args.download_dir,
            'batch_size': args.batch_size,
            'enable_validation': not args.no_validation
        }
        
        if args.resume:
            config_overrides['use_checkpointing'] = True
        
        print(f"\n🎯 Starting enterprise PMC download: {args.target:,} documents")
        print(f"📁 Download directory: {args.download_dir}")
        print(f"📊 Progress will be displayed every 5 seconds...")
        print("\n" + "=" * 60)
        
        start_time = time.time()
        
        # Execute the download and loading
        result = load_enterprise_pmc_dataset(
            target_documents=args.target,
            progress_callback=progress_callback,
            config_overrides=config_overrides
        )
        
        total_time = time.time() - start_time
        
        # Display results
        print("\n" + "=" * 60)
        print("📋 ENTERPRISE DOWNLOAD RESULTS")
        print("=" * 60)
        
        if result['success']:
            print("✅ Status: SUCCESS")
            print(f"📊 Final Document Count: {result.get('final_document_count', 0):,}")
            print(f"🎯 Target Achieved: {'YES' if result.get('target_achieved', False) else 'NO'}")
            print(f"⏱️  Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            print(f"🚀 Throughput: {result.get('overall_throughput_docs_per_second', 0):.2f} docs/sec")
            
            # Download phase results
            download_results = result.get('download_results', {})
            if download_results:
                print(f"\n📥 Download Phase:")
                print(f"   Downloaded: {download_results.get('downloaded_documents', 0):,}")
                print(f"   Validated: {download_results.get('validated_documents', 0):,}")
                print(f"   Failed: {download_results.get('failed_documents', 0):,}")
                print(f"   Time: {download_results.get('download_time_seconds', 0):.2f}s")
            
            # Loading phase results
            loading_results = result.get('loading_results', {})
            if loading_results:
                print(f"\n📊 Loading Phase:")
                print(f"   Loaded: {loading_results.get('loaded_doc_count', 0):,}")
                print(f"   Tokens: {loading_results.get('loaded_token_count', 0):,}")
                print(f"   Errors: {loading_results.get('error_count', 0):,}")
                print(f"   Time: {loading_results.get('loading_time_seconds', 0):.2f}s")
            
            print(f"\n📁 Data Location: {args.download_dir}")
            print("🎉 Enterprise dataset ready for RAG testing!")
            
            return 0
            
        else:
            print("❌ Status: FAILED")
            print(f"💥 Error: {result.get('error', 'Unknown error')}")
            print(f"📊 Documents Processed: {result.get('final_document_count', 0):,}")
            print(f"⏱️  Time Before Failure: {total_time:.2f} seconds")
            
            if 'failed_phase' in result:
                print(f"🚫 Failed Phase: {result['failed_phase_name']}")
            
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        print("💾 Progress may have been saved to checkpoint file")
        print("🔄 Use --resume flag to continue from where you left off")
        return 130
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n💥 Unexpected error: {e}")
        print("📋 Check pmc_enterprise_download.log for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())