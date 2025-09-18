#!/usr/bin/env python3
"""
Large-Scale PMC Document Ingestion Starter
Kicks off background ingestion of 1000+ PMC documents
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path
from datetime import datetime

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

def count_pmc_files():
    """Count available PMC documents."""
    pmc_dir = project_root / "data" / "downloaded_pmc_docs"
    if pmc_dir.exists():
        xml_files = list(pmc_dir.glob("*.xml"))
        return len(xml_files)
    return 0

def test_database_connection():
    """Quick test of database connection."""
    try:
        from common.config import get_iris_config
        from common.iris_client import IRISClient
        
        print("🔌 Testing database connection...")
        config = get_iris_config()
        
        with IRISClient(config) as client:
            # Simple test query
            result = client.query("SELECT 1 as test")
            if result and result[0].get('test') == 1:
                print("✅ Database connection successful!")
                return True
            else:
                print("❌ Database connection failed - no results")
                return False
                
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def start_ingestion_process():
    """Start the large-scale ingestion process in background."""
    print("\n🚀 STARTING LARGE-SCALE PMC INGESTION")
    print("=" * 50)
    
    # Create ingestion script
    ingestion_script = """#!/usr/bin/env python3
import sys
import time
import logging
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/ingestion_{int(time.time())}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    logger.info("Starting PMC document ingestion...")
    
    # Import the actual ingestion pipeline
    try:
        from evaluation_framework.pmc_data_pipeline import PMCDataPipeline
        
        # Initialize pipeline
        pipeline = PMCDataPipeline(
            data_dir="data/downloaded_pmc_docs",
            batch_size=50,  # Process 50 docs at a time
            max_workers=4,  # Use 4 parallel workers
            enable_progress_tracking=True
        )
        
        # Start ingestion
        logger.info("Beginning large-scale ingestion of PMC documents...")
        results = pipeline.ingest_documents()
        
        logger.info(f"Ingestion completed! Processed {results.get('total_processed', 0)} documents")
        
    except ImportError as e:
        logger.error(f"Failed to import PMC pipeline: {e}")
        # Fallback to basic processing
        logger.info("Using fallback basic ingestion...")
        basic_ingestion()
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 1
    
    return 0

def basic_ingestion():
    '''Basic fallback ingestion'''
    from data.pmc_processor import PMCProcessor
    import glob
    
    processor = PMCProcessor()
    pmc_files = glob.glob("data/downloaded_pmc_docs/*.xml")
    
    logger.info(f"Found {len(pmc_files)} PMC files to process")
    
    for i, file_path in enumerate(pmc_files):
        try:
            logger.info(f"Processing {i+1}/{len(pmc_files)}: {Path(file_path).name}")
            processor.process_file(file_path)
            
            # Log progress every 50 files
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i+1}/{len(pmc_files)} files processed")
                
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            continue

if __name__ == "__main__":
    sys.exit(main())
"""
    
    # Write ingestion script
    ingestion_file = project_root / "scripts" / "run_ingestion.py"
    with open(ingestion_file, 'w') as f:
        f.write(ingestion_script)
    
    # Make executable
    os.chmod(ingestion_file, 0o755)
    
    # Create logs directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Start ingestion in background
    timestamp = int(time.time())
    log_file = logs_dir / f"ingestion_{timestamp}.log"
    
    print(f"📝 Ingestion logs will be written to: {log_file}")
    print("🏃 Starting ingestion process...")
    
    # Start subprocess
    process = subprocess.Popen([
        sys.executable, str(ingestion_file)
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    print(f"✅ Ingestion process started (PID: {process.pid})")
    print(f"📋 Monitor progress with: tail -f {log_file}")
    print("🔄 Process will continue in background while we develop visualizations...")
    
    return process

def main():
    """Main function."""
    print("🚀 LARGE-SCALE PMC INGESTION STARTER")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now()}")
    
    # Count available files
    file_count = count_pmc_files()
    print(f"📊 Available PMC files: {file_count}")
    
    if file_count < 100:
        print("⚠️  Warning: Less than 100 PMC files available")
        print("   Consider downloading more PMC documents for large-scale evaluation")
    
    # Test database connection
    if not test_database_connection():
        print("⚠️  Database connection failed - ingestion may have issues")
        print("   Continuing anyway - some ingestion modes can work offline")
    
    # Start ingestion
    process = start_ingestion_process()
    
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS:")
    print("1. Ingestion is now running in background")
    print("2. Developing data visualization components...")
    print("3. Monitor ingestion progress in logs/")
    print("4. Visualizations will be ready when ingestion completes")
    print("=" * 60)
    
    return process

if __name__ == "__main__":
    main()