#!/usr/bin/env python3
"""
Simple 100K Validation Pipeline - Streamlined Version

This script provides a focused approach to 100k validation:
1. Check if data exists (skip download if sufficient)
2. Run validation on existing data
3. Generate simple report

Usage:
    python scripts/simple_100k_validation.py
    python scripts/simple_100k_validation.py --target-docs 50000
"""

import os
import sys
import logging
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_100k_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Simple100kValidator:
    """Simplified 100k validation pipeline"""
    
    def __init__(self, target_docs: int = 100000):
        self.target_docs = target_docs
        self.start_time = time.time()
        
    def check_data_availability(self) -> dict:
        """Check if sufficient data is available"""
        logger.info(f"🔍 Checking data availability for {self.target_docs:,} documents...")
        
        data_dir = Path("data/pmc_100k_downloaded")
        if not data_dir.exists():
            return {
                "sufficient": False,
                "found": 0,
                "message": "No data directory found"
            }
        
        xml_files = list(data_dir.rglob("*.xml"))
        found_count = len(xml_files)
        sufficient = found_count >= self.target_docs * 0.9  # 90% threshold
        
        logger.info(f"📊 Found {found_count:,} documents (need {self.target_docs:,})")
        
        return {
            "sufficient": sufficient,
            "found": found_count,
            "message": f"Found {found_count:,} documents" + 
                      (" - sufficient" if sufficient else " - insufficient")
        }
    
    def run_validation(self) -> dict:
        """Run the validation using existing ultimate validation script"""
        logger.info(f"🧪 Running validation on available data...")
        
        import subprocess
        
        # Use the existing ultimate validation script with proper parameters
        cmd = [
            sys.executable, 
            "scripts/ultimate_100k_enterprise_validation.py",
            "--docs", str(self.target_docs),
            "--skip-ingestion",
            "--fast-mode"
        ]
        
        logger.info(f"🔄 Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            success = result.returncode == 0
            
            return {
                "success": success,
                "returncode": result.returncode,
                "stdout": result.stdout[-2000:] if result.stdout else "",  # Last 2000 chars
                "stderr": result.stderr[-1000:] if result.stderr else "",   # Last 1000 chars
                "command": ' '.join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Validation timed out after 1 hour")
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Validation timed out after 1 hour",
                "command": ' '.join(cmd)
            }
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "command": ' '.join(cmd)
            }
    
    def generate_report(self, data_check: dict, validation_result: dict) -> str:
        """Generate simple validation report"""
        total_time = time.time() - self.start_time
        
        report = {
            "simple_100k_validation": {
                "timestamp": datetime.now().isoformat(),
                "target_documents": self.target_docs,
                "total_time_seconds": total_time,
                "data_availability": data_check,
                "validation_result": {
                    "success": validation_result["success"],
                    "returncode": validation_result["returncode"],
                    "command": validation_result["command"]
                }
            }
        }
        
        # Save report
        timestamp = int(time.time())
        report_file = f"simple_100k_validation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_file
    
    def print_summary(self, data_check: dict, validation_result: dict, report_file: str):
        """Print validation summary"""
        total_time = time.time() - self.start_time
        
        logger.info("\n" + "="*80)
        logger.info("🎯 SIMPLE 100K VALIDATION SUMMARY")
        logger.info("="*80)
        
        logger.info(f"📊 Target Documents: {self.target_docs:,}")
        logger.info(f"📁 Documents Found: {data_check['found']:,}")
        logger.info(f"✅ Data Sufficient: {data_check['sufficient']}")
        logger.info(f"🧪 Validation Success: {validation_result['success']}")
        logger.info(f"⏱️ Total Time: {total_time:.1f} seconds")
        
        if validation_result["success"]:
            logger.info("🎉 VALIDATION COMPLETED SUCCESSFULLY!")
            logger.info("🚀 System appears ready for 100k scale operation")
        else:
            logger.warning("⚠️ Validation encountered issues")
            logger.warning(f"Return code: {validation_result['returncode']}")
            if validation_result["stderr"]:
                logger.warning(f"Error: {validation_result['stderr'][:200]}...")
        
        logger.info(f"📄 Report saved: {report_file}")
        logger.info("="*80)
    
    def run(self) -> bool:
        """Run the complete simple validation"""
        logger.info(f"🚀 Starting Simple 100K Validation Pipeline...")
        
        try:
            # Check data availability
            data_check = self.check_data_availability()
            
            if not data_check["sufficient"]:
                logger.warning("⚠️ Insufficient data for validation")
                logger.warning("💡 Consider running download first or reducing target document count")
                
                # Generate report anyway
                validation_result = {
                    "success": False,
                    "returncode": -2,
                    "stdout": "",
                    "stderr": "Insufficient data for validation",
                    "command": "skipped"
                }
            else:
                # Run validation
                validation_result = self.run_validation()
            
            # Generate report and summary
            report_file = self.generate_report(data_check, validation_result)
            self.print_summary(data_check, validation_result, report_file)
            
            return validation_result["success"]
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple 100K Validation Pipeline")
    parser.add_argument("--target-docs", type=int, default=100000,
                       help="Target number of documents for validation")
    
    args = parser.parse_args()
    
    validator = Simple100kValidator(args.target_docs)
    success = validator.run()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)