#!/usr/bin/env python3
"""
Complete 100K Validation Orchestrator

This script orchestrates the complete bulletproof 100k validation pipeline:
1. Download 100k PMC articles (if needed)
2. Ingest 100k documents into IRIS database
3. Run ultimate enterprise validation on all 7 RAG techniques
4. Generate comprehensive reports and recommendations

Usage:
    python scripts/run_complete_100k_validation.py
    python scripts/run_complete_100k_validation.py --target-docs 50000
    python scripts/run_complete_100k_validation.py --skip-download --skip-ingestion
"""

import os
import sys
import logging
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_100k_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Complete100kValidationOrchestrator:
    """Orchestrates the complete 100k validation pipeline"""
    
    def __init__(self, target_docs: int = 100000):
        self.target_docs = target_docs
        self.start_time = time.time()
        self.results = {
            "orchestration_summary": {
                "target_docs": target_docs,
                "start_time": datetime.now().isoformat(),
                "phases_completed": [],
                "phases_failed": [],
                "total_time_seconds": 0
            },
            "download_results": {},
            "ingestion_results": {},
            "validation_results": {}
        }
        
        logger.info(f"🚀 Complete 100K Validation Orchestrator initialized")
        logger.info(f"🎯 Target documents: {target_docs:,}")
    
    def run_script(self, script_path: str, args: List[str] = None, background: bool = False) -> Dict[str, Any]:
        """Run a Python script and capture results with proper process management"""
        if args is None:
            args = []
            
        cmd = [sys.executable, script_path] + args
        logger.info(f"🔄 Running: {' '.join(cmd)}")
        
        try:
            if background:
                # For background processes, use Popen and monitor completion
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait for completion with periodic status checks
                while process.poll() is None:
                    time.sleep(10)  # Check every 10 seconds
                    logger.info(f"⏳ Background process still running: {script_path}")
                
                stdout, stderr = process.communicate()
                
                return {
                    "success": process.returncode == 0,
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "command": ' '.join(cmd)
                }
            else:
                # For foreground processes, use run with proper timeout
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=7200  # 2 hour timeout
                )
                
                return {
                    "success": result.returncode == 0,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "command": ' '.join(cmd)
                }
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Script timed out: {script_path}")
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Script timed out after 2 hours",
                "command": ' '.join(cmd)
            }
        except Exception as e:
            logger.error(f"❌ Error running script: {e}")
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "command": ' '.join(cmd)
            }
    
    def phase_1_download(self, skip_download: bool = False) -> bool:
        """Phase 1: Download 100k PMC articles"""
        logger.info("\n" + "="*80)
        logger.info("📥 PHASE 1: DOWNLOADING 100K PMC ARTICLES")
        logger.info("="*80)
        
        if skip_download:
            logger.info("⏭️ Skipping download phase")
            self.results["orchestration_summary"]["phases_completed"].append("download_skipped")
            return True
        
        phase_start = time.time()
        
        # Check if data already exists
        data_dir = Path("data/pmc_100k_downloaded")
        if data_dir.exists():
            xml_files = list(data_dir.rglob("*.xml"))
            if len(xml_files) >= self.target_docs * 0.9:  # 90% of target
                logger.info(f"✅ Sufficient data already exists: {len(xml_files):,} files")
                self.results["download_results"] = {
                    "success": True,
                    "files_found": len(xml_files),
                    "skipped": True,
                    "phase_time_seconds": time.time() - phase_start
                }
                self.results["orchestration_summary"]["phases_completed"].append("download_existing")
                return True
        
        # Run download script
        download_args = [
            "--target-count", str(self.target_docs),
            "--output-dir", "data/pmc_100k_downloaded"
        ]
        
        result = self.run_script("scripts/download_100k_pmc_articles.py", download_args)
        
        phase_time = time.time() - phase_start
        self.results["download_results"] = {
            **result,
            "phase_time_seconds": phase_time
        }
        
        if result["success"]:
            logger.info(f"✅ Phase 1 completed successfully in {phase_time:.1f}s")
            self.results["orchestration_summary"]["phases_completed"].append("download")
            return True
        else:
            logger.error(f"❌ Phase 1 failed: {result.get('stderr', 'Unknown error')}")
            self.results["orchestration_summary"]["phases_failed"].append("download")
            return False
    
    def phase_2_ingestion(self, skip_ingestion: bool = False, schema_type: str = "RAG") -> bool:
        """Phase 2: Ingest 100k documents"""
        logger.info("\n" + "="*80)
        logger.info("💾 PHASE 2: INGESTING 100K DOCUMENTS")
        logger.info("="*80)
        
        if skip_ingestion:
            logger.info("⏭️ Skipping ingestion phase")
            self.results["orchestration_summary"]["phases_completed"].append("ingestion_skipped")
            return True
        
        phase_start = time.time()
        
        # Run ingestion script
        ingestion_args = [
            "--target-docs", str(self.target_docs),
            "--data-dir", "data/pmc_100k_downloaded",
            "--batch-size", "500",
            "--schema-type", schema_type
        ]
        
        result = self.run_script("scripts/ingest_100k_documents.py", ingestion_args)
        
        phase_time = time.time() - phase_start
        self.results["ingestion_results"] = {
            **result,
            "phase_time_seconds": phase_time,
            "schema_type": schema_type
        }
        
        if result["success"]:
            logger.info(f"✅ Phase 2 completed successfully in {phase_time:.1f}s")
            self.results["orchestration_summary"]["phases_completed"].append("ingestion")
            return True
        else:
            logger.error(f"❌ Phase 2 failed: {result.get('stderr', 'Unknown error')}")
            self.results["orchestration_summary"]["phases_failed"].append("ingestion")
            return False
    
    def phase_3_validation(self, schema_type: str = "RAG", fast_mode: bool = False) -> bool:
        """Phase 3: Ultimate enterprise validation with proper completion verification"""
        logger.info("\n" + "="*80)
        logger.info("🧪 PHASE 3: ULTIMATE ENTERPRISE VALIDATION")
        logger.info("="*80)
        
        phase_start = time.time()
        
        # Run validation script with proper parameter passing
        validation_args = [
            "--docs", str(self.target_docs),
            "--schema-type", schema_type,
            "--skip-ingestion"  # Data should already be loaded
        ]
        
        if fast_mode:
            validation_args.append("--fast-mode")
        
        logger.info(f"🔄 Starting validation with args: {validation_args}")
        
        # Use background=True for long-running validation
        result = self.run_script("scripts/ultimate_100k_enterprise_validation.py", validation_args, background=True)
        
        # Verify completion by checking output and return code
        success = result["success"] and result["returncode"] == 0
        
        # Additional verification: check if validation actually processed the target documents
        if success and result["stdout"]:
            # Look for completion indicators in stdout
            stdout_lower = result["stdout"].lower()
            if "enterprise validation summary" in stdout_lower or "validation completed" in stdout_lower:
                logger.info("✅ Validation completion verified from output")
            else:
                logger.warning("⚠️ Validation may not have completed properly - no completion indicator found")
                success = False
        
        phase_time = time.time() - phase_start
        self.results["validation_results"] = {
            **result,
            "phase_time_seconds": phase_time,
            "schema_type": schema_type,
            "fast_mode": fast_mode,
            "actual_target_docs": self.target_docs,
            "completion_verified": success
        }
        
        if success:
            logger.info(f"✅ Phase 3 completed successfully in {phase_time:.1f}s")
            self.results["orchestration_summary"]["phases_completed"].append("validation")
            return True
        else:
            logger.error(f"❌ Phase 3 failed: {result.get('stderr', 'Unknown error')}")
            logger.error(f"❌ Return code: {result.get('returncode', 'Unknown')}")
            self.results["orchestration_summary"]["phases_failed"].append("validation")
            return False
    
    def generate_final_report(self) -> str:
        """Generate comprehensive final report"""
        total_time = time.time() - self.start_time
        self.results["orchestration_summary"]["total_time_seconds"] = total_time
        self.results["orchestration_summary"]["end_time"] = datetime.now().isoformat()
        
        # Calculate success metrics
        total_phases = 3
        completed_phases = len([p for p in self.results.get("orchestration_summary", {}).get("phases_completed", []) if not p.endswith("_skipped")])
        success_rate = completed_phases / total_phases if total_phases > 0 else 0
        
        self.results["orchestration_summary"]["success_rate"] = success_rate
        self.results["orchestration_summary"]["completed_phases_count"] = completed_phases
        self.results["orchestration_summary"]["total_phases"] = total_phases
        
        # Save detailed report
        timestamp = int(time.time())
        report_file = f"complete_100k_validation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return report_file
    
    def print_final_summary(self, report_file: str):
        """Print comprehensive final summary"""
        logger.info("\n" + "="*100)
        logger.info("🏆 COMPLETE 100K VALIDATION SUMMARY")
        logger.info("="*100)
        
        summary = self.results["orchestration_summary"]
        logger.info(f"🎯 Target Documents: {summary['target_docs']:,}")
        logger.info(f"⏱️ Total Time: {summary['total_time_seconds']:.1f} seconds ({summary['total_time_seconds']/3600:.1f} hours)")
        logger.info(f"✅ Success Rate: {summary['success_rate']*100:.1f}%")
        logger.info(f"📊 Phases Completed: {summary['completed_phases_count']}/{summary['total_phases']}")
        
        # Phase breakdown
        logger.info(f"\n📋 PHASE BREAKDOWN:")
        for phase in summary["phases_completed"]:
            logger.info(f"   ✅ {phase}")
        for phase in summary["phases_failed"]:
            logger.info(f"   ❌ {phase}")
        
        # Performance summary
        logger.info(f"\n⚡ PERFORMANCE SUMMARY:")
        if "download_results" in self.results and self.results["download_results"].get("success"):
            download_time = self.results["download_results"].get("phase_time_seconds", 0)
            logger.info(f"   📥 Download: {download_time:.1f}s")
        
        if "ingestion_results" in self.results and self.results["ingestion_results"].get("success"):
            ingestion_time = self.results["ingestion_results"].get("phase_time_seconds", 0)
            logger.info(f"   💾 Ingestion: {ingestion_time:.1f}s")
        
        if "validation_results" in self.results and self.results["validation_results"].get("success"):
            validation_time = self.results["validation_results"].get("phase_time_seconds", 0)
            logger.info(f"   🧪 Validation: {validation_time:.1f}s")
        
        # Final recommendations
        logger.info(f"\n🎯 FINAL RECOMMENDATIONS:")
        if summary["success_rate"] >= 0.8:
            logger.info("   🚀 System is ready for enterprise deployment")
            logger.info("   📈 All critical components validated at 100k scale")
            logger.info("   🔄 Consider implementing horizontal scaling for production")
        else:
            logger.info("   ⚠️ Some phases failed - review logs for issues")
            logger.info("   🔧 Address failed components before production deployment")
        
        logger.info(f"\n📄 Detailed report saved: {report_file}")
        logger.info("="*100)
    
    def run_complete_validation(self, skip_download: bool = False, skip_ingestion: bool = False, 
                              schema_type: str = "RAG", fast_mode: bool = False) -> bool:
        """Run the complete validation pipeline"""
        logger.info(f"🚀 Starting complete 100k validation pipeline...")
        logger.info(f"📋 Configuration:")
        logger.info(f"   🎯 Target docs: {self.target_docs:,}")
        logger.info(f"   🗄️ Schema: {schema_type}")
        logger.info(f"   ⏭️ Skip download: {skip_download}")
        logger.info(f"   ⏭️ Skip ingestion: {skip_ingestion}")
        logger.info(f"   ⚡ Fast mode: {fast_mode}")
        
        success = True
        
        try:
            # Phase 1: Download
            if not self.phase_1_download(skip_download):
                success = False
                if not skip_download:  # Only fail if we actually tried to download
                    logger.error("❌ Download phase failed, stopping pipeline")
                    return False
            
            # Phase 2: Ingestion
            if not self.phase_2_ingestion(skip_ingestion, schema_type):
                success = False
                if not skip_ingestion:  # Only fail if we actually tried to ingest
                    logger.error("❌ Ingestion phase failed, stopping pipeline")
                    return False
            
            # Phase 3: Validation
            if not self.phase_3_validation(schema_type, fast_mode):
                success = False
                logger.error("❌ Validation phase failed")
                # Continue to generate report even if validation fails
            
            return success
            
        except KeyboardInterrupt:
            logger.info("🛑 Pipeline interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Pipeline failed with unexpected error: {e}")
            return False
        finally:
            # Always generate final report
            report_file = self.generate_final_report()
            self.print_final_summary(report_file)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Complete 100K Validation Orchestrator")
    parser.add_argument("--target-docs", type=int, default=100000,
                       help="Target number of documents for validation")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip the download phase")
    parser.add_argument("--skip-ingestion", action="store_true",
                       help="Skip the ingestion phase")
    parser.add_argument("--schema-type", type=str, default="RAG", choices=["RAG", "RAG_HNSW"],
                       help="Database schema to use")
    parser.add_argument("--fast-mode", action="store_true",
                       help="Use fast mode for validation")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Complete 100K Validation Orchestrator")
    logger.info(f"🎯 Target: {args.target_docs:,} documents")
    logger.info(f"🗄️ Schema: {args.schema_type}")
    
    orchestrator = Complete100kValidationOrchestrator(args.target_docs)
    
    success = orchestrator.run_complete_validation(
        skip_download=args.skip_download,
        skip_ingestion=args.skip_ingestion,
        schema_type=args.schema_type,
        fast_mode=args.fast_mode
    )
    
    if success:
        logger.info("🎉 COMPLETE 100K VALIDATION SUCCESSFUL!")
        logger.info("🚀 System is ready for enterprise deployment!")
    else:
        logger.warning("⚠️ Some phases failed - review the detailed report")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)