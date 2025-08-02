#!/usr/bin/env python3
"""
Cleanup Performance Optimization Files

This script organizes the repository after the successful performance optimization,
moving temporary investigation files to archive and keeping only the essential files.
"""

import os
import shutil
from datetime import datetime

def cleanup_repository():
    """Clean up repository after performance optimization."""
    print("🧹 CLEANING UP REPOSITORY AFTER PERFORMANCE OPTIMIZATION")
    print("=" * 60)
    
    # Create archive directory if it doesn't exist
    archive_dir = "archive/performance_investigation"
    os.makedirs(archive_dir, exist_ok=True)
    
    # Files to archive (temporary investigation files)
    files_to_archive = [
        # Investigation and analysis files
        "analyze_actual_corruption.py",
        "analyze_column_mismatch.py", 
        "analyze_embedding_integrity.py",
        "analyze_token_embeddings.py",
        "detailed_token_embedding_analysis.py",
        "investigate_colbert_tables.py",
        "investigate_data_samples.py",
        "investigate_real_database_state.py",
        "list_error_investigation.py",
        
        # Emergency fix files (no longer needed)
        "apply_conservative_fix.py",
        "emergency_database_cleanup.py",
        "emergency_list_error_fix.py",
        "emergency_recovery_with_list_error_fix.py",
        "simple_emergency_cleanup.py",
        "fix_actual_corruption.py",
        "fix_column_mismatch.py",
        
        # Temporary validation files
        "validate_column_fix.py",
        "verify_colbert_fix.py",
        "verify_token_embeddings_fix.py",
        "post_cleanup_verification.py",
        "comprehensive_integrity_check.py",
        
        # Backup and recovery files
        "backfill_token_embeddings.py",
        "complete_fresh_start_fixed.py",
        "complete_recovery_process.py",
        "database_recovery_orchestrator.py",
        "fresh_start_complete.py",
        "simple_fresh_start.py",
        
        # Test files for specific fixes
        "test_background_ingestion_fix.py",
        "test_emergency_list_error_fix.py",
        "test_fresh_start.py",
        "test_list_error_fix.py",
        
        # Temporary monitoring files
        "monitor_fresh_start.py",
        "monitor_token_embeddings.py",
        "quick_token_check.py",
        "safe_token_check.py",
        "simple_data_check.py",
        "check_cleanup_progress.py",
        
        # JSON reports and logs from investigation
        "database_integrity_report_20250527_164608.json",
        "embedding_integrity_analysis.json",
        "embedding_integrity_assessment_20250527_124713.json",
        "list_error_investigation_20250527_164656.json",
        "simple_list_error_check_20250527_164722.json",
        "emergency_recovery_checkpoint.json",
        "token_embedding_backfill_analysis.json",
        
        # Temporary ingestion files
        "run_conservative_ingestion.py",
        "run_fresh_ingestion.py",
        "simple_performance_fix.py",
        
        # Old performance investigation
        "fix_performance_degradation.py",  # Superseded by add_performance_indexes.py
    ]
    
    # Files to keep in root (essential performance optimization files)
    files_to_keep = [
        "add_performance_indexes.py",
        "validate_index_performance.py", 
        "monitor_index_performance_improvements.py",
        "investigate_performance_degradation.py",  # Keep as diagnostic tool
    ]
    
    # Archive temporary files
    archived_count = 0
    for filename in files_to_archive:
        if os.path.exists(filename):
            try:
                shutil.move(filename, os.path.join(archive_dir, filename))
                print(f"📦 Archived: {filename}")
                archived_count += 1
            except Exception as e:
                print(f"❌ Error archiving {filename}: {e}")
    
    # Archive old log files (keep recent ones)
    log_files_to_archive = [
        "emergency_recovery.log",
        "performance_fix_20250527_163523.log",
        "optimized_ingestion_output.log",
    ]
    
    for log_file in log_files_to_archive:
        if os.path.exists(log_file):
            try:
                shutil.move(log_file, os.path.join(archive_dir, log_file))
                print(f"📦 Archived log: {log_file}")
                archived_count += 1
            except Exception as e:
                print(f"❌ Error archiving {log_file}: {e}")
    
    # Archive old markdown files that are superseded
    old_docs_to_archive = [
        "EMERGENCY_LIST_ERROR_FIX_COMPLETE.md",
        "phase1_fix_status_report.md",
    ]
    
    for doc_file in old_docs_to_archive:
        if os.path.exists(doc_file):
            try:
                shutil.move(doc_file, os.path.join(archive_dir, doc_file))
                print(f"📦 Archived doc: {doc_file}")
                archived_count += 1
            except Exception as e:
                print(f"❌ Error archiving {doc_file}: {e}")
    
    print(f"\n📊 CLEANUP SUMMARY:")
    print(f"   📦 Files archived: {archived_count}")
    print(f"   📁 Archive location: {archive_dir}")
    print(f"   ✅ Essential performance files kept in root")
    
    # List essential files kept
    print(f"\n🔧 ESSENTIAL PERFORMANCE FILES KEPT:")
    for filename in files_to_keep:
        if os.path.exists(filename):
            print(f"   ✅ {filename}")
    
    # Create archive README
    archive_readme = os.path.join(archive_dir, "README.md")
    with open(archive_readme, 'w') as f:
        f.write(f"""# Performance Investigation Archive

This directory contains files from the performance optimization investigation completed on {datetime.now().strftime('%Y-%m-%d')}.

## Investigation Summary

A severe ingestion performance degradation was successfully diagnosed and resolved through strategic database index optimization:

- **Problem**: Batch timing increased from 1.6s to 65+ seconds (3,895% degradation)
- **Root Cause**: Missing indexes on token embedding table with 409K+ records
- **Solution**: Added 3 critical performance indexes
- **Result**: 1.6x-2.6x speedup achieved, ingestion "much faster"

## Files Archived

These files were used during the investigation and are preserved for reference:

### Investigation and Analysis
- Various `analyze_*.py` and `investigate_*.py` scripts
- JSON reports with detailed analysis results

### Emergency Fixes and Recovery
- Emergency cleanup and recovery scripts
- Backup and restoration utilities

### Temporary Validation
- Test scripts for specific fixes
- Validation and verification utilities

### Logs and Reports
- Investigation logs and performance reports
- JSON analysis results

## Current Solution

The active performance optimization is implemented in:
- `add_performance_indexes.py` - Creates critical indexes
- `validate_index_performance.py` - Validates effectiveness  
- `monitor_index_performance_improvements.py` - Real-time monitoring

See [INGESTION_PERFORMANCE_OPTIMIZATION.md](../../docs/INGESTION_PERFORMANCE_OPTIMIZATION.md) for complete documentation.
""")
    
    print(f"   📝 Created archive README: {archive_readme}")
    
    return archived_count

def main():
    """Main cleanup function."""
    print(f"⏰ Cleanup started at: {datetime.now()}")
    
    archived_count = cleanup_repository()
    
    print(f"\n✅ Repository cleanup completed at: {datetime.now()}")
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Review the cleaned repository structure")
    print(f"   2. Commit the performance optimization changes")
    print(f"   3. Push to remote repository")
    print(f"   4. Continue monitoring ingestion performance")
    
    if archived_count > 0:
        print(f"\n📦 {archived_count} files have been archived and can be safely committed.")
    
    return archived_count > 0

if __name__ == "__main__":
    main()