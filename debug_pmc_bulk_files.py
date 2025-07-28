#!/usr/bin/env python3
"""
Debug script to explore PMC bulk file structure and find actual downloadable files.
"""

import ftplib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explore_bulk_directories():
    """Explore the bulk file directories to find actual tar.gz files."""
    
    FTP_HOST = "ftp.ncbi.nlm.nih.gov"
    
    try:
        logger.info(f"Connecting to {FTP_HOST}...")
        ftp = ftplib.FTP(FTP_HOST)
        ftp.login()
        
        # Explore the bulk directories
        bulk_dirs = [
            "/pub/pmc/oa_bulk/oa_comm",
            "/pub/pmc/oa_bulk/oa_noncomm", 
            "/pub/pmc/oa_bulk/oa_other"
        ]
        
        all_bulk_files = []
        
        for bulk_dir in bulk_dirs:
            try:
                logger.info(f"\n🔍 Exploring: {bulk_dir}")
                ftp.cwd(bulk_dir)
                
                files = ftp.nlst()
                tar_files = [f for f in files if f.endswith('.tar.gz')]
                
                logger.info(f"✅ Found {len(tar_files)} .tar.gz files in {bulk_dir}")
                
                for tar_file in tar_files[:10]:  # Show first 10
                    try:
                        size = ftp.size(tar_file)
                        size_gb = size / (1024 * 1024 * 1024) if size else 0
                        logger.info(f"  📦 {tar_file} ({size_gb:.2f} GB)")
                        
                        all_bulk_files.append({
                            'filename': tar_file,
                            'directory': bulk_dir,
                            'full_path': f"{bulk_dir}/{tar_file}",
                            'size_bytes': size,
                            'size_gb': size_gb
                        })
                    except Exception as e:
                        logger.warning(f"  ❌ Could not get size for {tar_file}: {e}")
                        all_bulk_files.append({
                            'filename': tar_file,
                            'directory': bulk_dir,
                            'full_path': f"{bulk_dir}/{tar_file}",
                            'size_bytes': None,
                            'size_gb': 0
                        })
                
                if len(tar_files) > 10:
                    logger.info(f"  ... and {len(tar_files) - 10} more files")
                    
            except Exception as e:
                logger.warning(f"❌ Could not access {bulk_dir}: {e}")
        
        ftp.quit()
        
        # Summary
        logger.info(f"\n📊 SUMMARY:")
        logger.info(f"Total bulk files found: {len(all_bulk_files)}")
        
        if all_bulk_files:
            # Sort by size
            sized_files = [f for f in all_bulk_files if f['size_bytes']]
            if sized_files:
                sized_files.sort(key=lambda x: x['size_bytes'], reverse=True)
                logger.info(f"\n🏆 TOP 5 LARGEST FILES:")
                for i, file_info in enumerate(sized_files[:5], 1):
                    logger.info(f"  {i}. {file_info['filename']} ({file_info['size_gb']:.2f} GB)")
                    logger.info(f"     Path: {file_info['full_path']}")
        
        return all_bulk_files
        
    except Exception as e:
        logger.error(f"❌ FTP exploration failed: {e}")
        return []

if __name__ == "__main__":
    bulk_files = explore_bulk_directories()
    print(f"\nFound {len(bulk_files)} total bulk files")