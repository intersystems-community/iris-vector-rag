#!/usr/bin/env python3
"""
Debug script to test PMC FTP connection and explore available files.
"""

import ftplib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pmc_ftp_connection():
    """Test connection to PMC FTP and explore directory structure."""
    
    FTP_HOST = "ftp.ncbi.nlm.nih.gov"
    
    try:
        logger.info(f"Connecting to {FTP_HOST}...")
        ftp = ftplib.FTP(FTP_HOST)
        ftp.login()
        
        logger.info("✅ Successfully connected to PMC FTP")
        
        # Test different potential paths
        test_paths = [
            "/pub/pmc/oa_bulk",
            "/pub/pmc",
            "/pub/pmc/oa_package",
            "/pub/pmc/oa_comm",
            "/pub/pmc/oa_noncomm"
        ]
        
        for path in test_paths:
            try:
                logger.info(f"\n🔍 Testing path: {path}")
                ftp.cwd("/")  # Reset to root
                ftp.cwd(path)
                
                files = ftp.nlst()
                logger.info(f"✅ Path exists! Found {len(files)} items")
                
                # Show first 10 files
                for i, filename in enumerate(files[:10]):
                    try:
                        size = ftp.size(filename)
                        size_mb = size / (1024 * 1024) if size else 0
                        logger.info(f"  {filename} ({size_mb:.1f} MB)")
                    except:
                        logger.info(f"  {filename} (size unknown)")
                
                if len(files) > 10:
                    logger.info(f"  ... and {len(files) - 10} more files")
                    
                # Look for tar.gz files specifically
                tar_files = [f for f in files if f.endswith('.tar.gz')]
                if tar_files:
                    logger.info(f"📦 Found {len(tar_files)} .tar.gz files:")
                    for tar_file in tar_files[:5]:
                        logger.info(f"  {tar_file}")
                
            except Exception as e:
                logger.warning(f"❌ Path {path} not accessible: {e}")
        
        ftp.quit()
        logger.info("\n✅ FTP exploration complete")
        
    except Exception as e:
        logger.error(f"❌ FTP connection failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_pmc_ftp_connection()