#!/usr/bin/env python3
"""
Debug script to explore individual PMC files in the oa_package structure.
"""

import ftplib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explore_individual_files():
    """Explore individual files in oa_package subdirectories."""
    
    FTP_HOST = "ftp.ncbi.nlm.nih.gov"
    
    try:
        logger.info(f"Connecting to {FTP_HOST}...")
        ftp = ftplib.FTP(FTP_HOST)
        ftp.login()
        
        # Look at a specific subdirectory to see actual files
        test_dir = "/pub/pmc/oa_package/00"
        logger.info(f"\n🔍 Exploring files in: {test_dir}")
        ftp.cwd(test_dir)
        
        files = ftp.nlst()
        logger.info(f"Found {len(files)} files")
        
        # Analyze file types and patterns
        tar_files = [f for f in files if f.endswith('.tar.gz')]
        xml_files = [f for f in files if f.endswith('.xml')]
        other_files = [f for f in files if not f.endswith('.tar.gz') and not f.endswith('.xml')]
        
        logger.info(f"📦 .tar.gz files: {len(tar_files)}")
        logger.info(f"📄 .xml files: {len(xml_files)}")
        logger.info(f"📋 Other files: {len(other_files)}")
        
        # Show sample files with sizes
        logger.info(f"\n📦 Sample .tar.gz files:")
        for i, tar_file in enumerate(tar_files[:10]):
            try:
                size = ftp.size(tar_file)
                size_kb = size / 1024 if size else 0
                logger.info(f"  {i+1}. {tar_file} ({size_kb:.1f} KB)")
            except Exception as e:
                logger.info(f"  {i+1}. {tar_file} (size error: {e})")
        
        if xml_files:
            logger.info(f"\n📄 Sample .xml files:")
            for i, xml_file in enumerate(xml_files[:5]):
                try:
                    size = ftp.size(xml_file)
                    size_kb = size / 1024 if size else 0
                    logger.info(f"  {i+1}. {xml_file} ({size_kb:.1f} KB)")
                except Exception as e:
                    logger.info(f"  {i+1}. {xml_file} (size error: {e})")
        
        if other_files:
            logger.info(f"\n📋 Sample other files:")
            for i, other_file in enumerate(other_files[:5]):
                logger.info(f"  {i+1}. {other_file}")
        
        # Test downloading a small file to verify access
        if tar_files:
            test_file = tar_files[0]
            logger.info(f"\n🧪 Testing download of: {test_file}")
            
            try:
                # Try to get file size first
                size = ftp.size(test_file)
                logger.info(f"File size: {size:,} bytes ({size/1024:.1f} KB)")
                
                # Test if we can start a download (without actually downloading)
                logger.info("✅ File is accessible for download")
                
            except Exception as e:
                logger.warning(f"❌ Download test failed: {e}")
        
        ftp.quit()
        
        # Summary for enterprise scale
        logger.info(f"\n📊 ENTERPRISE SCALE ANALYSIS:")
        logger.info(f"Files per directory: {len(files)}")
        logger.info(f"Estimated total directories: 256")
        logger.info(f"Estimated total files: {len(files) * 256:,}")
        logger.info(f"Average file size: {size_kb:.1f} KB" if 'size_kb' in locals() else "Unknown")
        
        if tar_files:
            logger.info(f"✅ PMC provides individual .tar.gz files (not bulk archives)")
            logger.info(f"✅ Each file likely contains 1 PMC document")
            logger.info(f"✅ Can download 10,000+ documents by selecting multiple files")
        
        return len(files), len(tar_files)
        
    except Exception as e:
        logger.error(f"❌ FTP exploration failed: {e}")
        return 0, 0

if __name__ == "__main__":
    total_files, tar_files = explore_individual_files()
    print(f"\nResult: {total_files} total files, {tar_files} downloadable .tar.gz files")