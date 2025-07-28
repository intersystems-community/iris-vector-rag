#!/usr/bin/env python3
"""
Debug script to explore PMC oa_package structure - the new way PMC distributes documents.
"""

import ftplib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explore_package_structure():
    """Explore the oa_package directory structure."""
    
    FTP_HOST = "ftp.ncbi.nlm.nih.gov"
    
    try:
        logger.info(f"Connecting to {FTP_HOST}...")
        ftp = ftplib.FTP(FTP_HOST)
        ftp.login()
        
        # Explore oa_package structure
        package_dir = "/pub/pmc/oa_package"
        logger.info(f"\n🔍 Exploring: {package_dir}")
        ftp.cwd(package_dir)
        
        # Get subdirectories (should be numbered 00, 01, 02, etc.)
        subdirs = ftp.nlst()
        logger.info(f"Found {len(subdirs)} subdirectories")
        
        # Explore first few subdirectories
        sample_dirs = subdirs[:5]  # Look at first 5
        
        total_files = 0
        for subdir in sample_dirs:
            try:
                logger.info(f"\n📁 Exploring subdirectory: {subdir}")
                ftp.cwd(f"{package_dir}/{subdir}")
                
                files = ftp.nlst()
                tar_files = [f for f in files if f.endswith('.tar.gz')]
                xml_files = [f for f in files if f.endswith('.xml')]
                
                logger.info(f"  📦 .tar.gz files: {len(tar_files)}")
                logger.info(f"  📄 .xml files: {len(xml_files)}")
                logger.info(f"  📋 Total files: {len(files)}")
                
                total_files += len(files)
                
                # Show sample files
                if tar_files:
                    logger.info(f"  Sample .tar.gz files:")
                    for tar_file in tar_files[:3]:
                        try:
                            size = ftp.size(tar_file)
                            size_mb = size / (1024 * 1024) if size else 0
                            logger.info(f"    {tar_file} ({size_mb:.1f} MB)")
                        except:
                            logger.info(f"    {tar_file} (size unknown)")
                
                if xml_files:
                    logger.info(f"  Sample .xml files:")
                    for xml_file in xml_files[:3]:
                        logger.info(f"    {xml_file}")
                        
            except Exception as e:
                logger.warning(f"❌ Could not access {subdir}: {e}")
        
        # Estimate total scale
        avg_files_per_dir = total_files / len(sample_dirs) if sample_dirs else 0
        estimated_total = avg_files_per_dir * len(subdirs)
        
        logger.info(f"\n📊 SUMMARY:")
        logger.info(f"Total subdirectories: {len(subdirs)}")
        logger.info(f"Sampled directories: {len(sample_dirs)}")
        logger.info(f"Files in sampled dirs: {total_files}")
        logger.info(f"Average files per directory: {avg_files_per_dir:.1f}")
        logger.info(f"Estimated total files: {estimated_total:.0f}")
        
        ftp.quit()
        return True
        
    except Exception as e:
        logger.error(f"❌ FTP exploration failed: {e}")
        return False

if __name__ == "__main__":
    explore_package_structure()