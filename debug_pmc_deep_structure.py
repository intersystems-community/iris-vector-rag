#!/usr/bin/env python3
"""
Debug script to explore the deep nested structure of PMC oa_package.
"""

import ftplib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explore_deep_structure():
    """Explore the deep nested structure to find actual PMC files."""
    
    FTP_HOST = "ftp.ncbi.nlm.nih.gov"
    
    try:
        logger.info(f"Connecting to {FTP_HOST}...")
        ftp = ftplib.FTP(FTP_HOST)
        ftp.login()
        
        # Go deeper: /pub/pmc/oa_package/00/00/
        test_path = "/pub/pmc/oa_package/00/00"
        logger.info(f"\n🔍 Exploring deeper: {test_path}")
        
        try:
            ftp.cwd(test_path)
            files = ftp.nlst()
            logger.info(f"Found {len(files)} items in {test_path}")
            
            # Check file types
            tar_files = [f for f in files if f.endswith('.tar.gz')]
            xml_files = [f for f in files if f.endswith('.xml')]
            dirs = [f for f in files if not '.' in f]
            
            logger.info(f"📦 .tar.gz files: {len(tar_files)}")
            logger.info(f"📄 .xml files: {len(xml_files)}")
            logger.info(f"📁 Directories: {len(dirs)}")
            
            # Show actual files
            if tar_files:
                logger.info(f"\n✅ FOUND .tar.gz FILES!")
                for i, tar_file in enumerate(tar_files[:10]):
                    try:
                        size = ftp.size(tar_file)
                        size_kb = size / 1024 if size else 0
                        logger.info(f"  {i+1}. {tar_file} ({size_kb:.1f} KB)")
                    except Exception as e:
                        logger.info(f"  {i+1}. {tar_file} (size error: {e})")
            
            if xml_files:
                logger.info(f"\n✅ FOUND .xml FILES!")
                for i, xml_file in enumerate(xml_files[:5]):
                    try:
                        size = ftp.size(xml_file)
                        size_kb = size / 1024 if size else 0
                        logger.info(f"  {i+1}. {xml_file} ({size_kb:.1f} KB)")
                    except Exception as e:
                        logger.info(f"  {i+1}. {xml_file} (size error: {e})")
            
            # If still directories, go one more level
            if dirs and not tar_files and not xml_files:
                logger.info(f"\n🔍 Going one level deeper: {test_path}/{dirs[0]}")
                ftp.cwd(f"{test_path}/{dirs[0]}")
                deeper_files = ftp.nlst()
                
                deeper_tar = [f for f in deeper_files if f.endswith('.tar.gz')]
                deeper_xml = [f for f in deeper_files if f.endswith('.xml')]
                
                logger.info(f"Found {len(deeper_files)} items at deeper level")
                logger.info(f"📦 .tar.gz files: {len(deeper_tar)}")
                logger.info(f"📄 .xml files: {len(deeper_xml)}")
                
                if deeper_tar:
                    logger.info(f"\n✅ FOUND .tar.gz FILES AT DEEPER LEVEL!")
                    for i, tar_file in enumerate(deeper_tar[:5]):
                        try:
                            size = ftp.size(tar_file)
                            size_kb = size / 1024 if size else 0
                            logger.info(f"  {i+1}. {tar_file} ({size_kb:.1f} KB)")
                        except Exception as e:
                            logger.info(f"  {i+1}. {tar_file} (size error: {e})")
                
                return len(deeper_tar), f"{test_path}/{dirs[0]}"
            
            return len(tar_files), test_path
            
        except Exception as e:
            logger.warning(f"❌ Could not access {test_path}: {e}")
            return 0, test_path
        
    except Exception as e:
        logger.error(f"❌ FTP connection failed: {e}")
        return 0, ""
    
    finally:
        try:
            ftp.quit()
        except:
            pass

if __name__ == "__main__":
    tar_count, path = explore_deep_structure()
    print(f"\nResult: Found {tar_count} .tar.gz files at path: {path}")