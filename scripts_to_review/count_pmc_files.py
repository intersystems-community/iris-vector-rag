#!/usr/bin/env python
"""
Count PMC XML files in the data directory.
This script helps verify that we have enough real PMC documents for testing.
"""

import os
import logging
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_pmc_files():
    """Count the XML files in the PMC data directory"""
    # Path to PMC documents
    pmc_dir = os.path.join(os.getcwd(), "data", "pmc_oas_downloaded")
    logger.info(f"Looking for PMC files in: {os.path.abspath(pmc_dir)}")
    
    # Count XML files
    xml_files = []
    for dirpath, dirnames, filenames in os.walk(pmc_dir):
        for filename in filenames:
            if filename.endswith('.xml'):
                xml_files.append(os.path.join(dirpath, filename))
    
    # Print results
    logger.info(f"Found {len(xml_files)} PMC XML files")
    
    # Show directory structure
    logger.info("Directory structure:")
    all_dirs = set()
    for dirpath, dirnames, _ in os.walk(pmc_dir):
        all_dirs.add(dirpath)
    
    for d in sorted(all_dirs):
        count = len(glob.glob(os.path.join(d, "*.xml")))
        logger.info(f"  {d}: {count} XML files")
    
    # Show some sample files
    if xml_files:
        logger.info("Sample XML files:")
        for file in xml_files[:5]:
            logger.info(f"  {file}")
    
    return len(xml_files)

if __name__ == "__main__":
    count = count_pmc_files()
    print(f"\nTotal PMC XML files: {count}")
    
    # Also check if the conftest_real_pmc.py is properly set up
    try:
        conftest_path = os.path.join("tests", "conftest_real_pmc.py")
        with open(conftest_path, 'r') as f:
            content = f.read()
            if "pmc_oas_downloaded" in content and "SourceDocuments" in content:
                print("✅ conftest_real_pmc.py appears to be properly set up to load PMC documents")
            else:
                print("❌ conftest_real_pmc.py may not be correctly set up to load PMC documents")
    except Exception as e:
        print(f"❌ Error checking conftest_real_pmc.py: {e}")
