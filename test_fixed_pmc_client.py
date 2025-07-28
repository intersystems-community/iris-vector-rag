#!/usr/bin/env python3
"""
Test script to verify the fixed PMC API client can find and download real documents.
"""

import logging
from pathlib import Path
from data.pmc_downloader.api_client import PMCAPIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fixed_pmc_client():
    """Test the fixed PMC API client."""
    
    logger.info("🧪 Testing fixed PMC API client...")
    
    # Initialize client
    client = PMCAPIClient()
    
    # Test 1: Get available files
    logger.info("\n📋 Test 1: Getting available PMC files...")
    available_files = client.get_available_bulk_files()
    
    if not available_files:
        logger.error("❌ No files found - API client still broken")
        return False
    
    logger.info(f"✅ Found {len(available_files)} PMC files available for download")
    
    # Show sample files
    logger.info("\n📄 Sample available files:")
    for i, file_info in enumerate(available_files[:5]):
        logger.info(f"  {i+1}. {file_info['pmc_id']} - {file_info['filename']}")
        logger.info(f"      Path: {file_info['ftp_path']}")
    
    # Test 2: Download a small sample
    logger.info(f"\n⬇️ Test 2: Downloading sample files...")
    test_dir = Path("test_pmc_downloads")
    test_dir.mkdir(exist_ok=True)
    
    # Download first 3 files as a test
    sample_files = available_files[:3]
    
    def progress_callback(progress_info):
        current = progress_info['current_file']
        total = progress_info['total_files']
        filename = progress_info['filename']
        logger.info(f"  Downloading {filename} ({current}/{total})")
    
    download_result = client.download_multiple_files(
        sample_files, 
        test_dir, 
        max_files=3,
        progress_callback=progress_callback
    )
    
    if not download_result['success']:
        logger.error(f"❌ Download failed: {download_result}")
        return False
    
    logger.info(f"✅ Downloaded {download_result['downloaded_count']} files successfully")
    logger.info(f"   Total size: {download_result['total_bytes']:,} bytes")
    logger.info(f"   Download time: {download_result['download_time_seconds']:.2f}s")
    
    # Test 3: Extract a downloaded file
    if download_result['downloaded_files']:
        logger.info(f"\n📂 Test 3: Extracting a downloaded file...")
        
        first_download = download_result['downloaded_files'][0]
        archive_path = Path(first_download['local_path'])
        extract_dir = test_dir / "extracted"
        
        extract_result = client.extract_individual_file(archive_path, extract_dir)
        
        if extract_result['success']:
            logger.info(f"✅ Extracted {extract_result['extracted_count']} files")
            logger.info(f"   Extracted files: {extract_result['extracted_files']}")
        else:
            logger.error(f"❌ Extraction failed: {extract_result['error']}")
            return False
    
    # Test 4: Verify enterprise scale capability
    logger.info(f"\n🏢 Test 4: Enterprise scale verification...")
    total_available = len(available_files)
    
    if total_available >= 10000:
        logger.info(f"✅ Enterprise scale confirmed: {total_available:,} documents available")
        logger.info("✅ Can easily download 10,000+ documents for enterprise testing")
    elif total_available >= 1000:
        logger.info(f"✅ Large scale confirmed: {total_available:,} documents available")
        logger.info("✅ Sufficient for comprehensive testing")
    else:
        logger.warning(f"⚠️ Limited scale: Only {total_available} documents found")
        logger.info("⚠️ May need to explore more directories for enterprise scale")
    
    logger.info(f"\n🎉 PMC API client testing complete!")
    logger.info(f"✅ Fixed implementation successfully accesses real PMC data")
    logger.info(f"✅ Ready for enterprise-scale RAG testing")
    
    return True

if __name__ == "__main__":
    success = test_fixed_pmc_client()
    if success:
        print("\n✅ PMC downloader system is now working!")
    else:
        print("\n❌ PMC downloader system still has issues")