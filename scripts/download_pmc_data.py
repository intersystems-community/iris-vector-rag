#!/usr/bin/env python3
"""
PMC Data Downloader

This script downloads additional PMC (PubMed Central) data to scale up to the full 92k archive.
It uses the NCBI E-utilities API to download PMC articles in bulk.

Usage:
    python scripts/download_pmc_data.py --target-count 50000
    python scripts/download_pmc_data.py --full-archive  # Downloads up to 92k
"""

import os
import sys
import logging
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from typing import List, Dict, Any
import json
import gzip
from urllib.parse import urlencode
import tarfile

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PMCDataDownloader:
    """Downloads PMC data from NCBI"""
    
    def __init__(self, output_dir: str = "data/pmc_oas_downloaded"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.pmc_ftp_base = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/"
        
        # Rate limiting for NCBI API (max 3 requests per second)
        self.last_request_time = 0
        self.min_request_interval = 0.34  # Slightly more than 1/3 second
        
    def rate_limit(self):
        """Enforce rate limiting for NCBI API"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def get_pmc_id_list(self, query: str = "open access[filter]", max_ids: int = 50000) -> List[str]:
        """Get list of PMC IDs using E-search"""
        logger.info(f"Searching for PMC articles with query: {query}")
        
        pmc_ids = []
        retstart = 0
        retmax = 10000  # Maximum allowed by NCBI
        
        while len(pmc_ids) < max_ids:
            self.rate_limit()
            
            # Build search URL
            params = {
                'db': 'pmc',
                'term': query,
                'retmode': 'json',
                'retmax': min(retmax, max_ids - len(pmc_ids)),
                'retstart': retstart,
                'tool': 'rag_templates',
                'email': 'research@example.com'  # Replace with actual email
            }
            
            url = self.base_url + "esearch.fcgi?" + urlencode(params)
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if 'esearchresult' in data and 'idlist' in data['esearchresult']:
                    batch_ids = data['esearchresult']['idlist']
                    pmc_ids.extend(batch_ids)
                    
                    logger.info(f"Retrieved {len(batch_ids)} PMC IDs (total: {len(pmc_ids)})")
                    
                    if len(batch_ids) < retmax:
                        # No more results
                        break
                        
                    retstart += retmax
                else:
                    logger.warning("No results found in E-search response")
                    break
                    
            except Exception as e:
                logger.error(f"Error in E-search: {e}")
                break
        
        logger.info(f"Found {len(pmc_ids)} total PMC IDs")
        return pmc_ids
    
    def download_pmc_articles(self, pmc_ids: List[str], batch_size: int = 200) -> int:
        """Download PMC articles using E-fetch"""
        logger.info(f"Downloading {len(pmc_ids)} PMC articles in batches of {batch_size}")
        
        downloaded_count = 0
        
        for i in range(0, len(pmc_ids), batch_size):
            batch_ids = pmc_ids[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(pmc_ids) + batch_size - 1) // batch_size
            
            logger.info(f"Downloading batch {batch_num}/{total_batches} ({len(batch_ids)} articles)")
            
            self.rate_limit()
            
            # Build fetch URL
            params = {
                'db': 'pmc',
                'id': ','.join(batch_ids),
                'retmode': 'xml',
                'tool': 'rag_templates',
                'email': 'research@example.com'  # Replace with actual email
            }
            
            url = self.base_url + "efetch.fcgi?" + urlencode(params)
            
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                # Parse the XML response
                root = ET.fromstring(response.content)
                
                # Extract individual articles
                articles = root.findall('.//article')
                
                for article in articles:
                    try:
                        # Extract PMC ID
                        pmc_id_elem = article.find('.//article-id[@pub-id-type="pmc"]')
                        if pmc_id_elem is not None:
                            pmc_id = pmc_id_elem.text
                            if not pmc_id.startswith('PMC'):
                                pmc_id = 'PMC' + pmc_id
                        else:
                            # Fallback to using batch ID
                            pmc_id = f"PMC{batch_ids[len(articles)]}" if len(articles) < len(batch_ids) else f"PMC{int(time.time())}"
                        
                        # Create directory structure
                        pmc_dir = self.output_dir / f"{pmc_id[:6]}xxxxxx"
                        pmc_dir.mkdir(exist_ok=True)
                        
                        # Save article XML
                        article_file = pmc_dir / f"{pmc_id}.xml"
                        
                        # Convert article element back to XML string
                        article_xml = ET.tostring(article, encoding='unicode')
                        
                        with open(article_file, 'w', encoding='utf-8') as f:
                            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                            f.write(article_xml)
                        
                        downloaded_count += 1
                        
                        if downloaded_count % 100 == 0:
                            logger.info(f"Downloaded {downloaded_count} articles...")
                            
                    except Exception as e:
                        logger.error(f"Error processing article: {e}")
                        continue
                
            except Exception as e:
                logger.error(f"Error downloading batch {batch_num}: {e}")
                continue
        
        logger.info(f"Downloaded {downloaded_count} PMC articles")
        return downloaded_count
    
    def download_oa_bulk_files(self) -> int:
        """Download Open Access bulk files from PMC FTP"""
        logger.info("Downloading Open Access bulk files from PMC FTP...")
        
        # PMC provides bulk downloads of Open Access articles
        bulk_files = [
            "oa_comm/xml/oa_comm_xml.PMC000xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC001xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC002xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC003xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC004xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC005xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC006xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC007xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC008xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC009xxxxxx.baseline.2023.tar.gz",
            "oa_comm/xml/oa_comm_xml.PMC010xxxxxx.baseline.2023.tar.gz",
        ]
        
        total_downloaded = 0
        
        for bulk_file in bulk_files:
            url = self.pmc_ftp_base + bulk_file
            filename = Path(bulk_file).name
            local_path = self.output_dir / filename
            
            if local_path.exists():
                logger.info(f"Skipping {filename} (already exists)")
                continue
            
            logger.info(f"Downloading {filename}...")
            
            try:
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()
                
                # Download with progress
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            if total_size > 0 and downloaded_size % (1024*1024*10) == 0:  # Every 10MB
                                progress = (downloaded_size / total_size) * 100
                                logger.info(f"Progress: {progress:.1f}% ({downloaded_size/(1024*1024):.1f}MB/{total_size/(1024*1024):.1f}MB)")
                
                logger.info(f"Downloaded {filename} ({downloaded_size/(1024*1024):.1f}MB)")
                
                # Extract the archive
                logger.info(f"Extracting {filename}...")
                with tarfile.open(local_path, 'r:gz') as tar:
                    tar.extractall(path=self.output_dir)
                
                # Count extracted files
                extracted_count = 0
                for member in tar.getmembers():
                    if member.name.endswith('.xml'):
                        extracted_count += 1
                
                total_downloaded += extracted_count
                logger.info(f"Extracted {extracted_count} XML files from {filename}")
                
                # Remove the archive to save space
                local_path.unlink()
                logger.info(f"Removed archive {filename}")
                
            except Exception as e:
                logger.error(f"Error downloading {filename}: {e}")
                continue
        
        logger.info(f"Total documents downloaded from bulk files: {total_downloaded}")
        return total_downloaded
    
    def get_current_document_count(self) -> int:
        """Count current XML documents"""
        count = len(list(self.output_dir.rglob("*.xml")))
        logger.info(f"Current document count: {count}")
        return count
    
    def download_to_target(self, target_count: int) -> int:
        """Download documents to reach target count"""
        current_count = self.get_current_document_count()
        
        if current_count >= target_count:
            logger.info(f"Target already reached: {current_count} >= {target_count}")
            return current_count
        
        needed = target_count - current_count
        logger.info(f"Need {needed} more documents to reach target of {target_count}")
        
        # First try bulk downloads (more efficient)
        if needed > 10000:
            logger.info("Attempting bulk download for large target...")
            bulk_downloaded = self.download_oa_bulk_files()
            current_count = self.get_current_document_count()
            
            if current_count >= target_count:
                logger.info(f"Target reached with bulk download: {current_count}")
                return current_count
        
        # If still need more, use E-utilities API
        remaining_needed = target_count - current_count
        if remaining_needed > 0:
            logger.info(f"Using E-utilities API to download {remaining_needed} more documents...")
            
            # Get PMC IDs
            pmc_ids = self.get_pmc_id_list(max_ids=remaining_needed)
            
            if pmc_ids:
                # Download articles
                downloaded = self.download_pmc_articles(pmc_ids)
                current_count = self.get_current_document_count()
        
        return current_count

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download PMC data for production scale testing")
    parser.add_argument("--target-count", type=int, default=50000,
                       help="Target number of documents to download")
    parser.add_argument("--full-archive", action="store_true",
                       help="Download full archive (up to 92k documents)")
    parser.add_argument("--output-dir", type=str, default="data/pmc_oas_downloaded",
                       help="Output directory for downloaded data")
    parser.add_argument("--bulk-only", action="store_true",
                       help="Only download bulk files (faster)")
    
    args = parser.parse_args()
    
    target_count = 92000 if args.full_archive else args.target_count
    
    logger.info(f"PMC Data Downloader - Target: {target_count} documents")
    logger.info(f"Output directory: {args.output_dir}")
    
    downloader = PMCDataDownloader(args.output_dir)
    
    if args.bulk_only:
        # Only download bulk files
        downloaded = downloader.download_oa_bulk_files()
        final_count = downloader.get_current_document_count()
    else:
        # Download to target
        final_count = downloader.download_to_target(target_count)
    
    logger.info("=" * 60)
    logger.info(f"🎉 Download complete!")
    logger.info(f"📊 Final document count: {final_count}")
    logger.info(f"🎯 Target was: {target_count}")
    
    if final_count >= target_count:
        logger.info("✅ Target reached successfully!")
    else:
        logger.info(f"⚠️  Target not fully reached (missing {target_count - final_count} documents)")
    
    return final_count >= target_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)