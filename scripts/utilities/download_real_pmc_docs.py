#!/usr/bin/env python3
"""
Download Real PMC Documents

Downloads real PMC XML documents from PubMed Central using E-utilities API.
Works with existing data/loader_fixed.py infrastructure.
"""

import logging
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

import requests

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.loader_fixed import process_and_load_documents

logger = logging.getLogger(__name__)


class SimplePMCDownloader:
    """Simple PMC downloader using E-utilities API."""

    def __init__(self, download_dir: str = "data/downloaded_pmc_docs"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self.search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "PMC-Downloader/1.0 (Research-Evaluation)"}
        )

    def get_pmc_ids(self, count: int = 1000) -> List[str]:
        """Get PMC IDs for open access biomedical documents."""
        try:
            params = {
                "db": "pmc",
                "term": "open access[filter]",
                "retmax": count,
                "retmode": "xml",
                "tool": "evaluation_downloader",
                "email": "researcher@example.com",
            }

            logger.info(f"Searching for {count} open access PMC documents...")
            response = self.session.get(self.search_url, params=params)
            response.raise_for_status()

            root = ET.fromstring(response.content)
            id_list = root.find("IdList")

            if id_list is None:
                logger.error("No PMC IDs found")
                return []

            pmc_ids = [id_elem.text for id_elem in id_list.findall("Id")]
            logger.info(f"Found {len(pmc_ids)} PMC IDs")

            time.sleep(0.5)  # Be nice to NCBI
            return pmc_ids

        except Exception as e:
            logger.error(f"Error getting PMC IDs: {e}")
            return []

    def download_pmc_documents(self, pmc_ids: List[str], batch_size: int = 10) -> int:
        """Download PMC documents in small batches."""
        downloaded_count = 0
        total_batches = (len(pmc_ids) + batch_size - 1) // batch_size

        logger.info(f"Downloading {len(pmc_ids)} documents in {total_batches} batches")

        for i in range(0, len(pmc_ids), batch_size):
            batch = pmc_ids[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            logger.info(
                f"Batch {batch_num}/{total_batches}: downloading {len(batch)} documents"
            )

            for pmc_id in batch:
                try:
                    xml_path = self.download_single_pmc(pmc_id)
                    if xml_path:
                        downloaded_count += 1
                        if downloaded_count % 50 == 0:
                            logger.info(
                                f"Downloaded {downloaded_count} documents so far"
                            )

                    time.sleep(0.4 + random.uniform(0, 0.2))  # Rate limiting

                except Exception as e:
                    logger.warning(f"Error downloading PMC{pmc_id}: {e}")

            # Pause between batches
            time.sleep(1)

        return downloaded_count

    def download_single_pmc(self, pmc_id: str) -> str:
        """Download a single PMC document."""
        xml_filename = f"PMC{pmc_id}.xml"
        xml_path = self.download_dir / xml_filename

        # Skip if already exists
        if xml_path.exists():
            return str(xml_path)

        try:
            params = {
                "db": "pmc",
                "id": pmc_id,
                "retmode": "xml",
                "tool": "evaluation_downloader",
                "email": "researcher@example.com",
            }

            response = self.session.get(self.fetch_url, params=params, timeout=30)
            response.raise_for_status()

            # Save XML content
            with open(xml_path, "wb") as f:
                f.write(response.content)

            # Quick validation
            if self._is_valid_pmc_xml(xml_path):
                logger.debug(f"Downloaded PMC{pmc_id}")
                return str(xml_path)
            else:
                xml_path.unlink()
                return None

        except Exception as e:
            logger.warning(f"Failed to download PMC{pmc_id}: {e}")
            return None

    def _is_valid_pmc_xml(self, xml_path: Path) -> bool:
        """Quick validation of PMC XML."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Check for basic PMC structure
            has_title = root.find(".//article-title") is not None
            has_content = len(ET.tostring(root, encoding="unicode")) > 500

            return has_title and has_content

        except:
            return False


def download_and_load_1000_pmc_docs() -> Dict[str, Any]:
    """Download and load 1000+ real PMC documents using existing infrastructure."""
    start_time = time.time()

    logger.info("🚀 Starting download of 1000+ real PMC documents")

    downloader = SimplePMCDownloader()

    # Check existing downloads
    existing_files = list(downloader.download_dir.glob("*.xml"))
    existing_count = len(existing_files)
    logger.info(f"Found {existing_count} existing downloaded files")

    # Determine how many more we need
    target_count = 1000

    if existing_count < target_count:
        needed = target_count - existing_count
        logger.info(f"Need to download {needed} more documents")

        # Get PMC IDs (get extra in case some fail)
        pmc_ids = downloader.get_pmc_ids(needed + 200)

        if pmc_ids:
            # Download documents
            new_downloads = downloader.download_pmc_documents(pmc_ids[:needed])
            logger.info(f"Successfully downloaded {new_downloads} new documents")
        else:
            logger.error("Failed to get PMC IDs")
            return {"success": False, "error": "Failed to get PMC IDs"}

    # Load all documents using existing infrastructure
    logger.info("Loading documents into database using existing loader...")
    load_result = process_and_load_documents(
        str(downloader.download_dir), limit=target_count
    )

    duration = time.time() - start_time

    return {
        "success": True,
        "download_dir": str(downloader.download_dir),
        "total_xml_files": len(list(downloader.download_dir.glob("*.xml"))),
        "load_result": load_result,
        "duration_minutes": duration / 60,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    result = download_and_load_1000_pmc_docs()

    if result["success"]:
        load_stats = result["load_result"]
        print("=" * 80)
        print("✅ PMC DOWNLOAD AND LOAD COMPLETE!")
        print(f"📁 Download Directory: {result['download_dir']}")
        print(f"📊 XML Files Downloaded: {result['total_xml_files']}")
        print(f"📈 Documents Loaded: {load_stats.get('loaded_doc_count', 0)}")
        print(f"📦 Chunks Created: {load_stats.get('loaded_chunk_count', 0)}")
        print(f"⏱️ Total Time: {result['duration_minutes']:.1f} minutes")
        print("=" * 80)
    else:
        print(f"❌ Error: {result.get('error')}")
        sys.exit(1)
