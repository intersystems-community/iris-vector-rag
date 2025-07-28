"""
PMC XML Processor

This module provides functions for processing PMC XML files and extracting
relevant metadata (abstract, author, title, keywords).
"""

import os
import logging
import xml.etree.ElementTree as ET
from typing import Dict, Any, Generator
import time

logger = logging.getLogger(__name__)

def extract_pmc_metadata(xml_file_path: str) -> Dict[str, Any]:
    """
    Extract core metadata from a PMC XML file.
    
    Args:
        xml_file_path: Path to the PMC XML file
        
    Returns:
        Dictionary with extracted fields (pmc_id, title, abstract, authors, keywords)
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Extract PMC ID from filename or article-id elements
        pmc_id = os.path.basename(xml_file_path).replace('.xml', '')
        
        # Extract title
        title_elem = root.find(".//article-title")
        title = title_elem.text if title_elem is not None and title_elem.text else "Unknown Title"
        
        # Extract abstract
        abstract_elem = root.find(".//abstract")
        abstract = ""
        if abstract_elem is not None:
            # Concatenate all paragraph text in the abstract
            for p in abstract_elem.findall(".//p"):
                if p.text:
                    abstract += p.text + " "
                # Get any other text within inline elements
                for elem in p.findall(".//*"):
                    if elem.text and elem.tag not in ["xref", "sup"]:  # Skip reference markers
                        abstract += elem.text + " "
                        
        abstract = abstract.strip()
        
        # Extract authors
        authors = []
        contrib_group = root.find(".//contrib-group")
        if contrib_group is not None:
            for author in contrib_group.findall(".//contrib[@contrib-type='author']"):
                surname = author.find(".//surname")
                given_names = author.find(".//given-names")
                if surname is not None and given_names is not None:
                    authors.append(f"{given_names.text} {surname.text}")
        
        # Extract keywords
        keywords = []
        kwd_group = root.find(".//kwd-group")
        if kwd_group is not None:
            for kwd in kwd_group.findall(".//kwd"):
                if kwd.text:
                    keywords.append(kwd.text)
        
        # Create content by combining title, abstract, and other text
        content = f"{title}\n\n{abstract}"
        if authors:
            content += f"\n\nAuthors: {', '.join(authors)}"
        if keywords:
            content += f"\n\nKeywords: {', '.join(keywords)}"
        
        return {
            "pmc_id": pmc_id,
            "title": title,
            "content": content,
            "abstract": abstract,
            "authors": authors,
            "keywords": keywords,
            "metadata": {
                "source": "PMC",
                "file_path": xml_file_path,
                "pmc_id": pmc_id
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing {xml_file_path}: {e}")
        pmc_id = os.path.basename(xml_file_path).replace('.xml', '')
        return {
            "pmc_id": pmc_id,
            "title": "Error",
            "content": f"Failed to process: {str(e)}",
            "abstract": f"Failed to process: {str(e)}",
            "authors": [],
            "keywords": [],
            "metadata": {
                "source": "PMC",
                "file_path": xml_file_path,
                "pmc_id": pmc_id,
                "error": str(e)
            }
        }

def process_pmc_files(directory: str, limit: int = 1000) -> Generator[Dict[str, Any], None, None]:
    """
    Process PMC XML files from a directory, up to the specified limit.
    
    Args:
        directory: Directory containing PMC XML files
        limit: Maximum number of files to process
        
    Yields:
        Dictionary with extracted metadata for each file
    """
    processed = 0
    start_time = time.time()
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        logger.info(f"Walking directory: {root}, found {len(dirs)} dirs, {len(files)} files.")
        for file in files:
            if file.endswith('.xml'): # Process all XMLs, then check limit
                file_path = os.path.join(root, file)
                logger.debug(f"Found XML file: {file_path}")
                if processed >= limit:
                    logger.info(f"Limit of {limit} files reached. Stopping further processing.")
                    return # Exit the generator completely

                try:
                    logger.debug(f"Attempting to extract metadata from: {file_path}")
                    metadata = extract_pmc_metadata(file_path)
                    if metadata.get("title") == "Error" and "Failed to process" in metadata.get("abstract", ""):
                        logger.warning(f"Metadata extraction indicated an error for {file_path}. Skipping yield.")
                        # Optionally, we could still increment 'processed' if we count attempts
                        # For now, only successfully yielded docs count towards 'processed'
                    else:
                        yield metadata
                        processed += 1
                        logger.debug(f"Successfully processed and yielded metadata for: {file_path} (Total processed: {processed})")
                        if processed % 100 == 0 and processed > 0: # Ensure processed > 0 for division
                            elapsed = time.time() - start_time
                            if elapsed > 0: # Avoid division by zero
                                logger.info(f"Processed {processed} files in {elapsed:.2f}s ({processed/elapsed:.2f} files/s)")
                            else:
                                logger.info(f"Processed {processed} files very quickly.")
                except Exception as e:
                    logger.error(f"Outer exception processing {file_path}: {e}")
            else:
                logger.debug(f"Skipping non-XML file: {file} in {root}")
        
        # This break was inside the files loop, potentially exiting the os.walk prematurely for the current root.
        # Moved the limit check to be more robust at the start of processing each file.
        # if processed >= limit:
        #     logger.info(f"Reached limit of {limit} files after processing directory {root}")
        #     break # This break only breaks the inner files loop for the current root.
                     # The main 'return' above handles exiting the generator.
    
    if processed < limit:
        logger.info(f"Finished walking all directories. Processed {processed} files (limit was {limit}).")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Sample usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pmc_processor.py <pmc_directory> [limit]")
        sys.exit(1)
    
    pmc_dir = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"Processing up to {limit} PMC files from {pmc_dir}")
    
    for i, doc in enumerate(process_pmc_files(pmc_dir, limit)):
        print(f"Document {i+1}: {doc['pmc_id']}")
        print(f"  Title: {doc['title'][:60]}...")
        print(f"  Abstract: {doc['abstract'][:100]}...")
        print(f"  Authors: {', '.join(doc['authors'][:3])}{' and more' if len(doc['authors']) > 3 else ''}")
        print(f"  Keywords: {', '.join(doc['keywords'][:5])}{' and more' if len(doc['keywords']) > 5 else ''}")
        print()
