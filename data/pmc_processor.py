"""
PMC XML Processor

This module provides functions for processing PMC XML files and extracting
relevant metadata (abstract, author, title, keywords).
"""

import os
import logging
import xml.etree.ElementTree as ET
from typing import Dict, Any, Generator, List, Optional
import time

logger = logging.getLogger(__name__)

def _chunk_pmc_content(content: str, pmc_id: str, chunk_size: int = 8000, overlap: int = 400) -> List[Dict[str, Any]]:
    """
    Chunk PMC content into manageable pieces for LLM processing.
    
    Args:
        content: Full PMC content to chunk
        pmc_id: PMC document ID
        chunk_size: Target size for each chunk (characters)
        overlap: Overlap between chunks (characters)
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    if len(content) <= chunk_size:
        return [{
            "chunk_id": f"{pmc_id}_chunk_0",
            "text": content,
            "chunk_index": 0,
            "start_pos": 0,
            "end_pos": len(content),
            "metadata": {
                "is_complete_doc": True,
                "chunk_size": len(content)
            }
        }]
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(content):
        end = min(start + chunk_size, len(content))
        
        # Try to break at sentence boundaries to preserve context
        if end < len(content):
            # Look for sentence ending within last 20% of chunk
            search_start = max(start + int(chunk_size * 0.8), start + 200)
            sentence_end = _find_sentence_boundary(content, search_start, end)
            if sentence_end > search_start:
                end = sentence_end
        
        chunk_text = content[start:end].strip()
        
        if len(chunk_text) > 100:  # Only keep meaningful chunks
            chunks.append({
                "chunk_id": f"{pmc_id}_chunk_{chunk_index}",
                "text": chunk_text,
                "chunk_index": chunk_index,
                "start_pos": start,
                "end_pos": end,
                "metadata": {
                    "chunk_size": len(chunk_text),
                    "overlap_with_previous": min(overlap, start) if start > 0 else 0,
                    "strategy": "fixed_size_with_sentences"
                }
            })
            chunk_index += 1
        
        # Move start position with overlap, but ensure progress
        next_start = end - overlap
        if next_start <= start:
            # If overlap would prevent progress, move forward by at least 100 chars
            next_start = start + 100
        start = next_start
        
        # Prevent infinite loop
        if start >= len(content):
            break
    
    return chunks

def _find_sentence_boundary(text: str, start: int, end: int) -> int:
    """Find the best sentence boundary within the given range."""
    import re
    
    # Look for sentence endings (., !, ?) followed by space or end of text
    sentence_pattern = r'[.!?]\s+'
    
    # Search backwards from end to start
    search_text = text[start:end]
    matches = list(re.finditer(sentence_pattern, search_text))
    
    if matches:
        # Return position after the last sentence ending
        last_match = matches[-1]
        return start + last_match.end()
    
    return end

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
        
        # Extract body text for full article content
        body_text = ""
        body_elem = root.find(".//body")
        if body_elem is not None:
            # Extract all text from paragraphs and sections in the body
            for p in body_elem.findall(".//p"):
                if p.text:
                    body_text += p.text + " "
                # Also get text from child elements
                for child in p:
                    if child.text:
                        body_text += child.text + " "
                    if child.tail:
                        body_text += child.tail + " "
            
            # Clean up extra whitespace
            body_text = " ".join(body_text.split())
        
        # Create comprehensive content by combining title, abstract, and full body
        content = f"{title}\n\n{abstract}"
        if body_text:
            content += f"\n\n{body_text}"
        if authors:
            content += f"\n\nAuthors: {', '.join(authors)}"
        if keywords:
            content += f"\n\nKeywords: {', '.join(keywords)}"
        
        # Check if content is too large for LLM context (roughly 16k token limit = ~64k chars)
        content_length = len(content)
        needs_chunking = content_length > 12000  # Conservative threshold for chunking
        
        result = {
            "doc_id": pmc_id,
            "title": title,
            "content": content,
            "abstract": abstract,
            "authors": authors,
            "keywords": keywords,
            "metadata": {
                "source": "PMC",
                "file_path": xml_file_path,
                "pmc_id": pmc_id,
                "content_length": content_length,
                "needs_chunking": needs_chunking,
                "has_full_body": len(body_text) > 0
            }
        }
        
        # If chunking is needed, add chunked versions
        if needs_chunking:
            result["chunks"] = _chunk_pmc_content(content, pmc_id)
            result["metadata"]["chunk_count"] = len(result["chunks"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {xml_file_path}: {e}")
        pmc_id = os.path.basename(xml_file_path).replace('.xml', '')
        return {
            "doc_id": pmc_id,
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
