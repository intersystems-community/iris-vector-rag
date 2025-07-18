# tests/test_pmc_processor.py
# Tests for the PMC XML processor

import pytest
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch, mock_open

# Make sure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from data.pmc_processor import extract_pmc_metadata, process_pmc_files

# --- Constants for testing ---

# Sample PMC XML content for testing
SAMPLE_PMC_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE article PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD v1.2 20190208//EN" "JATS-archivearticle1.dtd">
<article>
  <front>
    <article-meta>
      <article-id pub-id-type="pmc">PMC123456</article-id>
      <title-group>
        <article-title>Test Article Title</article-title>
      </title-group>
      <contrib-group>
        <contrib contrib-type="author">
          <name>
            <surname>Smith</surname>
            <given-names>John</given-names>
          </name>
        </contrib>
        <contrib contrib-type="author">
          <name>
            <surname>Johnson</surname>
            <given-names>Jane</given-names>
          </name>
        </contrib>
      </contrib-group>
      <abstract>
        <p>This is a test abstract for the PMC article. It contains some sample text to test extraction.</p>
        <p>This is a second paragraph with some <italic>formatted text</italic> included.</p>
      </abstract>
      <kwd-group>
        <kwd>keyword1</kwd>
        <kwd>keyword2</kwd>
        <kwd>keyword3</kwd>
      </kwd-group>
    </article-meta>
  </front>
  <body>
    <p>This is the article body which shouldn't be included in the abstract.</p>
  </body>
</article>
"""

# --- Helper functions ---

def create_sample_xml_file(path):
    """Create a sample PMC XML file at the given path"""
    with open(path, 'w') as f:
        f.write(SAMPLE_PMC_XML)
    return path

# --- Unit Tests ---

def test_extract_pmc_metadata_with_mock_file():
    """Test extraction of metadata using a mock file"""
    # Setup
    with patch("builtins.open", mock_open(read_data=SAMPLE_PMC_XML)):
        with patch("xml.etree.ElementTree.parse") as mock_parse:
            # Create a mock ElementTree and root
            tree = ET.ElementTree(ET.fromstring(SAMPLE_PMC_XML))
            mock_parse.return_value = tree
            
            # Call the function with a dummy path
            result = extract_pmc_metadata("dummy/path/PMC123456.xml")
    
    # Assert the results
    assert result["pmc_id"] == "PMC123456"
    assert result["title"] == "Test Article Title"
    assert "This is a test abstract" in result["abstract"]
    assert "second paragraph" in result["abstract"]
    assert len(result["authors"]) == 2
    assert "John Smith" in result["authors"]
    assert "Jane Johnson" in result["authors"]
    assert len(result["keywords"]) == 3
    assert "keyword1" in result["keywords"]
    assert "keyword2" in result["keywords"]
    assert "keyword3" in result["keywords"]

def test_extract_pmc_metadata_with_temp_file():
    """Test extraction of metadata using a temporary file"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Write sample XML to the file
        create_sample_xml_file(temp_path)
        
        # Call the function
        result = extract_pmc_metadata(temp_path)
        
        # Assert the results
        assert result["title"] == "Test Article Title"
        assert "This is a test abstract" in result["abstract"]
        assert len(result["authors"]) == 2
        assert len(result["keywords"]) == 3
    finally:
        # Cleanup
        os.unlink(temp_path)

def test_extract_pmc_metadata_with_missing_fields():
    """Test extraction when some fields are missing"""
    # Modified XML with missing fields
    modified_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <article>
      <front>
        <article-meta>
          <article-id pub-id-type="pmc">PMC123456</article-id>
          <title-group>
            <article-title>Test Article Title</article-title>
          </title-group>
          <!-- No contrib-group -->
          <!-- No abstract -->
          <!-- No kwd-group -->
        </article-meta>
      </front>
      <body>
        <p>This is the article body which shouldn't be included.</p>
      </body>
    </article>
    """
    
    # Setup
    with patch("builtins.open", mock_open(read_data=modified_xml)):
        with patch("xml.etree.ElementTree.parse") as mock_parse:
            # Create a mock ElementTree and root
            tree = ET.ElementTree(ET.fromstring(modified_xml))
            mock_parse.return_value = tree
            
            # Call the function with a dummy path
            result = extract_pmc_metadata("dummy/path/PMC123456.xml")
    
    # Assert the results
    assert result["pmc_id"] == "PMC123456"
    assert result["title"] == "Test Article Title"
    assert result["abstract"] == ""  # Empty abstract
    assert result["authors"] == []   # Empty authors list
    assert result["keywords"] == []  # Empty keywords list

def test_extract_pmc_metadata_error_handling():
    """Test error handling in extract_pmc_metadata"""
    # Setup the mock to raise an exception
    with patch("xml.etree.ElementTree.parse", side_effect=Exception("XML Parsing Error")):
        # Call the function with a dummy path
        result = extract_pmc_metadata("dummy/path/PMC123456.xml")
    
    # Assert the error handling
    assert result["pmc_id"] == "PMC123456"
    assert result["title"] == "Error"
    assert "Failed to process" in result["abstract"]
    assert result["authors"] == []
    assert result["keywords"] == []

def test_process_pmc_files_with_mocked_directory():
    """Test processing of multiple PMC files with limit"""
    # Simulated filenames to process
    filenames = [f"PMC{i}.xml" for i in range(1, 11)]  # 10 files
    
    # Mock the os.walk to return our simulated file list
    with patch("os.walk") as mock_walk:
        mock_walk.return_value = [("/fake/dir", [], filenames)]
        
        # Mock extract_pmc_metadata to return predictable results
        with patch("data.pmc_processor.extract_pmc_metadata") as mock_extract:
            mock_extract.side_effect = lambda path: {
                "pmc_id": Path(path).stem,
                "title": f"Title for {Path(path).stem}",
                "abstract": f"Abstract for {Path(path).stem}",
                "authors": [f"Author1 for {Path(path).stem}", f"Author2 for {Path(path).stem}"],
                "keywords": [f"Keyword1 for {Path(path).stem}", f"Keyword2 for {Path(path).stem}"]
            }
            
            # Process with a limit of 5
            results = list(process_pmc_files("/fake/dir", limit=5))
    
    # Assert the results
    assert len(results) == 5  # Should respect the limit
    assert all(isinstance(r, dict) for r in results)
    assert all(key in r for r in results for key in ["pmc_id", "title", "abstract", "authors", "keywords"])
    assert results[0]["pmc_id"] == "PMC1"
    assert results[4]["pmc_id"] == "PMC5"

def test_process_pmc_files_error_handling():
    """Test error handling in process_pmc_files"""
    # Simulated filenames to process
    filenames = [f"PMC{i}.xml" for i in range(1, 6)]  # 5 files
    
    # Mock the os.walk to return our simulated file list
    with patch("os.walk") as mock_walk:
        mock_walk.return_value = [("/fake/dir", [], filenames)]
        
        # Mock extract_pmc_metadata to raise an exception for one file
        def mock_extract_side_effect(path):
            if "PMC3" in path:
                raise Exception("Processing error")
            return {
                "pmc_id": Path(path).stem,
                "title": f"Title for {Path(path).stem}",
                "abstract": f"Abstract for {Path(path).stem}",
                "authors": [],
                "keywords": []
            }
            
        with patch("data.pmc_processor.extract_pmc_metadata") as mock_extract:
            mock_extract.side_effect = mock_extract_side_effect
            
            # Process all files
            results = list(process_pmc_files("/fake/dir", limit=5))
    
    # Assert the results
    assert len(results) == 4  # One file should be skipped due to error
    assert "PMC3" not in [r["pmc_id"] for r in results]

# --- Integration Tests ---

@pytest.mark.integration
def test_extract_pmc_metadata_with_real_sample():
    """Test extraction with a real sample file if available"""
    # Look for sample files in common locations
    sample_paths = [
        "data/pmc_oas_downloaded/sample.xml",
        "data/pmc_oas_downloaded/PMC006xxxxxx/PMC6627345.xml",
        "data/pmc_oas_downloaded/PMC010xxxxxx/PMC10070455.xml"
    ]
    
    sample_path = None
    for path in sample_paths:
        if os.path.exists(path):
            sample_path = path
            break
    
    if not sample_path:
        pytest.skip("No sample PMC XML files found for testing")
    
    # Process the sample file
    result = extract_pmc_metadata(sample_path)
    
    # Basic validation
    assert result["pmc_id"] != ""
    assert result["title"] != ""
    assert result["title"] != "Unknown Title"
    assert result["title"] != "Error"
    assert len(result["abstract"]) > 0
    # Authors and keywords might be empty in some samples, so no assertions for those

@pytest.mark.integration
def test_process_pmc_files_with_real_directory():
    """Test processing of real PMC files if available"""
    # Check if the data directory exists
    data_dir = "data/pmc_oas_downloaded"
    if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
        pytest.skip(f"PMC data directory {data_dir} not found")
    
    # Process a small number of files
    results = list(process_pmc_files(data_dir, limit=2))
    
    # Validate results
    assert len(results) > 0  # Should find at least one file
    assert all(isinstance(r, dict) for r in results)
    assert all(key in r for r in results for key in ["pmc_id", "title", "abstract", "authors", "keywords"])
