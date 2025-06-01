import os
import glob
import re
import xml.etree.ElementTree as ET
import pytest

SAMPLE_DATA_DIR = "data/sample_10_docs/"
EXPECTED_FILE_COUNT = 10
PMC_ID_PATTERN = re.compile(r"^(PMC\d+)\.xml$", re.IGNORECASE)

def test_sample_directory_exists():
    """Tests if the sample data directory exists."""
    assert os.path.isdir(SAMPLE_DATA_DIR), f"Directory '{SAMPLE_DATA_DIR}' does not exist."

def test_exact_number_of_pmc_files():
    """Tests if there are exactly the expected number of PMC XML files."""
    if not os.path.isdir(SAMPLE_DATA_DIR):
        pytest.skip(f"Directory '{SAMPLE_DATA_DIR}' does not exist, skipping file count test.")
    
    xml_files = glob.glob(os.path.join(SAMPLE_DATA_DIR, "*.xml"))
    # Also consider .XML in case of mixed case filenames
    xml_files_upper = glob.glob(os.path.join(SAMPLE_DATA_DIR, "*.XML"))
    all_xml_files = list(set(xml_files + xml_files_upper))
    
    assert len(all_xml_files) == EXPECTED_FILE_COUNT, \
        f"Expected {EXPECTED_FILE_COUNT} XML files, but found {len(all_xml_files)} in '{SAMPLE_DATA_DIR}'."

def test_pmc_files_have_valid_filenames_and_readable_xml():
    """
    Tests if each XML file in the sample directory:
    1. Has a filename matching the PMC ID pattern (e.g., PMC12345.xml).
    2. Contains readable XML content.
    """
    if not os.path.isdir(SAMPLE_DATA_DIR):
        pytest.skip(f"Directory '{SAMPLE_DATA_DIR}' does not exist, skipping file content tests.")

    xml_files = glob.glob(os.path.join(SAMPLE_DATA_DIR, "*.xml"))
    xml_files_upper = glob.glob(os.path.join(SAMPLE_DATA_DIR, "*.XML"))
    all_xml_files = list(set(xml_files + xml_files_upper))

    if not all_xml_files and os.path.exists(SAMPLE_DATA_DIR): # only fail if dir exists but no files
         # If the test_exact_number_of_pmc_files is supposed to catch this,
         # this might be redundant or could be a specific fail for this test's purpose.
         # For now, let's assume test_exact_number_of_pmc_files handles the "no files" case primarily.
         # If that test is skipped, this one might still run.
        if len(all_xml_files) < EXPECTED_FILE_COUNT : # Check if we have less than expected.
             pytest.skip(f"Less than {EXPECTED_FILE_COUNT} XML files found, cannot reliably test all aspects.")


    processed_files_count = 0
    for filepath in all_xml_files:
        filename = os.path.basename(filepath)
        
        # Test 1: Valid PMC ID in filename
        match = PMC_ID_PATTERN.match(filename)
        assert match, \
            f"Filename '{filename}' does not match the expected PMC ID pattern (e.g., PMC12345.xml)."
        
        # Test 2: Readable XML content
        try:
            ET.parse(filepath)
        except ET.ParseError as e:
            pytest.fail(f"File '{filename}' is not readable XML or is corrupted. Error: {e}")
        except Exception as e:
            pytest.fail(f"An unexpected error occurred while parsing '{filename}'. Error: {e}")
        processed_files_count +=1
    
    # This ensures that if files were found, they were processed.
    # If EXPECTED_FILE_COUNT is 0, this would need adjustment. But it's 10.
    if len(all_xml_files) > 0 :
        assert processed_files_count == len(all_xml_files), \
            f"Processed {processed_files_count} files, but found {len(all_xml_files)} XML files."
    # If there are no XML files, but the directory exists, this test should not fail here,
    # as test_exact_number_of_pmc_files covers the count.
    # If it skipped due to dir not existing, this test also skips.

def test_no_unexpected_files_in_sample_dir():
    """Tests that only XML files (and potentially a README.md) are in the sample directory."""
    if not os.path.isdir(SAMPLE_DATA_DIR):
        pytest.skip(f"Directory '{SAMPLE_DATA_DIR}' does not exist, skipping unexpected files test.")

    allowed_extensions = {".xml", ".md"} # Allow .xml and .md for README
    # Handle case insensitivity for extensions
    allowed_extensions_lower = {ext.lower() for ext in allowed_extensions}

    unexpected_files = []
    for item in os.listdir(SAMPLE_DATA_DIR):
        item_path = os.path.join(SAMPLE_DATA_DIR, item)
        if os.path.isfile(item_path):
            _, ext = os.path.splitext(item)
            if ext.lower() not in allowed_extensions_lower:
                unexpected_files.append(item)
    
    assert not unexpected_files, \
        f"Found unexpected files in '{SAMPLE_DATA_DIR}': {unexpected_files}. Only XML and README.md are expected."