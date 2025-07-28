"""
Test suite for scripts/ingest_1000_docs.py

This test follows TDD principles to ensure the ingestion script:
1. Uses the RAG class from rag_templates.simple
2. Loads documents from a specified directory
3. Handles potential errors during ingestion
4. Logs the number of documents successfully ingested
5. Ingests at least 1000 documents
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import logging
import subprocess

# Add the project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestIngest1000Docs:
    """Test suite for the ingest_1000_docs.py script."""
    
    def test_script_exists(self):
        """Test that the ingest_1000_docs.py script exists."""
        script_path = Path("scripts/ingest_1000_docs.py")
        assert script_path.exists(), f"Script {script_path} should exist"
    
    def test_script_is_executable(self):
        """Test that the script can be executed with uv run."""
        script_path = Path("scripts/ingest_1000_docs.py")
        assert script_path.exists(), f"Script {script_path} should exist"
        
        # Test that the script can be imported without syntax errors
        try:
            result = subprocess.run(
                ["uv", "run", "python", "-m", "py_compile", str(script_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0, f"Script has syntax errors: {result.stderr}"
        except subprocess.TimeoutExpired:
            pytest.fail("Script compilation timed out")
    
    def test_script_imports_rag_class(self):
        """Test that the script imports the RAG class from rag_templates.simple."""
        script_path = Path("scripts/ingest_1000_docs.py")
        assert script_path.exists(), f"Script {script_path} should exist"
        
        with open(script_path, 'r') as f:
            content = f.read()
            assert "from rag_templates.simple import RAG" in content, \
                "Script should import RAG class from rag_templates.simple"
    
    def test_script_has_main_function(self):
        """Test that the script has a main function."""
        script_path = Path("scripts/ingest_1000_docs.py")
        assert script_path.exists(), f"Script {script_path} should exist"
        
        with open(script_path, 'r') as f:
            content = f.read()
            assert "def main(" in content, "Script should have a main function"
    
    def test_script_accepts_directory_argument(self):
        """Test that the script accepts a directory argument."""
        script_path = Path("scripts/ingest_1000_docs.py")
        assert script_path.exists(), f"Script {script_path} should exist"
        
        with open(script_path, 'r') as f:
            content = f.read()
            # Should have argument parsing for directory
            assert any(keyword in content for keyword in ["argparse", "sys.argv", "directory"]), \
                "Script should accept directory argument"
    
    def test_script_handles_errors(self):
        """Test that the script has error handling."""
        script_path = Path("scripts/ingest_1000_docs.py")
        assert script_path.exists(), f"Script {script_path} should exist"
        
        with open(script_path, 'r') as f:
            content = f.read()
            assert "try:" in content and "except" in content, \
                "Script should have error handling with try/except blocks"
    
    def test_script_has_logging(self):
        """Test that the script includes logging functionality."""
        script_path = Path("scripts/ingest_1000_docs.py")
        assert script_path.exists(), f"Script {script_path} should exist"
        
        with open(script_path, 'r') as f:
            content = f.read()
            assert "logging" in content, "Script should use logging"
            assert any(level in content for level in ["info", "error", "warning", "debug"]), \
                "Script should log messages"
    
    @patch('rag_templates.simple.RAG')
    def test_script_uses_rag_class(self, mock_rag_class):
        """Test that the script creates and uses a RAG instance."""
        # This test will pass once the script is implemented
        mock_rag_instance = MagicMock()
        mock_rag_instance.get_document_count.side_effect = [0, 5]  # Initial: 0, Final: 5
        mock_rag_class.return_value = mock_rag_instance
        
        # Import and run the script's main function
        try:
            from scripts.ingest_1000_docs import main
            
            # Create a temporary directory with some test files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create some test XML files
                for i in range(5):
                    test_file = Path(temp_dir) / f"PMC{i}.xml"
                    test_file.write_text(f"""<?xml version="1.0"?>
                    <article>
                        <article-title>Test Article {i}</article-title>
                        <abstract><p>Test abstract {i}</p></abstract>
                    </article>""")
                
                # Run the main function - expect SystemExit since < 1000 docs
                with pytest.raises(SystemExit) as exc_info:
                    main(temp_dir)
                
                # Verify it exits with code 1 (expected for < 1000 docs)
                assert exc_info.value.code == 1
                
                # Verify RAG class was instantiated
                mock_rag_class.assert_called_once()
                
                # Verify add_documents was called
                mock_rag_instance.add_documents.assert_called()
                
        except ImportError:
            pytest.fail("Cannot import main function from scripts.ingest_1000_docs")
    
    def test_script_logs_document_count(self):
        """Test that the script logs the number of documents ingested."""
        script_path = Path("scripts/ingest_1000_docs.py")
        assert script_path.exists(), f"Script {script_path} should exist"
        
        with open(script_path, 'r') as f:
            content = f.read()
            # Should log document count information
            assert any(phrase in content.lower() for phrase in [
                "documents ingested", "documents loaded", "documents processed",
                "successfully ingested", "total documents"
            ]), "Script should log document count information"
    
    @patch('scripts.ingest_1000_docs.process_pmc_files')
    def test_script_processes_at_least_1000_docs(self, mock_process_pmc):
        """Test that the script is designed to process at least 1000 documents."""
        # Mock the PMC processor to return 1000+ documents
        mock_documents = [
            {
                'pmc_id': f'PMC{i}',
                'title': f'Test Document {i}',
                'content': f'Test content {i}',
                'abstract': f'Test abstract {i}',
                'authors': [f'Author {i}'],
                'keywords': [f'keyword{i}']
            }
            for i in range(1200)  # More than 1000 documents
        ]
        mock_process_pmc.return_value = iter(mock_documents)
        
        try:
            from scripts.ingest_1000_docs import main
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run the main function - should succeed with 1200 docs
                main(temp_dir)
                
                # If we get here without SystemExit, the script succeeded
                # The script should have processed 1200 documents successfully
                assert True, "Script successfully processed 1000+ documents"
                
        except SystemExit as e:
            # Should not exit with error when 1000+ docs are processed
            pytest.fail(f"Script should not exit with error when processing 1000+ docs, but got exit code: {e.code}")
        except ImportError:
            pytest.fail("Cannot import main function from scripts.ingest_1000_docs")
    
    def test_script_handles_missing_directory(self):
        """Test that the script handles missing directory gracefully."""
        try:
            from scripts.ingest_1000_docs import main
            
            # Test with non-existent directory
            with pytest.raises((FileNotFoundError, ValueError, SystemExit)):
                main("/non/existent/directory")
                
        except ImportError:
            pytest.fail("Cannot import main function from scripts.ingest_1000_docs")
    
    def test_script_command_line_interface(self):
        """Test that the script can be run from command line."""
        script_path = Path("scripts/ingest_1000_docs.py")
        assert script_path.exists(), f"Script {script_path} should exist"
        
        with open(script_path, 'r') as f:
            content = f.read()
            assert 'if __name__ == "__main__"' in content, \
                "Script should have command line interface"


class TestIngest1000DocsIntegration:
    """Integration tests for the ingest_1000_docs.py script with real data."""
    
    def test_script_with_sample_data(self):
        """Test the script with the sample PMC data."""
        script_path = Path("scripts/ingest_1000_docs.py")
        sample_data_dir = Path("data/sample_10_docs")
        
        if not script_path.exists():
            pytest.skip("Script not yet implemented")
        
        if not sample_data_dir.exists():
            pytest.skip("Sample data directory not found")
        
        # Run the script with sample data
        try:
            result = subprocess.run(
                ["uv", "run", "python", str(script_path), str(sample_data_dir)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Script should run without errors
            assert result.returncode == 0, f"Script failed with error: {result.stderr}"
            
            # Should log some information about processing
            assert "documents" in result.stdout.lower() or "documents" in result.stderr.lower(), \
                "Script should log information about document processing"
                
        except subprocess.TimeoutExpired:
            pytest.fail("Script execution timed out")
        except FileNotFoundError:
            pytest.skip("UV not available for testing")
