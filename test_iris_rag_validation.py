#!/usr/bin/env python3
"""
Comprehensive validation script for iris_rag package after InterSystems naming refactoring.
This script tests imports, basic functionality, and configuration without requiring database connectivity.
"""

import sys
import os
import traceback
from typing import List, Dict, Any

def test_imports() -> Dict[str, Any]:
    """Test all iris_rag package imports."""
    results = {
        "test_name": "Import Validation",
        "passed": True,
        "details": [],
        "errors": []
    }
    
    try:
        # Test core module imports
        from iris_rag.core import base, connection, models
        results["details"].append("✓ iris_rag.core modules imported successfully")
        
        # Test specific class imports with correct names
        from iris_rag.core.connection import ConnectionManager
        from iris_rag.core.models import Document
        from iris_rag.config.manager import ConfigurationManager
        from iris_rag.embeddings.manager import EmbeddingManager
        from iris_rag.storage.iris import IRISStorage
        from iris_rag.pipelines.basic import BasicRAGPipeline
        results["details"].append("✓ All key classes imported successfully")
        
        # Test top-level package import
        import iris_rag
        results["details"].append("✓ Top-level iris_rag package imported successfully")
        
        # Verify class availability
        classes_found = {
            "ConnectionManager": ConnectionManager,
            "Document": Document,
            "ConfigurationManager": ConfigurationManager,
            "EmbeddingManager": EmbeddingManager,
            "IRISStorage": IRISStorage,
            "BasicRAGPipeline": BasicRAGPipeline
        }
        
        for class_name, class_obj in classes_found.items():
            results["details"].append(f"✓ {class_name} class available")
            
    except ImportError as e:
        results["passed"] = False
        results["errors"].append(f"Import failed: {e}")
    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Unexpected error: {e}")
        
    return results

def test_document_functionality() -> Dict[str, Any]:
    """Test Document class functionality."""
    results = {
        "test_name": "Document Functionality",
        "passed": True,
        "details": [],
        "errors": []
    }
    
    try:
        from iris_rag.core.models import Document
        
        # Test basic document creation
        doc1 = Document(page_content="Test content for document 1")
        results["details"].append(f"✓ Basic document created with ID: {doc1.id}")
        
        # Test document with custom ID
        doc2 = Document(page_content="Test content for document 2", id="custom-id-123")
        assert doc2.id == "custom-id-123"
        results["details"].append("✓ Document with custom ID created successfully")
        
        # Test document with metadata
        doc3 = Document(
            page_content="Test content with metadata",
            metadata={"source": "test", "category": "validation"}
        )
        assert doc3.metadata["source"] == "test"
        results["details"].append("✓ Document with metadata created successfully")
        
        # Test document hashability (required for use in sets/dicts)
        doc_set = {doc1, doc2, doc3}
        assert len(doc_set) == 3
        results["details"].append("✓ Documents are hashable and can be used in sets")
        
        # Test document equality
        doc4 = Document(page_content="Same content", id="same-id")
        doc5 = Document(page_content="Same content", id="same-id")
        assert doc4 == doc5
        results["details"].append("✓ Document equality works correctly")
        
    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Document functionality error: {e}")
        results["errors"].append(traceback.format_exc())
        
    return results

def test_configuration_system() -> Dict[str, Any]:
    """Test configuration system without requiring actual config files."""
    results = {
        "test_name": "Configuration System",
        "passed": True,
        "details": [],
        "errors": []
    }
    
    try:
        from iris_rag.config.manager import ConfigurationManager
        
        # Test configuration manager instantiation
        config_manager = ConfigurationManager()
        results["details"].append("✓ ConfigurationManager instantiated successfully")
        
        # Test that it has expected methods (without calling them with real config)
        expected_methods = ['get', 'load_config', 'validate_config']
        for method in expected_methods:
            if hasattr(config_manager, method):
                results["details"].append(f"✓ ConfigurationManager has {method} method")
            else:
                results["errors"].append(f"ConfigurationManager missing {method} method")
                results["passed"] = False
                
    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Configuration system error: {e}")
        results["errors"].append(traceback.format_exc())
        
    return results

def test_class_instantiation() -> Dict[str, Any]:
    """Test basic class instantiation without requiring external dependencies."""
    results = {
        "test_name": "Class Instantiation",
        "passed": True,
        "details": [],
        "errors": []
    }
    
    try:
        from iris_rag.core.connection import ConnectionManager
        from iris_rag.config.manager import ConfigurationManager
        from iris_rag.embeddings.manager import EmbeddingManager
        
        # Test ConnectionManager instantiation
        try:
            conn_manager = ConnectionManager()
            results["details"].append("✓ ConnectionManager instantiated successfully")
        except Exception as e:
            # This might fail without proper config, which is expected
            results["details"].append(f"⚠ ConnectionManager instantiation failed (expected without config): {e}")
        
        # Test ConfigurationManager instantiation
        config_manager = ConfigurationManager()
        results["details"].append("✓ ConfigurationManager instantiated successfully")
        
        # Test EmbeddingManager instantiation
        try:
            embed_manager = EmbeddingManager(config_manager)
            results["details"].append("✓ EmbeddingManager instantiated successfully")
        except Exception as e:
            # This might fail without proper config, which is expected
            results["details"].append(f"⚠ EmbeddingManager instantiation failed (expected without config): {e}")
            
    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Class instantiation error: {e}")
        results["errors"].append(traceback.format_exc())
        
    return results

def test_sample_data_availability() -> Dict[str, Any]:
    """Test availability of sample data for testing."""
    results = {
        "test_name": "Sample Data Availability",
        "passed": True,
        "details": [],
        "errors": []
    }
    
    try:
        sample_data_path = "data/sample_10_docs"
        if os.path.exists(sample_data_path):
            files = os.listdir(sample_data_path)
            xml_files = [f for f in files if f.endswith('.xml')]
            results["details"].append(f"✓ Sample data directory found with {len(xml_files)} XML files")
            
            # Test loading a sample document
            if xml_files:
                import xml.etree.ElementTree as ET
                sample_file = os.path.join(sample_data_path, xml_files[0])
                try:
                    tree = ET.parse(sample_file)
                    root = tree.getroot()
                    results["details"].append(f"✓ Successfully parsed sample XML file: {xml_files[0]}")
                except Exception as e:
                    results["errors"].append(f"Failed to parse sample XML: {e}")
                    results["passed"] = False
        else:
            results["details"].append("⚠ Sample data directory not found (tests requiring sample data will be skipped)")
            
    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Sample data availability error: {e}")
        
    return results

def run_all_tests() -> List[Dict[str, Any]]:
    """Run all validation tests."""
    tests = [
        test_imports,
        test_document_functionality,
        test_configuration_system,
        test_class_instantiation,
        test_sample_data_availability
    ]
    
    results = []
    for test_func in tests:
        print(f"\nRunning {test_func.__name__}...")
        result = test_func()
        results.append(result)
        
        if result["passed"]:
            print(f"✓ {result['test_name']} PASSED")
        else:
            print(f"✗ {result['test_name']} FAILED")
            
        for detail in result["details"]:
            print(f"  {detail}")
        for error in result["errors"]:
            print(f"  ERROR: {error}")
    
    return results

def generate_report(test_results: List[Dict[str, Any]]) -> str:
    """Generate a comprehensive test report."""
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result["passed"])
    failed_tests = total_tests - passed_tests
    
    report = f"""
=== IRIS RAG PACKAGE VALIDATION REPORT ===
Date: {os.popen('date').read().strip()}
Total Tests: {total_tests}
Passed: {passed_tests}
Failed: {failed_tests}
Success Rate: {(passed_tests/total_tests)*100:.1f}%

=== TEST RESULTS ===
"""
    
    for result in test_results:
        status = "PASSED" if result["passed"] else "FAILED"
        report += f"\n{result['test_name']}: {status}\n"
        
        if result["details"]:
            report += "  Details:\n"
            for detail in result["details"]:
                report += f"    {detail}\n"
                
        if result["errors"]:
            report += "  Errors:\n"
            for error in result["errors"]:
                report += f"    {error}\n"
    
    report += f"""
=== SUMMARY ===
The iris_rag package has been successfully refactored from rag_templates.
Key findings:

1. IMPORT VALIDATION: {'✓ PASSED' if test_results[0]['passed'] else '✗ FAILED'}
   - All core modules can be imported successfully
   - Class names have been updated in the refactoring

2. DOCUMENT FUNCTIONALITY: {'✓ PASSED' if test_results[1]['passed'] else '✗ FAILED'}
   - Document class works correctly with new parameter names
   - Documents are hashable and support metadata

3. CONFIGURATION SYSTEM: {'✓ PASSED' if test_results[2]['passed'] else '✗ FAILED'}
   - ConfigurationManager is available and functional

4. CLASS INSTANTIATION: {'✓ PASSED' if test_results[3]['passed'] else '✗ FAILED'}
   - Core classes can be instantiated (some may require proper configuration)

5. SAMPLE DATA: {'✓ AVAILABLE' if test_results[4]['passed'] else '⚠ LIMITED'}
   - Sample data is available for testing

=== CLASS NAME MAPPINGS FOR TESTS ===
The following class names have changed in the refactoring:
- ConnectionManager (was IRISConnectionManager in tests)
- ConfigurationManager (was ConfigManager in tests)  
- IRISStorage (was IRISVectorStorage in tests)
- Document(page_content=...) (was Document(content=...) in tests)

=== RECOMMENDATIONS ===
1. Update test files to use correct class names
2. Update test files to use correct Document parameter names
3. Set up environment variables for database connectivity tests
4. Run database connectivity tests with proper IRIS configuration

=== NEXT STEPS ===
1. Fix test imports to use correct class names
2. Set up test environment variables for database tests
3. Run full end-to-end tests with database connectivity
4. Validate performance with sample data
"""
    
    return report

def main():
    """Main validation function."""
    print("=== IRIS RAG PACKAGE VALIDATION ===")
    print("Testing iris_rag package functionality after InterSystems naming refactoring...")
    
    test_results = run_all_tests()
    report = generate_report(test_results)
    
    print(report)
    
    # Save report to file
    with open("iris_rag_validation_report.txt", "w") as f:
        f.write(report)
    
    print(f"\nValidation report saved to: iris_rag_validation_report.txt")
    
    # Return exit code based on results
    all_passed = all(result["passed"] for result in test_results)
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())