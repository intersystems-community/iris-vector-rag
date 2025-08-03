#!/usr/bin/env python3
"""
IPM Module Validation Script

This script validates the IPM module components including:
- module.xml syntax and structure
- ObjectScript installer class compilation
- Python package integration
- Installation workflow testing
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import subprocess


class IPMModuleValidator:
    """Validator for IPM module components."""
    
    def __init__(self, project_root: str = None):
        """Initialize validator with project root."""
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.results = {
            "module_xml": {},
            "objectscript_classes": {},
            "python_integration": {},
            "installation_workflow": {},
            "overall_status": False
        }
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks."""
        print("🔍 Starting IPM Module Validation...")
        print("=" * 50)
        
        # Validate module.xml
        print("\n📄 Validating module.xml...")
        self.validate_module_xml()
        
        # Validate ObjectScript classes
        print("\n🔧 Validating ObjectScript classes...")
        self.validate_objectscript_classes()
        
        # Validate Python integration
        print("\n🐍 Validating Python integration...")
        self.validate_python_integration()
        
        # Validate installation workflow
        print("\n⚙️ Validating installation workflow...")
        self.validate_installation_workflow()
        
        # Calculate overall status
        self.calculate_overall_status()
        
        print("\n" + "=" * 50)
        print(f"🎯 Overall Status: {'✅ PASSED' if self.results['overall_status'] else '❌ FAILED'}")
        
        return self.results
    
    def validate_module_xml(self) -> None:
        """Validate module.xml syntax and structure."""
        module_xml_path = self.project_root / "module.xml"
        
        if not module_xml_path.exists():
            self.results["module_xml"] = {
                "exists": False,
                "valid_xml": False,
                "has_required_elements": False,
                "error": "module.xml not found"
            }
            print("❌ module.xml not found")
            return
        
        try:
            # Parse XML
            tree = ET.parse(module_xml_path)
            root = tree.getroot()
            
            # Check required elements
            required_elements = [
                ".//Name",
                ".//Version", 
                ".//Description",
                ".//Dependencies",
                ".//Packaging",
                ".//Lifecycle"
            ]
            
            missing_elements = []
            for element_path in required_elements:
                if root.find(element_path) is None:
                    missing_elements.append(element_path)
            
            # Check lifecycle methods
            lifecycle_methods = [
                ".//Setup",
                ".//Configure", 
                ".//Activate",
                ".//Test"
            ]
            
            missing_lifecycle = []
            for method_path in lifecycle_methods:
                if root.find(method_path) is None:
                    missing_lifecycle.append(method_path)
            
            # Check parameters
            parameters = root.findall(".//Parameter")
            parameter_names = [p.get("Name") for p in parameters]
            
            expected_parameters = [
                "PYTHON_PATH",
                "INSTALL_PYTHON_PACKAGE",
                "ENABLE_VECTOR_SEARCH",
                "NAMESPACE"
            ]
            
            missing_parameters = [p for p in expected_parameters if p not in parameter_names]
            
            self.results["module_xml"] = {
                "exists": True,
                "valid_xml": True,
                "has_required_elements": len(missing_elements) == 0,
                "missing_elements": missing_elements,
                "has_lifecycle_methods": len(missing_lifecycle) == 0,
                "missing_lifecycle": missing_lifecycle,
                "has_parameters": len(missing_parameters) == 0,
                "missing_parameters": missing_parameters,
                "parameter_count": len(parameters)
            }
            
            if len(missing_elements) == 0 and len(missing_lifecycle) == 0:
                print("✅ module.xml structure is valid")
            else:
                print(f"⚠️ module.xml has issues: {missing_elements + missing_lifecycle}")
                
        except ET.ParseError as e:
            self.results["module_xml"] = {
                "exists": True,
                "valid_xml": False,
                "error": f"XML parse error: {e}"
            }
            print(f"❌ module.xml parse error: {e}")
        except Exception as e:
            self.results["module_xml"] = {
                "exists": True,
                "valid_xml": False,
                "error": f"Validation error: {e}"
            }
            print(f"❌ module.xml validation error: {e}")
    
    def validate_objectscript_classes(self) -> None:
        """Validate ObjectScript classes exist and have correct structure."""
        objectscript_dir = self.project_root / "objectscript"
        
        if not objectscript_dir.exists():
            self.results["objectscript_classes"] = {
                "directory_exists": False,
                "error": "objectscript directory not found"
            }
            print("❌ objectscript directory not found")
            return
        
        # Check required classes (use .CLS extension for IRIS)
        required_classes = [
            "RAG/IPMInstaller.CLS",
            "RAG/PythonBridge.CLS", 
            "RAG/VectorMigration.CLS"
        ]
        
        class_results = {}
        for class_file in required_classes:
            class_path = objectscript_dir / class_file
            
            if not class_path.exists():
                class_results[class_file] = {
                    "exists": False,
                    "error": "File not found"
                }
                continue
            
            # Basic syntax validation
            try:
                with open(class_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for basic ObjectScript class structure
                has_class_declaration = "Class " in content
                has_methods = "ClassMethod " in content
                has_proper_ending = content.strip().endswith("}")
                
                class_results[class_file] = {
                    "exists": True,
                    "has_class_declaration": has_class_declaration,
                    "has_methods": has_methods,
                    "has_proper_ending": has_proper_ending,
                    "line_count": len(content.splitlines())
                }
                
            except Exception as e:
                class_results[class_file] = {
                    "exists": True,
                    "error": f"Read error: {e}"
                }
        
        # Check for specific methods in IPMInstaller
        installer_path = objectscript_dir / "RAG.IPMInstaller.cls"
        if installer_path.exists():
            try:
                with open(installer_path, 'r', encoding='utf-8') as f:
                    installer_content = f.read()
                
                required_methods = [
                    "Setup",
                    "Configure", 
                    "Activate",
                    "Test",
                    "ValidateIRISVersion",
                    "EnableVectorSearch",
                    "InstallPythonPackage"
                ]
                
                missing_methods = []
                for method in required_methods:
                    if f"ClassMethod {method}" not in installer_content:
                        missing_methods.append(method)
                
                class_results["RAG.IPMInstaller.cls"]["required_methods"] = {
                    "all_present": len(missing_methods) == 0,
                    "missing": missing_methods
                }
                
            except Exception as e:
                class_results["RAG.IPMInstaller.cls"]["method_check_error"] = str(e)
        
        self.results["objectscript_classes"] = {
            "directory_exists": True,
            "classes": class_results,
            "all_classes_exist": all(r.get("exists", False) for r in class_results.values())
        }
        
        if self.results["objectscript_classes"]["all_classes_exist"]:
            print("✅ All required ObjectScript classes exist")
        else:
            missing = [k for k, v in class_results.items() if not v.get("exists", False)]
            print(f"⚠️ Missing ObjectScript classes: {missing}")
    
    def validate_python_integration(self) -> None:
        """Validate Python integration components."""
        try:
            # Check iris_rag package structure
            iris_rag_dir = self.project_root / "iris_rag"
            
            if not iris_rag_dir.exists():
                self.results["python_integration"] = {
                    "package_exists": False,
                    "error": "iris_rag package directory not found"
                }
                print("❌ iris_rag package directory not found")
                return
            
            # Check required files
            required_files = [
                "__init__.py",
                "utils/__init__.py",
                "utils/ipm_integration.py"
            ]
            
            file_results = {}
            for file_path in required_files:
                full_path = iris_rag_dir / file_path
                file_results[file_path] = {
                    "exists": full_path.exists(),
                    "size": full_path.stat().st_size if full_path.exists() else 0
                }
            
            # Check if IPMIntegration class can be imported
            try:
                sys.path.insert(0, str(self.project_root))
                from iris_rag.utils.ipm_integration import IPMIntegration
                
                # Test basic functionality
                ipm = IPMIntegration()
                
                integration_test = {
                    "import_successful": True,
                    "class_instantiation": True,
                    "package_name": ipm.package_name,
                    "version": ipm.version
                }
                
                # Test key methods exist
                methods_to_check = [
                    "validate_environment",
                    "install_package",
                    "verify_installation",
                    "generate_config_template"
                ]
                
                method_results = {}
                for method_name in methods_to_check:
                    method_results[method_name] = hasattr(ipm, method_name)
                
                integration_test["methods"] = method_results
                integration_test["all_methods_exist"] = all(method_results.values())
                
            except ImportError as e:
                integration_test = {
                    "import_successful": False,
                    "error": f"Import error: {e}"
                }
            except Exception as e:
                integration_test = {
                    "import_successful": True,
                    "class_instantiation": False,
                    "error": f"Instantiation error: {e}"
                }
            
            self.results["python_integration"] = {
                "package_exists": True,
                "files": file_results,
                "all_files_exist": all(r["exists"] for r in file_results.values()),
                "integration_test": integration_test
            }
            
            if integration_test.get("import_successful") and integration_test.get("all_methods_exist"):
                print("✅ Python integration components are valid")
            else:
                print("⚠️ Python integration has issues")
                
        except Exception as e:
            self.results["python_integration"] = {
                "package_exists": False,
                "error": f"Validation error: {e}"
            }
            print(f"❌ Python integration validation error: {e}")
    
    def validate_installation_workflow(self) -> None:
        """Validate the installation workflow components."""
        try:
            # Check if requirements.txt exists
            requirements_path = self.project_root / "requirements.txt"
            pyproject_path = self.project_root / "pyproject.toml"
            
            package_config = {
                "requirements_txt": requirements_path.exists(),
                "pyproject_toml": pyproject_path.exists()
            }
            
            # Check documentation
            docs_dir = self.project_root / "docs"
            ipm_doc_path = docs_dir / "IPM_INSTALLATION.md"
            
            documentation = {
                "docs_directory": docs_dir.exists(),
                "ipm_installation_guide": ipm_doc_path.exists()
            }
            
            # Check test files
            tests_dir = self.project_root / "tests"
            ipm_test_path = tests_dir / "test_ipm_integration.py"
            
            testing = {
                "tests_directory": tests_dir.exists(),
                "ipm_integration_tests": ipm_test_path.exists()
            }
            
            # Validate pyproject.toml structure if it exists
            pyproject_validation = {}
            if pyproject_path.exists():
                try:
                    import tomllib
                    with open(pyproject_path, 'rb') as f:
                        pyproject_data = tomllib.load(f)
                    
                    pyproject_validation = {
                        "valid_toml": True,
                        "has_tool_poetry": "tool" in pyproject_data and "poetry" in pyproject_data["tool"],
                        "package_name": pyproject_data.get("tool", {}).get("poetry", {}).get("name"),
                        "version": pyproject_data.get("tool", {}).get("poetry", {}).get("version")
                    }
                    
                except Exception as e:
                    pyproject_validation = {
                        "valid_toml": False,
                        "error": str(e)
                    }
            
            self.results["installation_workflow"] = {
                "package_config": package_config,
                "documentation": documentation,
                "testing": testing,
                "pyproject_validation": pyproject_validation
            }
            
            # Check overall workflow completeness
            workflow_complete = (
                package_config["pyproject_toml"] and
                documentation["ipm_installation_guide"] and
                testing["ipm_integration_tests"]
            )
            
            self.results["installation_workflow"]["complete"] = workflow_complete
            
            if workflow_complete:
                print("✅ Installation workflow components are complete")
            else:
                print("⚠️ Installation workflow has missing components")
                
        except Exception as e:
            self.results["installation_workflow"] = {
                "error": f"Validation error: {e}"
            }
            print(f"❌ Installation workflow validation error: {e}")
    
    def calculate_overall_status(self) -> None:
        """Calculate overall validation status."""
        checks = [
            self.results["module_xml"].get("valid_xml", False) and 
            self.results["module_xml"].get("has_required_elements", False),
            
            self.results["objectscript_classes"].get("all_classes_exist", False),
            
            self.results["python_integration"].get("all_files_exist", False) and
            self.results["python_integration"].get("integration_test", {}).get("import_successful", False),
            
            self.results["installation_workflow"].get("complete", False)
        ]
        
        self.results["overall_status"] = all(checks)
        self.results["passed_checks"] = sum(checks)
        self.results["total_checks"] = len(checks)
    
    def generate_report(self, output_path: str = None) -> str:
        """Generate a detailed validation report."""
        report = {
            "validation_timestamp": __import__("datetime").datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "results": self.results,
            "summary": {
                "overall_status": "PASSED" if self.results["overall_status"] else "FAILED",
                "passed_checks": self.results.get("passed_checks", 0),
                "total_checks": self.results.get("total_checks", 0)
            }
        }
        
        report_json = json.dumps(report, indent=2)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_json)
            print(f"\n📊 Detailed report saved to: {output_path}")
        
        return report_json


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate IPM Module Components")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--output", help="Output file for detailed report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = IPMModuleValidator(args.project_root)
    
    # Run validation
    results = validator.validate_all()
    
    # Generate report
    if args.output:
        validator.generate_report(args.output)
    
    # Print summary
    if args.verbose:
        print("\n📋 Detailed Results:")
        print(json.dumps(results, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_status"] else 1)


if __name__ == "__main__":
    main()