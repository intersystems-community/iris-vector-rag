#!/usr/bin/env python3
"""
IPM Package Validation Script

Validates that the module.xml references existing files and directories
to ensure the IPM package will install correctly.
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

def validate_ipm_package(repo_path: str) -> Tuple[bool, List[str]]:
    """
    Validate IPM package structure against module.xml
    
    Args:
        repo_path: Path to repository root
        
    Returns:
        Tuple of (success, list_of_issues)
    """
    issues = []
    repo_root = Path(repo_path)
    
    # Check if module.xml exists
    module_xml = repo_root / "module.xml"
    if not module_xml.exists():
        issues.append("❌ module.xml not found")
        return False, issues
    
    try:
        # Parse module.xml
        tree = ET.parse(module_xml)
        root = tree.getroot()
        
        # Find the Module element
        module = root.find(".//Module")
        if module is None:
            issues.append("❌ No Module element found in module.xml")
            return False, issues
        
        # Check version
        version = module.find("Version")
        if version is not None:
            print(f"📦 Package Version: {version.text}")
        
        # Find Packaging element
        packaging = module.find("Packaging")
        if packaging is None:
            issues.append("❌ No Packaging element found")
            return False, issues
        
        # Validate resources
        resources = packaging.findall("Resource")
        print(f"🔍 Validating {len(resources)} resources...")
        
        for resource in resources:
            name = resource.get("Name")
            directory = resource.get("Directory", "")
            recurse = resource.get("Recurse") == "true"
            
            if directory:
                # Directory resource
                resource_path = repo_root / directory / name
                if directory == "":
                    # Root level file
                    resource_path = repo_root / name
                else:
                    # Check if it's a directory itself
                    dir_path = repo_root / directory.rstrip("/")
                    if name == directory.rstrip("/"):
                        resource_path = dir_path
                    elif name in ["iris_rag", "rag_templates", "common"]:
                        # These are directory names
                        resource_path = repo_root / name
            else:
                # Root level resource
                resource_path = repo_root / name
            
            # Check if resource exists
            if not resource_path.exists():
                issues.append(f"❌ Missing resource: {name} (expected at {resource_path})")
            else:
                if resource_path.is_dir() and recurse:
                    file_count = len(list(resource_path.rglob("*")))
                    print(f"✅ Directory: {name} ({file_count} files)")
                elif resource_path.is_file():
                    print(f"✅ File: {name}")
                else:
                    print(f"✅ Resource: {name}")
        
        # Check ObjectScript files specifically
        objectscript_dir = repo_root / "objectscript"
        if objectscript_dir.exists():
            required_cls_files = [
                "RAG.IPMInstaller.cls",
                "RAG.VectorMigration.cls", 
                "RAG.IFindSetup.cls",
                "RAG.SourceDocumentsIFind.cls"
            ]
            
            for cls_file in required_cls_files:
                cls_path = objectscript_dir / cls_file
                if not cls_path.exists():
                    issues.append(f"❌ Missing ObjectScript class: {cls_file}")
                else:
                    print(f"✅ ObjectScript: {cls_file}")
        
        # Validate key Python packages
        key_packages = ["iris_rag", "rag_templates", "common"]
        for package in key_packages:
            package_path = repo_root / package
            if not package_path.exists():
                issues.append(f"❌ Missing Python package: {package}")
            elif not (package_path / "__init__.py").exists():
                issues.append(f"⚠️  Python package missing __init__.py: {package}")
            else:
                print(f"✅ Python package: {package}")
        
        # Check essential files
        essential_files = [
            "README.md",
            "pyproject.toml", 
            "requirements.txt",
            "Makefile"
        ]
        
        for file_name in essential_files:
            file_path = repo_root / file_name
            if not file_path.exists():
                issues.append(f"❌ Missing essential file: {file_name}")
            else:
                print(f"✅ Essential file: {file_name}")
        
        success = len(issues) == 0
        return success, issues
        
    except ET.ParseError as e:
        issues.append(f"❌ XML parsing error: {e}")
        return False, issues
    except Exception as e:
        issues.append(f"❌ Validation error: {e}")
        return False, issues

def main():
    """Main validation function"""
    if len(sys.argv) != 2:
        print("Usage: python validate_ipm_package.py <repo_path>")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    if not os.path.exists(repo_path):
        print(f"❌ Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    print(f"🔍 Validating IPM package at: {repo_path}")
    print("=" * 60)
    
    success, issues = validate_ipm_package(repo_path)
    
    print("=" * 60)
    if success:
        print("✅ IPM package validation PASSED")
        print("🎉 Package structure is correct for CI/CD pipeline")
    else:
        print("❌ IPM package validation FAILED")
        print(f"\n📋 Issues found ({len(issues)}):")
        for issue in issues:
            print(f"  {issue}")
        sys.exit(1)

if __name__ == "__main__":
    main()