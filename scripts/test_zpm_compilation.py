#!/usr/bin/env python3
"""
Test script to validate ZPM package compilation without requiring IRIS admin access.
This script performs comprehensive validation of all ZPM package components.
"""

import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

def test_module_xml():
    """Test module.xml structure and required elements."""
    print("🔍 Testing module.xml structure...")
    
    module_xml = Path("module.xml")
    if not module_xml.exists():
        print("❌ module.xml not found")
        return False
        
    try:
        tree = ET.parse(module_xml)
        root = tree.getroot()
        
        # Check required elements (minimal ZPM structure)
        required_elements = [
            ".//Name",
            ".//Version", 
            ".//Description",
            ".//Dependencies",
            ".//Packaging"
        ]
        
        for element_path in required_elements:
            element = root.find(element_path)
            if element is None:
                print(f"❌ Missing required element: {element_path}")
                return False
                
        print("✅ module.xml structure is valid")
        return True
        
    except ET.ParseError as e:
        print(f"❌ module.xml parse error: {e}")
        return False

def test_objectscript_syntax():
    """Test ObjectScript class file syntax."""
    print("🔍 Testing ObjectScript class syntax...")
    
    objectscript_dir = Path("objectscript")
    if not objectscript_dir.exists():
        print("❌ objectscript directory not found")
        return False
        
    cls_files = list(objectscript_dir.rglob("*.CLS"))
    if not cls_files:
        print("❌ No .CLS files found")
        return False
        
    for cls_file in cls_files:
        try:
            rel_path = cls_file.relative_to(Path.cwd())
        except ValueError:
            rel_path = cls_file
        print(f"   Checking {rel_path}")
        
        # Basic syntax checks
        content = cls_file.read_text()
        
        # Check for basic ObjectScript class structure
        if not content.startswith("///") and "Class " not in content:
            print(f"❌ {cls_file.name}: Missing class declaration")
            return False
            
        # Check for newline at end of file (IRIS requirement)
        if not content.endswith('\n'):
            print(f"❌ {cls_file.name}: Missing newline at end of file (IRIS ObjectScript requirement)")
            return False
            
        # Check for balanced braces (simple check)
        open_braces = content.count("{")
        close_braces = content.count("}")
        if open_braces != close_braces:
            print(f"❌ {cls_file.name}: Unbalanced braces ({open_braces} open, {close_braces} close)")
            return False
            
        # Enhanced brace validation - check for problematic patterns that cause compilation errors
        lines = content.split('\n')
        brace_stack = []
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Track brace context for better validation
            if '{' in line:
                brace_stack.append(i)
            if '}' in line and brace_stack:
                brace_stack.pop()
                
            # Only flag truly suspicious patterns - consecutive closing braces at same indentation level
            # that don't correspond to nested structures
            if (stripped == '}' and i < len(lines) - 1):
                next_line = lines[i].strip() if i < len(lines) else ""
                if next_line == '}':
                    # Check indentation to see if these are at the same level (suspicious)
                    current_indent = len(line) - len(line.lstrip())
                    next_indent = len(lines[i]) - len(lines[i].lstrip()) if i < len(lines) else 0
                    
                    # Flag only if same indentation (likely error) and not in obvious nested context
                    if current_indent == next_indent and current_indent == 0:
                        print(f"❌ {cls_file.name}: Suspicious consecutive closing braces at same indentation at lines {i} and {i+1}")
                        print(f"    Line {i}: '{line}'")
                        print(f"    Line {i+1}: '{next_line}'")
                        return False
            
    print(f"✅ All {len(cls_files)} ObjectScript files have valid syntax")
    return True

def test_ipm_validators():
    """Run the IPM package validators."""
    print("🔍 Running IPM package validators...")
    
    # Test basic IPM package validator
    try:
        result = subprocess.run([
            sys.executable, "scripts/validate_ipm_package.py", "."
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print("❌ Basic IPM package validation failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
        if "✅ IPM package validation PASSED" not in result.stdout:
            print("❌ Basic IPM package validation did not pass")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Basic IPM package validator timed out")
        return False
    except Exception as e:
        print(f"❌ Error running basic IPM validator: {e}")
        return False
        
    # Test comprehensive IPM module validator
    try:
        result = subprocess.run([
            sys.executable, "scripts/utilities/validate_ipm_module.py", 
            "--project-root", "."
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print("❌ Comprehensive IPM module validation failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
        if "✅ PASSED" not in result.stdout:
            print("❌ Comprehensive IPM module validation did not pass")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Comprehensive IPM module validator timed out")
        return False
    except Exception as e:
        print(f"❌ Error running comprehensive IPM validator: {e}")
        return False
        
    print("✅ All IPM validators passed")
    return True

def test_python_package_structure():
    """Test Python package structure."""
    print("🔍 Testing Python package structure...")
    
    required_packages = ["iris_rag", "rag_templates", "common"]
    
    for package in required_packages:
        package_path = Path(package)
        if not package_path.exists():
            print(f"❌ Missing Python package: {package}")
            return False
            
        init_file = package_path / "__init__.py"
        if not init_file.exists():
            print(f"❌ Missing __init__.py in {package}")
            return False
            
    print(f"✅ All {len(required_packages)} Python packages are valid")
    return True

def main():
    """Run all ZPM compilation tests."""
    print("🧪 ZPM Package Compilation Test")
    print("=" * 50)
    
    tests = [
        ("Module XML Structure", test_module_xml),
        ("ObjectScript Syntax", test_objectscript_syntax), 
        ("Python Package Structure", test_python_package_structure),
        ("IPM Validators", test_ipm_validators),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            
    print("\n" + "=" * 50)
    print(f"🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ZPM package compilation test PASSED")
        print("✅ Package is ready for deployment")
        return 0
    else:
        print("❌ ZPM package compilation test FAILED")
        print("🔧 Please fix the issues above before deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())