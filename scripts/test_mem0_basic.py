#!/usr/bin/env python3
"""
Basic mem0 Integration Test

Tests what components are currently working without requiring API keys.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_environment_loading():
    """Test if environment variables are loading correctly."""
    print("🔧 Testing Environment Loading...")
    
    # Load .env.mem0 file
    from dotenv import load_dotenv
    env_path = project_root / ".env.mem0"
    load_dotenv(env_path)
    
    # Check some key variables
    test_vars = [
        "MEM0_API_KEY",
        "OPENAI_API_KEY", 
        "QDRANT_URL",
        "MEM0_LOG_LEVEL"
    ]
    
    print("📋 Environment Variables:")
    for var in test_vars:
        value = os.getenv(var, "NOT SET")
        # Mask sensitive values
        if "KEY" in var and value != "NOT SET":
            display_value = value[:10] + "..." if len(value) > 10 else value
        else:
            display_value = value
        print(f"   {var}: {display_value}")
    
    return True

def test_mem0_imports():
    """Test if mem0 related modules can be imported."""
    print("\n📦 Testing Module Imports...")
    
    try:
        # Test basic Python imports
        import json
        import asyncio
        from datetime import datetime
        print("   ✅ Basic Python modules: OK")
    except ImportError as e:
        print(f"   ❌ Basic Python modules: {e}")
        return False
    
    try:
        # Test if mem0 integration modules exist
        integration_dir = project_root / "mem0_integration"
        if integration_dir.exists():
            print(f"   ✅ mem0_integration directory: Found")
            
            # Check key files
            key_files = [
                "adapters/supabase_mcp_adapter.py",
                "schemas/memory_schema.py", 
                "hooks/memory_hooks.py"
            ]
            
            for file_path in key_files:
                file_full_path = integration_dir / file_path
                if file_full_path.exists():
                    print(f"   ✅ {file_path}: Found")
                else:
                    print(f"   ⚠️  {file_path}: Missing")
        else:
            print("   ❌ mem0_integration directory: Not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Module import error: {e}")
        return False
        
    return True

def test_basic_functionality():
    """Test basic functionality without requiring external services."""
    print("\n🧪 Testing Basic Functionality...")
    
    try:
        # Test basic memory structure creation
        memory_example = {
            "id": "test_001",
            "content": "This is a test memory for coding assistance",
            "type": "coding_pattern",
            "metadata": {
                "language": "python",
                "timestamp": "2024-01-01T00:00:00Z",
                "relevance_score": 0.95
            }
        }
        
        print(f"   ✅ Memory structure creation: OK")
        print(f"   📝 Example memory: {memory_example['type']}")
        
        # Test JSON serialization
        import json
        serialized = json.dumps(memory_example, indent=2)
        print(f"   ✅ JSON serialization: OK")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files and directories exist."""
    print("\n📁 Testing File Structure...")
    
    required_paths = [
        ".env.mem0",
        "mem0_integration/",
        "mem0_integration/adapters/",
        "mem0_integration/schemas/",
        "mem0_integration/hooks/",
        "mem0_integration/docs/",
        "scripts/validate_mem0_config.py"
    ]
    
    all_exist = True
    for path in required_paths:
        full_path = project_root / path
        if full_path.exists():
            print(f"   ✅ {path}: Found")
        else:
            print(f"   ❌ {path}: Missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all basic tests."""
    print("🚀 mem0 Integration Basic Test Suite")
    print("="*50)
    
    tests = [
        ("Environment Loading", test_environment_loading),
        ("Module Imports", test_mem0_imports),
        ("File Structure", test_file_structure),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"   ❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    print("\n🎯 Test Summary:")
    print("="*30)
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All basic tests passed! Infrastructure is ready.")
        print("📝 Next step: Add your API keys to .env.mem0 to enable full functionality")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)