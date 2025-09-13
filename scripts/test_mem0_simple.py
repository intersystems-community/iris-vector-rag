#!/usr/bin/env python3
"""
Simple mem0 Self-Hosted Test Script
Tests basic mem0 functionality with minimal configuration.
"""

import os
import sys
from pathlib import Path

def test_mem0_basic():
    """Test basic mem0 functionality with OpenAI API key only."""
    
    print("🧪 Testing mem0 Self-Hosted Setup...")
    print("=" * 50)
    
    # Check if OpenAI API key is set
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("💡 Tip: Make sure you've set OPENAI_API_KEY in your .env.mem0 file")
        return False
    
    print(f"✅ OpenAI API Key found: {openai_key[:7]}...")
    
    # Test mem0 import
    try:
        import mem0
        print(f"✅ mem0 imported successfully (version: {mem0.__version__})")
    except ImportError as e:
        print(f"❌ Error importing mem0: {e}")
        print("💡 Tip: Install with 'pip install mem0ai'")
        return False
    
    # Test basic mem0 client creation
    try:
        from mem0 import Memory
        
        # Initialize mem0 client with default configuration for self-hosting
        # The default configuration will use OpenAI API key from environment
        m = Memory()
        print("✅ mem0 Memory client created successfully")
        
        # Test basic memory operations
        user_id = "test_user_001"
        
        # Add a memory
        result = m.add("I love programming in Python", user_id=user_id)
        print(f"✅ Memory added successfully: {result}")
        
        # Search memories
        memories = m.search("programming", user_id=user_id)
        print(f"✅ Memory search successful: Found {len(memories)} memories")
        
        # Get all memories
        all_memories = m.get_all(user_id=user_id)
        print(f"✅ Retrieved all memories: {len(all_memories)} total")
        
        print("\n🎉 mem0 Self-Hosted Test PASSED!")
        print("🚀 mem0 is ready for local development!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing mem0 functionality: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        return False

def load_env_file():
    """Load environment variables from .env and .env.mem0 files."""
    # First load the main .env file
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    # Then load .env.mem0 which may have variable substitutions
    mem0_env_file = Path('.env.mem0')
    if not mem0_env_file.exists():
        print("❌ .env.mem0 file not found")
        return False
    
    # Simple env file parser with variable substitution support
    with open(mem0_env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Handle variable substitution syntax ${VAR_NAME}
                if value.startswith('${') and value.endswith('}'):
                    var_name = value[2:-1]  # Extract variable name
                    if var_name in os.environ:
                        value = os.environ[var_name]
                    else:
                        print(f"⚠️  Warning: Variable {var_name} not found for substitution")
                        continue
                os.environ[key] = value
    
    return True

if __name__ == "__main__":
    print("🔧 Loading environment variables...")
    if not load_env_file():
        sys.exit(1)
    
    success = test_mem0_basic()
    sys.exit(0 if success else 1)