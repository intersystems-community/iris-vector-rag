#!/usr/bin/env python3
"""
Direct test of mem0 functionality and error reproduction
"""

import os
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))


# Load environment variables from .env.mem0
def load_env_file(env_file_path):
    """Load environment variables from a file with shell variable expansion support"""
    import re

    if os.path.exists(env_file_path):
        with open(env_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)

                    # Handle shell variable expansion syntax ${VAR:-default}
                    def expand_var(match):
                        var_expr = match.group(1)
                        if ":-" in var_expr:
                            var_name, default_value = var_expr.split(":-", 1)
                            return os.environ.get(var_name, default_value)
                        else:
                            return os.environ.get(var_expr, "")

                    # Replace ${VAR:-default} patterns
                    value = re.sub(r"\$\{([^}]+)\}", expand_var, value)

                    # Set the environment variable
                    os.environ[key] = value
        print(f"✓ Loaded environment variables from {env_file_path}")
    else:
        print(f"✗ Environment file not found: {env_file_path}")


# Load the environment file
env_file = Path(__file__).parent.parent / ".env.mem0"
load_env_file(str(env_file))


try:
    from mem0_integration.adapters.supabase_mcp_adapter import SupabaseMCPAdapter

    print("✓ Successfully imported SupabaseMCPAdapter")
except Exception as e:
    print(f"✗ Failed to import SupabaseMCPAdapter: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback

    traceback.print_exc()
    sys.exit(1)


async def test_mem0_operations():
    """Test mem0 memory operations directly"""
    print("\n=== Testing mem0 Operations ===")

    try:
        # Create adapter instance
        adapter = SupabaseMCPAdapter()
        print("✓ Created SupabaseMCPAdapter instance")

        # Initialize adapter
        await adapter.initialize()
        print("✓ Initialized adapter")

        # Test storing a memory
        result = await adapter.store_memory(
            content="This is a test memory for debugging",
            user_id="test_user",
            metadata={"type": "debugging", "timestamp": "2025-01-01"},
        )
        print(f"✓ Stored memory: {result}")

        # Test retrieving memories
        memories = await adapter.get_memories(user_id="test_user", limit=10)
        print(f"✓ Retrieved memories: {len(memories)} found")
        for memory in memories:
            print(
                f"  - {memory.get('id', 'unknown')}: {memory.get('content', 'no content')[:50]}..."
            )

        # Test searching memories
        search_results = await adapter.search_memories(
            query="test debugging", user_id="test_user", limit=5
        )
        print(f"✓ Search results: {len(search_results)} found")

        return True

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import asyncio

    print("Starting mem0 direct testing...")
    success = asyncio.run(test_mem0_operations())

    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Tests failed!")
        sys.exit(1)
