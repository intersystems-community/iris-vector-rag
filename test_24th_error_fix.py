#!/usr/bin/env python3
"""
Quick test to verify ValidatedPipelineFactory constructor fix
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from iris_rag.config.manager import ConfigurationManager
    from common.iris_connection_manager import IRISConnectionManager
    from iris_rag.validation.factory import ValidatedPipelineFactory
    
    print("✓ All imports successful")
    
    # Test the constructor with connection_manager parameter
    config_manager = ConfigurationManager()
    connection_manager = IRISConnectionManager(config_manager)
    
    # This should now work without error
    factory = ValidatedPipelineFactory(config_manager, connection_manager=connection_manager)
    print("✓ ValidatedPipelineFactory constructor with connection_manager parameter works")
    
    # Test without connection_manager (backward compatibility)
    factory2 = ValidatedPipelineFactory(config_manager)
    print("✓ ValidatedPipelineFactory constructor without connection_manager parameter works")
    
    print("✅ 24th error category fix verified successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)