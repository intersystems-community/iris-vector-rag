print("DEBUG_IMPORTS: Script started")

print("DEBUG_IMPORTS: Importing os")
import os
print("DEBUG_IMPORTS: Imported os")

print("DEBUG_IMPORTS: Importing sys")
import sys
print("DEBUG_IMPORTS: Imported sys")

print("DEBUG_IMPORTS: Importing argparse")
import argparse
print("DEBUG_IMPORTS: Imported argparse")

print("DEBUG_IMPORTS: Importing logging")
import logging
print("DEBUG_IMPORTS: Imported logging")

print("DEBUG_IMPORTS: Importing pathlib.Path")
from pathlib import Path
print("DEBUG_IMPORTS: Imported pathlib.Path")

print("DEBUG_IMPORTS: Importing load_dotenv from dotenv")
from dotenv import load_dotenv
print("DEBUG_IMPORTS: Imported load_dotenv from dotenv")

print("DEBUG_IMPORTS: About to call load_dotenv()")
load_dotenv()
print("DEBUG_IMPORTS: load_dotenv() completed")

# Add project root to path - exact same logic as original script
print("DEBUG_IMPORTS: Setting up project root path")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"DEBUG_IMPORTS: Project root path setup completed. Project root: {project_root}")
print(f"DEBUG_IMPORTS: sys.path is now: {sys.path}")

print("DEBUG_IMPORTS: Importing ComprehensiveRAGASEvaluationFramework from comprehensive_ragas_evaluation")
# This will likely be the problematic one if it's an import issue
try:
    from comprehensive_ragas_evaluation import ComprehensiveRAGASEvaluationFramework
    print("DEBUG_IMPORTS: Imported ComprehensiveRAGASEvaluationFramework")
except Exception as e:
    print(f"DEBUG_IMPORTS: FAILED to import ComprehensiveRAGASEvaluationFramework: {e}")
    import traceback
    print(f"DEBUG_IMPORTS: Full traceback: {traceback.format_exc()}")

print("DEBUG_IMPORTS: Script finished")