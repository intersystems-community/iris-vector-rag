#!/usr/bin/env python3
"""
Configure IRIS license and verify Vector Search is enabled.
"""

import sys
import os
import time
import subprocess

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.iris_connector import get_iris_connection

def wait_for_iris_ready(max_attempts=30):
    """Wait for IRIS to be ready to accept connections."""
    print("Waiting for IRIS to be ready...")
    
    for attempt in range(max_attempts):
        try:
            conn = get_iris_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            print("✓ IRIS is ready!")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_attempts}: IRIS not ready yet ({e})")
            time.sleep(2)
    
    print("✗ IRIS failed to become ready")
    return False

def configure_license():
    """Configure the license in IRIS."""
    print("Configuring IRIS license...")
    
    try:
        # Copy license file into the container
        result = subprocess.run([
            "docker", "cp", "./iris.key", "iris_db_rag_licensed:/usr/irissys/mgr/iris.key"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ License file copied to container")
        else:
            print(f"✗ Failed to copy license file: {result.stderr}")
            return False
        
        # Restart IRIS to pick up the license
        print("Restarting IRIS to apply license...")
        subprocess.run([
            "docker", "exec", "iris_db_rag_licensed", "iris", "restart", "iris"
        ], capture_output=True)
        
        # Wait for restart
        time.sleep(10)
        
        return wait_for_iris_ready()
        
    except Exception as e:
        print(f"✗ License configuration failed: {e}")
        return False

def verify_vector_search_license():
    """Verify that Vector Search is enabled in the license."""
    print("Verifying Vector Search license...")
    
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Check Vector Search feature
        cursor.execute("SELECT $SYSTEM.License.GetFeature('Vector Search')")
        result = cursor.fetchone()
        
        if result and result[0] == 1:
            print("✓ Vector Search is enabled in the license!")
            return True
        else:
            print("✗ Vector Search is not enabled in the license")
            return False
            
    except Exception as e:
        print(f"✗ License verification failed: {e}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def main():
    """Main configuration process."""
    print("IRIS 2025.1 License Configuration")
    print("=" * 50)
    
    # Wait for IRIS to be ready
    if not wait_for_iris_ready():
        return False
    
    # Configure license
    if not configure_license():
        return False
    
    # Verify Vector Search
    if not verify_vector_search_license():
        return False
    
    print("\n🎉 IRIS 2025.1 with Vector Search is ready!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)