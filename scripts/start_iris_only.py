import sys
from pathlib import Path

# Add project root to path to allow importing from ultimate_zero_to_ragas_demo
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from scripts.ultimate_zero_to_ragas_demo import start_iris_and_wait, ensure_iris_down

if __name__ == "__main__":
    # First, ensure any previous instances are down to release ports
    print("Ensuring IRIS is down before starting...")
    ensure_iris_down()
    
    print("Starting IRIS service...")
    # Now, start the service, allowing it to find an open port
    actual_port, password = start_iris_and_wait()
    
    if actual_port and password:
        print(f"IRIS service started successfully.")
        print(f"SuperServer Port: {actual_port}")
        print(f"Management Port: {actual_port + 30961}") # Default offset
        print(f"Password: {password}")
    else:
        print("Failed to start IRIS service.")