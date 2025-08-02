import sys
#!/usr/bin/env python3
"""
Ultimate Zero-to-RAGAS Demonstration

This script shows the complete RAG pipeline from absolute zero to RAGAS results
with maximum visibility into every step of the process.
"""
import subprocess
import time
import json
import argparse
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

# Attempt to import yaml, otherwise use a flag
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Docker/Compose Management Functions ---

def run_command(
    command_args: list,
    description: str,
    verbose: bool = False,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    stdin_content: Optional[str] = None
) -> bool:
    """Runs a generic command and handles output."""
    logger.info(f"Executing: {' '.join(command_args)} ({description})")
    try:
        process = subprocess.Popen(
            command_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=cwd,
            stdin=subprocess.PIPE if stdin_content else None
        )
        # If text=True, communicate expects string input.
        # stdin_content is already a string.
        stdout_data, stderr_data = process.communicate(input=stdin_content if stdin_content else None)
        
        if verbose and stdout_data:
            for line in stdout_data.splitlines():
                logger.info(line.strip())
        
        # process.wait() is implicitly called by communicate()

        if process.returncode != 0:
            logger.error(f"Error: '{' '.join(command_args)}' failed with return code {process.returncode}")
            # Log remaining output if any
            # Note: process.stdout will be None after communicate(), use stdout_data
            if stdout_data: # Check if there's any captured stdout
                 for line in stdout_data.splitlines(): # Iterate over captured stdout
                     logger.error(line.strip())
            return False
        logger.info(f"'{' '.join(command_args)}' completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Exception during '{' '.join(command_args)}': {e}")
        return False

def run_compose_command(
    sub_command_args: list,
    description: str,
    verbose: bool = False,
    env: Optional[Dict[str, str]] = None,
    main_compose_file: Optional[Path] = Path("docker-compose.yml"), # Optional if stdin_content is used
    override_compose_file: Optional[Path] = None, # This will be deprecated by stdin usage
    stdin_content: Optional[str] = None
) -> bool:
    """Helper to run docker-compose commands, optionally with an override file or stdin content."""
    
    actual_main_compose_file = main_compose_file
    compose_file_source_description = ""
    cwd_path = project_root_path() # Default CWD to project root

    if stdin_content:
        command = ["docker-compose", "-f", "-"]
        compose_file_source_description = "from stdin"
        # When using stdin, CWD might still be relevant if the compose content refers to relative paths
        # for build contexts or volume mounts.
        if actual_main_compose_file and actual_main_compose_file.exists():
            # If a base file path is provided (even with stdin), use its parent for CWD
            # This helps resolve relative paths within the compose data if they exist
            if not actual_main_compose_file.is_absolute():
                 resolved_base_file = project_root_path() / actual_main_compose_file
                 if resolved_base_file.exists():
                     cwd_path = resolved_base_file.parent
                 else: # Fallback if relative path from project root doesn't exist
                     parent_dir_candidate = Path("..") / actual_main_compose_file
                     if parent_dir_candidate.resolve().exists(): # Check resolved existence
                         cwd_path = parent_dir_candidate.resolve().parent
            else: # Absolute path
                cwd_path = actual_main_compose_file.parent
        # If no main_compose_file, cwd_path remains project_root_path()
        
    elif actual_main_compose_file:
        if not actual_main_compose_file.is_absolute() and not actual_main_compose_file.exists():
            project_root = project_root_path()
            candidate_path = project_root / actual_main_compose_file
            if candidate_path.exists():
                actual_main_compose_file = candidate_path
            else:
                parent_dir_candidate = Path("..") / actual_main_compose_file # Original fallback
                if parent_dir_candidate.resolve().exists(): # Check resolved existence
                    actual_main_compose_file = parent_dir_candidate.resolve()

        if not actual_main_compose_file.exists():
            logger.error(f"Main docker-compose file '{main_compose_file}' not found. Cannot run command.")
            return False
        
        command = ["docker-compose", "-f", str(actual_main_compose_file)]
        compose_file_source_description = f"from {actual_main_compose_file.name}"
        cwd_path = actual_main_compose_file.parent

        # Handle override file if not using stdin
        is_standard_override = (
            override_compose_file and
            override_compose_file.name == "docker-compose.override.yml" and
            override_compose_file.parent == actual_main_compose_file.parent
        )
        if override_compose_file and override_compose_file.exists() and not is_standard_override:
            command.extend(["-f", str(override_compose_file)])
            compose_file_source_description += f" and override {override_compose_file.name}"
        elif is_standard_override:
            logger.info(f"Relying on implicit override for {override_compose_file.name}")
    else:
        logger.error("No docker-compose file or stdin content provided.")
        return False

    command.extend(sub_command_args)
    return run_command(command, f"{description} (compose config {compose_file_source_description})", verbose, env, cwd=str(cwd_path), stdin_content=stdin_content)

def ensure_iris_down(verbose: bool = False, compose_file_path: Path = Path("docker-compose.yml")) -> bool:
    """Ensures IRIS Docker container is down and volumes are removed."""
    logger.info("Ensuring IRIS is completely down (including volumes)...")
    return run_compose_command(["down", "-v", "--remove-orphans"], "Docker Compose Down", verbose, main_compose_file=compose_file_path)

def get_iris_password_from_compose(compose_file_path: Path = Path("docker-compose.yml")) -> str:
    """Parses docker-compose.yml to get ISC_DEFAULT_PASSWORD."""
    default_password = "SYS" # Fallback password
    
    actual_compose_file_path = compose_file_path
    if not actual_compose_file_path.is_absolute() and not actual_compose_file_path.exists():
        project_root = project_root_path()
        candidate_path = project_root / compose_file_path
        if candidate_path.exists():
            actual_compose_file_path = candidate_path
        else: # Fallback to parent dir of script if not found at root
            parent_dir_candidate = Path("..") / compose_file_path
            if parent_dir_candidate.exists():
                actual_compose_file_path = parent_dir_candidate

    if not actual_compose_file_path.exists():
        logger.warning(f"Main docker-compose file '{compose_file_path}' not found at expected locations. Using default password '{default_password}'.")
        return default_password

    if YAML_AVAILABLE:
        try:
            with open(actual_compose_file_path, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            environment_vars = compose_config.get('services', {}).get('iris_db', {}).get('environment', [])
            if isinstance(environment_vars, dict): # Handle case where environment is a dict
                password = environment_vars.get('ISC_DEFAULT_PASSWORD')
            elif isinstance(environment_vars, list): # Handle case where environment is a list
                password = None
                for var in environment_vars:
                    if var.startswith("ISC_DEFAULT_PASSWORD="):
                        password = var.split("=", 1)[1]
                        break
            else:
                password = None

            if password:
                logger.info(f"Successfully parsed ISC_DEFAULT_PASSWORD: '{password}' from {actual_compose_file_path}")
                return password
            else:
                logger.warning(f"ISC_DEFAULT_PASSWORD not found in {actual_compose_file_path} or environment section has unexpected format. Using default password '{default_password}'.")
        except Exception as e:
            logger.error(f"Error parsing {actual_compose_file_path}: {e}. Using default password '{default_password}'.")
    else:
        logger.warning(f"PyYAML not installed. Cannot parse {actual_compose_file_path}. Using default password '{default_password}'.")
    
    return default_password

def start_iris_and_wait(
    verbose: bool = False,
    compose_file_path: Path = Path("docker-compose.yml"),
    max_port_attempts: int = 5,
    base_superserver_port: int = 1972,
    base_management_port: int = 52773
) -> Tuple[Optional[int], Optional[str]]:
    """
    Starts IRIS DB service, trying different ports if defaults are taken, and waits for it to be healthy.
    Returns a tuple of (actual_superserver_port, iris_password) or (None, None) on failure.
    """
    logger.info("Attempting to start IRIS DB service...")
    
    actual_compose_file_path = compose_file_path
    if not actual_compose_file_path.is_absolute() and not actual_compose_file_path.exists():
        project_root = project_root_path()
        candidate_path = project_root / compose_file_path
        if candidate_path.exists():
            actual_compose_file_path = candidate_path
        else: # Fallback to parent dir of script
            parent_dir_candidate = Path("..") / compose_file_path
            if parent_dir_candidate.resolve().exists():
                actual_compose_file_path = parent_dir_candidate.resolve()
    
    if not actual_compose_file_path.exists():
        logger.error(f"Main docker-compose file '{compose_file_path}' not found. Cannot start IRIS.")
        return None, None

    iris_password = get_iris_password_from_compose(actual_compose_file_path)
    # override_file_path is no longer used

    for attempt in range(max_port_attempts):
        current_superserver_port = base_superserver_port + attempt
        current_management_port = base_management_port + attempt
        
        logger.info(f"Attempt {attempt + 1}/{max_port_attempts}: Trying SuperServer port {current_superserver_port}, Management port {current_management_port}")

        if YAML_AVAILABLE:
            try:
                with open(actual_compose_file_path, 'r') as f:
                    compose_config = yaml.safe_load(f)
                
                if 'services' not in compose_config or 'iris_db' not in compose_config['services']:
                    logger.error(f"Invalid docker-compose.yml structure in {actual_compose_file_path}. Missing services.iris_db. Cannot modify ports.")
                    # This is a critical error with the base compose file, so we probably shouldn't proceed with this attempt.
                    success = False
                else:
                    compose_config['services']['iris_db']['ports'] = [
                        f"{current_superserver_port}:1972",
                        f"{current_management_port}:52773"
                    ]
                    modified_yaml_str = yaml.dump(compose_config)

                    up_command_args_with_wait = ["up", "-d", "--wait", "iris_db"]
                    up_command_args_no_wait = ["up", "-d", "iris_db"]

                    success = run_compose_command(
                        up_command_args_with_wait,
                        f"Docker Compose Up",
                        verbose,
                        main_compose_file=actual_compose_file_path,
                        stdin_content=modified_yaml_str
                    )

                    if not success:
                        logger.warning("`docker-compose up --wait` failed. Retrying without --wait...")
                        ensure_iris_down(verbose, compose_file_path=actual_compose_file_path)
                        success = run_compose_command(
                            up_command_args_no_wait,
                            f"Docker Compose Up (fallback)",
                            verbose,
                            main_compose_file=actual_compose_file_path,
                            stdin_content=modified_yaml_str
                        )
                        if success:
                            logger.info("Fallback `docker-compose up` (no --wait) succeeded. Allowing time for service to stabilize...")
                            time.sleep(15) # Give some time for service to potentially become healthy

            except Exception as e:
                logger.error(f"Error during PyYAML processing or docker-compose execution with stdin: {e}")
                success = False
        else: # YAML_AVAILABLE is False
            if attempt > 0:
                logger.warning("PyYAML not available, cannot try alternative ports. Sticking to default ports attempt.")
                break
            logger.warning("PyYAML not installed. Attempting to start IRIS with default ports from main docker-compose.yml.")
            up_command_args_with_wait = ["up", "-d", "--wait", "iris_db"]
            up_command_args_no_wait = ["up", "-d", "iris_db"]
            
            success = run_compose_command(
                up_command_args_with_wait,
                "Docker Compose Up with Wait (Default Ports)",
                verbose,
                main_compose_file=actual_compose_file_path
            )
            if not success:
                logger.warning("`docker-compose up --wait` (default ports) failed. Retrying without --wait...")
                ensure_iris_down(verbose, compose_file_path=actual_compose_file_path)
                success = run_compose_command(
                    up_command_args_no_wait,
                    "Docker Compose Up (Default Ports, fallback)",
                    verbose,
                    main_compose_file=actual_compose_file_path
                )
                if success:
                    logger.info("Fallback `docker-compose up` (no --wait, default ports) succeeded. Allowing time for service to stabilize...")
                    time.sleep(15)


        if success:
            logger.info(f"IRIS DB service started successfully on SuperServer port {current_superserver_port}.")
            # No override_file_path to clean up
            return current_superserver_port, iris_password
        else:
            logger.warning(f"Failed to start IRIS DB on SuperServer port {current_superserver_port}.")
            logger.info("Ensuring IRIS is down before next port attempt...")
            ensure_iris_down(verbose, compose_file_path=actual_compose_file_path)

            if not YAML_AVAILABLE:
                logger.error("Failed to start IRIS with default ports (PyYAML not available for dynamic ports).")
                break

    logger.error(f"Failed to start IRIS DB service after {max_port_attempts} attempts.")
    # No override_file_path to remove
    return None, None

# --- Original Script Functions (modified) ---

def run_make_command(target: str, description: str, verbose: bool = False, env: Optional[Dict[str, str]] = None) -> bool:
    """Runs a make command and handles output."""
    command = ["make", target]
    return run_command(command, description, verbose, env, cwd=str(project_root_path()))

def project_root_path() -> Path:
    """Returns the project root path, assuming this script is in a 'scripts' subdirectory."""
    # If this script is /path/to/project/scripts/script.py, root is /path/to/project
    return Path(__file__).parent.parent.resolve()

# Placeholder for actual implementation, will be filled in later
# For now, these functions will just print their purpose.

def show_initial_state(verbose: bool = False, env: Optional[Dict[str, str]] = None):
    """Show current database state before starting."""
    logger.info("\n📊 STEP 1: INITIAL DATABASE STATE")
    logger.info("-" * 50)
    logger.info("Showing current document counts, table sizes, etc. (Placeholder)")
    # Example: run_make_command("show-db-status", "Show DB Status", verbose, env)
    # Example: query database directly for counts

def clear_database(verbose: bool = False, env: Optional[Dict[str, str]] = None):
    """Clear database and show verification."""
    logger.info("\n🧹 STEP 2: CLEARING DATABASE")
    logger.info("-" * 50)
    if not run_make_command("clear-rag-data", "Clear RAG Data", verbose, env):
        logger.error("Failed to clear database. Aborting.")
        exit(1)
    logger.info("Database cleared. Verifying... (Placeholder)")
    # Example: run_make_command("show-db-status --empty-check", "Verify DB Empty", verbose, env)

def load_documents_with_progress(verbose: bool = False, env: Optional[Dict[str, str]] = None):
    """Load documents and show progress."""
    logger.info("\n📚 STEP 3: LOADING DOCUMENTS")
    logger.info("-" * 50)
    # Assuming make load-1000 or similar target exists and shows progress
    if not run_make_command("load-1000", "Load 1000 Documents", verbose, env): # Or a more specific demo target
        logger.error("Failed to load documents. Aborting.")
        exit(1)
    logger.info("Documents loaded. Showing summary... (Placeholder)")
    # Example: run_make_command("show-doc-load-summary", "Show Doc Load Summary", verbose, env)

def show_chunking_details(verbose: bool = False, env: Optional[Dict[str, str]] = None):
    """Show document chunking process."""
    logger.info("\n✂️ STEP 4: DOCUMENT CHUNKING")
    logger.info("-" * 50)
    logger.info("Displaying sample chunks, sizes, overlap... (Placeholder)")
    # This might involve querying the DB for chunked data or running a specific script
    # Example: python scripts/show_sample_chunks.py --count 5

def show_embedding_process(verbose: bool = False, env: Optional[Dict[str, str]] = None):
    """Show embedding generation."""
    logger.info("\n🧠 STEP 5: EMBEDDING GENERATION")
    logger.info("-" * 50)
    logger.info("Displaying embedding dimensions, sample vectors... (Placeholder)")
    # This might involve querying the DB for sample embeddings
    # Example: python scripts/show_sample_embeddings.py --count 3

def show_vector_storage(verbose: bool = False, env: Optional[Dict[str, str]] = None):
    """Show vector storage in IRIS."""
    logger.info("\n💾 STEP 6: VECTOR STORAGE")
    logger.info("-" * 50)
    logger.info("Displaying vector table sizes, sample stored vectors... (Placeholder)")
    # Example: run_make_command("show-vector-db-status", "Show Vector DB Status", verbose, env)

def demonstrate_search(verbose: bool = False, env: Optional[Dict[str, str]] = None):
    """Demonstrate search functionality."""
    logger.info("\n🔍 STEP 7: SEARCH DEMONSTRATION")
    logger.info("-" * 50)
    sample_query = "What is the role of apoptosis in cancer?"
    logger.info(f"Demonstrating search with query: '{sample_query}' (Placeholder)")
    # Example: python scripts/run_sample_search.py --query "{sample_query}"
    # Or: run_make_command(f"search QUERY='{sample_query}'", "Sample Search", verbose, env)

def show_rag_generation(verbose: bool = False, env: Optional[Dict[str, str]] = None):
    """Show RAG answer generation."""
    logger.info("\n🤖 STEP 8: RAG ANSWER GENERATION")
    logger.info("-" * 50)
    sample_question = "Explain the mechanism of CRISPR-Cas9."
    logger.info(f"Generating RAG response for: '{sample_question}' (Placeholder)")
    # Example: python scripts/run_sample_rag.py --question "{sample_question}"
    # Or: run_make_command(f"rag-generate QUESTION='{sample_question}'", "Sample RAG", verbose, env)

def run_ragas_evaluation(verbose: bool = False, env: Optional[Dict[str, str]] = None):
    """Run RAGAS evaluation."""
    logger.info("\n📈 STEP 9: RAGAS EVALUATION")
    logger.info("-" * 50)
    if not run_make_command("ragas-full", "Run Full RAGAS Evaluation", verbose, env): # Or a specific demo RAGAS target
        logger.error("RAGAS evaluation failed. Aborting.")
        exit(1)
    logger.info("RAGAS evaluation completed.")

def analyze_final_results(verbose: bool = False, env: Optional[Dict[str, str]] = None):
    """Analyze and display final results."""
    logger.info("\n🎯 STEP 10: FINAL RESULTS ANALYSIS")
    logger.info("-" * 50)
    logger.info("Parsing RAGAS results and displaying metrics... (Placeholder)")
    # This would involve reading the RAGAS output files (e.g., JSON/CSV)
    # and printing a formatted summary.
    # Example: python scripts/parse_ragas_results.py --latest

def main():
    parser = argparse.ArgumentParser(description="Ultimate Zero-to-RAGAS Demonstration Script")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output for all commands")
    parser.add_argument(
        "--compose-file",
        type=str,
        default="docker-compose.yml",
        help="Path to the docker-compose.yml file (relative to project root or absolute)."
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("🚀 ULTIMATE ZERO-TO-RAGAS DEMONSTRATION")
    logger.info("=" * 80)

    start_time = time.time()
    iris_password: Optional[str] = None
    actual_iris_port: Optional[int] = None

    compose_file_path_arg = Path(args.compose_file)
    # Resolve compose_file_path relative to project root if it's not absolute
    # This path will be further validated inside start_iris_and_wait and other compose functions
    if not compose_file_path_arg.is_absolute():
        main_compose_file_path = project_root_path() / compose_file_path_arg
    else:
        main_compose_file_path = compose_file_path_arg

    try:
        # Ensure IRIS is down first for a clean slate
        if not ensure_iris_down(args.verbose, compose_file_path=main_compose_file_path):
            logger.error("Failed to bring IRIS down. Aborting.")
            exit(1)

        # Start IRIS and get port and password
        actual_iris_port, iris_password = start_iris_and_wait(
            args.verbose,
            compose_file_path=main_compose_file_path
        )

        if actual_iris_port is None or iris_password is None:
            logger.error("Failed to start IRIS or retrieve necessary details. Aborting.")
            exit(1)

        logger.info(f"IRIS started successfully on SuperServer port: {actual_iris_port}")
        logger.info(f"Using IRIS password: '{iris_password}' (or default if parsing failed/unavailable)")

        # Prepare environment variables for make targets
        script_env = os.environ.copy()
        script_env["IRIS_USERNAME"] = "_SYSTEM" # Standard IRIS superuser
        script_env["IRIS_PASSWORD"] = iris_password
        script_env["ISC_DEFAULT_PASSWORD"] = iris_password # For consistency
        script_env["IRIS_PORT"] = str(actual_iris_port) # Pass the dynamically found port

        # Step 1: Database State (ZERO)
        step_start_time = time.time()
        show_initial_state(args.verbose, env=script_env)
        logger.info(f"Step 1 duration: {time.time() - step_start_time:.2f}s")
        
# NEW STEP: Setup Database Schema
        logger.info("\n🛠️ STEP 1.5: SETUP DATABASE SCHEMA")
        logger.info("-" * 50)
        step_start_time = time.time()
        if not run_make_command("setup-db", "Setup Database Schema", args.verbose, env=script_env):
            logger.error("Failed to setup database schema. Aborting.")
            # ensure_iris_down() might be called in finally, but good to be explicit if aborting early
            exit(1) 
        logger.info(f"Step 1.5 duration: {time.time() - step_start_time:.2f}s")
        # Step 2: Clear all data
        step_start_time = time.time()
        clear_database(args.verbose, env=script_env)
        logger.info(f"Step 2 duration: {time.time() - step_start_time:.2f}s")
        
        # Step 3: Load documents with progress
        step_start_time = time.time()
        load_documents_with_progress(args.verbose, env=script_env)
        logger.info(f"Step 3 duration: {time.time() - step_start_time:.2f}s")
        
        # Step 4: Show chunking details
        step_start_time = time.time()
        show_chunking_details(args.verbose, env=script_env)
        logger.info(f"Step 4 duration: {time.time() - step_start_time:.2f}s")
        
        # Step 5: Show embedding generation
        step_start_time = time.time()
        show_embedding_process(args.verbose, env=script_env)
        logger.info(f"Step 5 duration: {time.time() - step_start_time:.2f}s")
        
        # Step 6: Show vector storage
        step_start_time = time.time()
        show_vector_storage(args.verbose, env=script_env)
        logger.info(f"Step 6 duration: {time.time() - step_start_time:.2f}s")
        
        # Step 7: Demonstrate search
        step_start_time = time.time()
        demonstrate_search(args.verbose, env=script_env)
        logger.info(f"Step 7 duration: {time.time() - step_start_time:.2f}s")
        
        # Step 8: Show RAG generation
        step_start_time = time.time()
        show_rag_generation(args.verbose, env=script_env)
        logger.info(f"Step 8 duration: {time.time() - step_start_time:.2f}s")
        
        # Step 9: Run RAGAS evaluation
        step_start_time = time.time()
        run_ragas_evaluation(args.verbose, env=script_env)
        logger.info(f"Step 9 duration: {time.time() - step_start_time:.2f}s")
        
        # Step 10: Final results analysis
        step_start_time = time.time()
        analyze_final_results(args.verbose, env=script_env)
        logger.info(f"Step 10 duration: {time.time() - step_start_time:.2f}s")

    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}", exc_info=True)
    finally:
        logger.info("Performing final cleanup: Ensuring IRIS is down...")
        # Use the same main_compose_file_path for consistency in cleanup
        # The override file for 'down' should also be considered if it was used for 'up'
        # and might still exist if 'up' failed mid-process.
        override_file_for_down = main_compose_file_path.parent / "docker-compose.override.yml"
        current_override_for_down = override_file_for_down if YAML_AVAILABLE and override_file_for_down.exists() else None

        if not run_compose_command(
            ["down", "-v", "--remove-orphans"],
            "Final Docker Compose Down",
            args.verbose,
            main_compose_file=main_compose_file_path,
            override_compose_file=current_override_for_down
        ):
            logger.error("Failed to bring IRIS down during final cleanup.")
        else:
            logger.info("IRIS successfully brought down during final cleanup.")

        # Final explicit cleanup of the override file, if it exists and YAML was available
        if YAML_AVAILABLE and override_file_for_down.exists():
            try:
                os.remove(override_file_for_down)
                logger.info(f"Ensured cleanup of {override_file_for_down} in final block.")
            except OSError as e:
                logger.warning(f"Could not remove override file {override_file_for_down} in final block: {e}")

    logger.info("=" * 80)
    logger.info(f"🏁 ULTIMATE DEMONSTRATION COMPLETED IN: {time.time() - start_time:.2f}s")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
    sys.exit(0) # Explicitly exit with success code