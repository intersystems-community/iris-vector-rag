# common/db_init.py
import os
import re
from typing import Any, List

def split_sql_commands(sql_script: str) -> List[str]:
    """
    Smart SQL splitter: preserves CREATE PROCEDURE ... END; blocks.
    Assumes each CREATE PROCEDURE ends with END;
    and no nested BEGIN...END inside procedures (valid for IRIS).
    Also handles simple statements ending with ;
    """
    statements = []
    current_statement_lines = []
    in_procedure_block = False

    # Normalize line endings and split into lines
    normalized_script = sql_script.replace('\r\n', '\n').replace('\r', '\n')
    lines = normalized_script.split('\n')

    for line in lines:
        stripped_line_upper = line.strip().upper()

        if stripped_line_upper.startswith("CREATE PROCEDURE"):
            if current_statement_lines: # Should not happen if procedures are top-level
                statements.append("\n".join(current_statement_lines).strip())
                current_statement_lines = []
            in_procedure_block = True
        
        if in_procedure_block:
            current_statement_lines.append(line)
            if stripped_line_upper.endswith("END;"):
                statements.append("\n".join(current_statement_lines).strip())
                current_statement_lines = []
                in_procedure_block = False
        else:
            # Handle non-procedure statements, splitting by semicolon
            # This part needs to correctly handle multiple statements on one line too
            # and accumulate lines for multi-line non-procedure statements.
            # For simplicity, let's assume non-procedure statements are single-line or
            # that the old splitting logic was mostly fine for them if they don't span lines
            # in a way that confuses the simple split.
            # A more robust general SQL splitter is complex.
            # This simplified logic will join lines until a semicolon is found for non-proc statements.
            
            current_statement_lines.append(line)
            if line.strip().endswith(";"):
                full_command = "\n".join(current_statement_lines).strip()
                # Further split if multiple simple commands were on these lines, separated by ;
                # This re-introduces the original problem for lines like "CMD1; CMD2" if not handled carefully.
                # The original split was: `raw_commands = [cmd.strip() for cmd in cleaned_sql_script.split(';') if cmd.strip()]`
                # Let's refine this part to be closer to the original for non-procedure blocks,
                # but ensure the procedure block is treated as one.

                # If we are here, it means we are NOT in a procedure block.
                # The lines accumulated in current_statement_lines might form one or more simple statements.
                temp_buffer = "\n".join(current_statement_lines)
                parts = temp_buffer.split(';')
                for i, part_str in enumerate(parts):
                    clean_part = part_str.strip()
                    if clean_part:
                        # If it's not the last part (which might be empty after a trailing semicolon)
                        # or if it is the last part and it's not empty
                        if i < len(parts) - 1 or clean_part:
                             statements.append(clean_part + (";" if i < len(parts) -1 and parts[i+1].strip() else "")) # Add back semicolon if it's not the true end
                current_statement_lines = []

    # Add any remaining content in the buffer
    # This will correctly add the last statement if it was a procedure that didn't find END;
    # (like a simple CREATE PROC... {body}) or any other trailing non-semicolon-terminated statement.
    if current_statement_lines:
        statements.append("\n".join(current_statement_lines).strip())
        
    # Filter out empty statements that might result from splitting
    return [s for s in statements if s]


def initialize_database(iris_connector: Any, force_recreate: bool = False):
    """
    Initializes the database schema by executing SQL scripts.

    Args:
        iris_connector: An active InterSystems IRIS database connection object.
        force_recreate: If True, implies that DROP statements in SQL should execute.
                        The SQL script itself handles idempotency with DROP IF EXISTS.
    """
    print("Initializing database schema...")
    
    # List of SQL files to execute in order
    sql_files_to_execute = [
        ('db_init_simple.sql', 'sql'),       # Simple, working schema definitions
        # Note: vector_similarity.sql and vector_search_procs.sql removed for simplicity
        # We'll handle vector operations directly in Python code
    ]

    # Path to the ObjectScript .cls file inside the Docker container
    # IMPORTANT: The RAG.VectorSearchUtils class from common/VectorSearch.cls
    # must be manually compiled into the IRIS database prior to running this.
    # This script will no longer attempt to compile it automatically.

    cursor = None
    try:
        cursor = iris_connector.cursor()

        # Process SQL files
        for item_name, item_type in sql_files_to_execute:
            if item_type == 'sql':
                sql_file_path = os.path.join(os.path.dirname(__file__), item_name)
                if not os.path.exists(sql_file_path):
                    print(f"WARNING: SQL file not found at {sql_file_path}, skipping")
                    continue

                print(f"Processing SQL file: {item_name}")
                try:
                    with open(sql_file_path, 'r', encoding='utf-8') as f:
                        sql_script_content = f.read()
                    
                    cleaned_sql_script = re.sub(r'--[^\r\n]*', '', sql_script_content) # Remove single-line comments
                    cleaned_sql_script = re.sub(r'/\*.*?\*/', '', cleaned_sql_script, flags=re.DOTALL) # Remove multi-line comments
                    
                    # Use the new splitting logic
                    raw_commands = split_sql_commands(cleaned_sql_script)
                    
                    ddl_dml_commands = []
                    grant_commands = []

                    for command in raw_commands:
                        if command.upper().startswith("GRANT"):
                            grant_commands.append(command)
                        elif command:
                            ddl_dml_commands.append(command)

                    for command in ddl_dml_commands:
                        print(f"Executing DDL/DML: {command[:100]}...")
                        cursor.execute(command)
                    
                    for command in grant_commands:
                        print(f"Executing GRANT: {command[:100]}...")
                        cursor.execute(command)
                            
                except Exception as e_sql:
                    print(f"ERROR: Could not process SQL file {sql_file_path}: {e_sql}")
                    if hasattr(iris_connector, 'rollback'):
                        iris_connector.rollback()
                    raise
        
        if hasattr(iris_connector, 'commit'):
            iris_connector.commit()
        
        print("Database schema initialized successfully.")

    except Exception as e_main:
        print(f"ERROR: Failed to initialize database schema: {e_main}")
        if hasattr(iris_connector, 'rollback'):
            iris_connector.rollback()
        raise  # Re-raise the exception to be handled by the caller
    finally:
        if cursor:
            cursor.close()

if __name__ == '__main__':
    # This is a simple test block.
    # In a real scenario, you'd get a proper iris_connector from your application.
    print("Testing common.db_init.initialize_database (requires a mock or live IRIS connection)")
    
    # Example of how it might be used (requires a live IRIS connection setup)
    # from testcontainers.iris import IRISContainer
    # import sqlalchemy

    # is_arm64 = os.uname().machine == 'arm64'
    # default_image = "intersystemsdc/iris-community:latest"
    # iris_image_tag = os.getenv("IRIS_DOCKER_IMAGE", default_image)

    # print(f"Attempting to use IRIS Docker image: {iris_image_tag}")
    # try:
    #     with IRISContainer(iris_image_tag) as iris_container:
    #         connection_url = iris_container.get_connection_url()
    #         print(f"IRIS Testcontainer started. Connection URL: {connection_url}")
    #         engine = sqlalchemy.create_engine(connection_url)
    #         raw_dbapi_connection = None
    #         try:
    #             with engine.connect() as sa_connection:
    #                 raw_dbapi_connection = sa_connection.connection
    #                 print("Initializing database...")
    #                 initialize_database(raw_dbapi_connection)
    #                 print("Database initialization test complete.")
    #         finally:
    #             if engine:
    #                 engine.dispose()
    # except Exception as e:
    #     print(f"Testcontainer setup or initialization failed: {e}")
    #     print("Please ensure Docker is running and the IRIS image is available.")

    print("common.db_init.py executed.")
    # Note: The __main__ block above is for illustrative purposes and won't run
    # unless this script is executed directly. It also requires Docker and testcontainers.
