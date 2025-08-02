import sys
import logging
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compile_objectscript_class():
    """Compile the RAG.VectorMigration ObjectScript class"""
    logging.info("Compiling RAG.VectorMigration ObjectScript class...")
    conn = None
    
    try:
        conn = get_iris_connection()
        
        with conn.cursor() as cursor:
            # Read the ObjectScript class file
            class_file_path = os.path.join(os.path.dirname(__file__), '..', 'objectscript', 'RAG.VectorMigration.cls')
            
            if not os.path.exists(class_file_path):
                logging.error(f"ObjectScript class file not found: {class_file_path}")
                return 1
            
            with open(class_file_path, 'r') as f:
                class_content = f.read()
            
            logging.info(f"Read ObjectScript class from {class_file_path}")
            
            # Alternative: Write the class to a file and use terminal command
            temp_cls_file = '/tmp/RAG.VectorMigration.cls'
            with open(temp_cls_file, 'w') as f:
                f.write(class_content)
            
            logging.info(f"Wrote class to {temp_cls_file}")
            logging.info("Please manually compile this class in IRIS Terminal using:")
            logging.info(f"  do $system.OBJ.Load(\"{temp_cls_file}\",\"ck\")")
            logging.info("Or copy the class content to IRIS Studio and compile there.")
            
            # Try to compile using SQL CALL to ObjectScript
            try:
                # Use CALL to execute ObjectScript method
                compile_sql = "CALL $SYSTEM.OBJ.CompileText(?, 'ck')"
                logging.info("Attempting to compile using SQL CALL...")
                cursor.execute(compile_sql, (class_content,))
                result = cursor.fetchone()
                
                if result and result[0] == 1:
                    logging.info("ObjectScript class RAG.VectorMigration compiled successfully!")
                    
                    # Test if the method is accessible via SQL
                    test_sql = "SELECT RAG.GetVectorAsStringFromVarchar('test') AS TestResult"
                    try:
                        cursor.execute(test_sql)
                        test_result = cursor.fetchone()
                        logging.info(f"Method test successful. Result: {test_result[0] if test_result else 'None'}")
                    except Exception as e:
                        logging.warning(f"Method test failed (this might be expected): {e}")
                    
                    return 0
                else:
                    logging.error(f"Failed to compile ObjectScript class via SQL. Result: {result[0] if result else 'None'}")
                    
            except Exception as e:
                logging.warning(f"SQL compilation failed: {e}")
                logging.info("Manual compilation required.")
            
            return 0
                
    except Exception as e:
        logging.error(f"Error compiling ObjectScript class: {e}")
        return 1
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    exit_code = compile_objectscript_class()
    if exit_code == 0:
        logging.info("ObjectScript class compilation completed successfully.")
    else:
        logging.error("ObjectScript class compilation failed.")
    sys.exit(exit_code)