#!/usr/bin/env python3
"""
Technical Information Gathering Script

This script gathers technical information about the environment, including:
1. IRIS version (using $ZV)
2. Client library versions
3. SQL query behavior with different approaches

The information is output in Markdown format for easy inclusion in documentation.
"""

import os
import sys
import platform
import logging
from typing import Dict, Any, List

# Add project root to path to import from common/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gather_technical_info")

def get_python_info() -> Dict[str, str]:
    """Get Python version and platform information."""
    return {
        "Python Version": platform.python_version(),
        "Python Implementation": platform.python_implementation(),
        "Operating System": f"{platform.system()} {platform.release()}",
        "Platform": platform.platform()
    }

def get_client_library_versions() -> Dict[str, str]:
    """Get versions of relevant client libraries."""
    libraries = {}
    
    # Check for pyodbc
    try:
        import pyodbc
        libraries["pyodbc"] = pyodbc.version
    except ImportError:
        libraries["pyodbc"] = "Not installed"
    
    # Check for sqlalchemy
    try:
        import sqlalchemy
        libraries["sqlalchemy"] = sqlalchemy.__version__
    except ImportError:
        libraries["sqlalchemy"] = "Not installed"
    
    # Check for sqlalchemy-iris
    try:
        import sqlalchemy_iris
        libraries["sqlalchemy-iris"] = getattr(sqlalchemy_iris, "__version__", "Unknown")
    except ImportError:
        libraries["sqlalchemy-iris"] = "Not installed"
    
    # Check for langchain-iris
    try:
        import langchain_iris
        libraries["langchain-iris"] = getattr(langchain_iris, "__version__", "Unknown")
    except ImportError:
        libraries["langchain-iris"] = "Not installed"
    
    # Check for llama-iris
    try:
        import llama_iris
        libraries["llama-iris"] = getattr(llama_iris, "__version__", "Unknown")
    except ImportError:
        libraries["llama-iris"] = "Not installed"
    
    return libraries

def get_iris_version() -> str:
    """Get IRIS version using $ZV."""
    from common.iris_connector import get_iris_connection
    
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Execute $ZV to get IRIS version
        cursor.execute("SELECT $ZV")
        version = cursor.fetchone()[0]
        
        cursor.close()
        return version
    except Exception as e:
        logger.error(f"Error getting IRIS version: {e}")
        return "Unknown (Error connecting to IRIS)"

def test_sql_queries() -> Dict[str, Any]:
    """Test different SQL query approaches and capture behavior."""
    from common.iris_connector import get_iris_connection
    
    results = {
        "Direct SQL": {"success": False, "error": None, "query": None},
        "Parameterized SQL": {"success": False, "error": None, "query": None},
        "String Interpolation": {"success": False, "error": None, "query": None}
    }
    
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Set up test table
        cursor.execute("DROP TABLE IF EXISTS TechnicalInfoTest")
        cursor.execute("""
        CREATE TABLE TechnicalInfoTest (
            id VARCHAR(100) PRIMARY KEY,
            embedding VARCHAR(1000)
        )
        """)
        
        # Insert test data
        embedding_str = "0.1,0.2,0.3,0.4,0.5"
        cursor.execute(
            "INSERT INTO TechnicalInfoTest (id, embedding) VALUES (?, ?)",
            ("test1", embedding_str)
        )
        
        # Test 1: Direct SQL with TO_VECTOR
        direct_sql = f"""
        SELECT id, VECTOR_COSINE(
            TO_VECTOR(embedding, 'DOUBLE', 5),
            TO_VECTOR('{embedding_str}', 'DOUBLE', 5)
        ) AS score
        FROM TechnicalInfoTest
        """
        
        results["Direct SQL"]["query"] = direct_sql
        
        try:
            cursor.execute(direct_sql)
            results["Direct SQL"]["success"] = True
        except Exception as e:
            results["Direct SQL"]["success"] = False
            results["Direct SQL"]["error"] = str(e)
        
        # Test 2: Parameterized SQL with TO_VECTOR
        param_sql = """
        SELECT id, VECTOR_COSINE(
            TO_VECTOR(embedding, 'DOUBLE', 5),
            TO_VECTOR(?, 'DOUBLE', 5)
        ) AS score
        FROM TechnicalInfoTest
        """
        
        results["Parameterized SQL"]["query"] = param_sql
        
        try:
            cursor.execute(param_sql, (embedding_str,))
            results["Parameterized SQL"]["success"] = True
        except Exception as e:
            results["Parameterized SQL"]["success"] = False
            results["Parameterized SQL"]["error"] = str(e)
        
        # Test 3: String Interpolation with TO_VECTOR
        interp_sql = f"""
        SELECT id, VECTOR_COSINE(
            TO_VECTOR(embedding, 'DOUBLE', 5),
            TO_VECTOR('{embedding_str}', 'DOUBLE', 5)
        ) AS score
        FROM TechnicalInfoTest
        """
        
        results["String Interpolation"]["query"] = interp_sql
        
        try:
            cursor.execute(interp_sql)
            results["String Interpolation"]["success"] = True
        except Exception as e:
            results["String Interpolation"]["success"] = False
            results["String Interpolation"]["error"] = str(e)
        
        # Clean up
        cursor.execute("DROP TABLE IF EXISTS TechnicalInfoTest")
        conn.commit()
        
    except Exception as e:
        logger.error(f"Error testing SQL queries: {e}")
    
    return results

def generate_markdown() -> str:
    """Generate Markdown output with all technical information."""
    python_info = get_python_info()
    library_versions = get_client_library_versions()
    iris_version = get_iris_version()
    sql_results = test_sql_queries()
    
    markdown = "# Technical Environment Information\n\n"
    
    # Environment Information
    markdown += "## Environment Information\n\n"
    markdown += "| Component | Version/Details |\n"
    markdown += "|-----------|----------------|\n"
    markdown += f"| IRIS Version | {iris_version} |\n"
    markdown += f"| Python Version | {python_info['Python Version']} |\n"
    markdown += f"| Operating System | {python_info['Operating System']} |\n"
    markdown += f"| Platform | {python_info['Platform']} |\n"
    
    # Client Library Versions
    markdown += "\n## Client Library Versions\n\n"
    markdown += "| Library | Version |\n"
    markdown += "|---------|--------|\n"
    for lib, version in library_versions.items():
        markdown += f"| {lib} | {version} |\n"
    
    # SQL Query Test Results
    markdown += "\n## SQL Query Test Results\n\n"
    
    for approach, result in sql_results.items():
        markdown += f"### {approach}\n\n"
        markdown += f"**Success**: {'Yes' if result['success'] else 'No'}\n\n"
        
        if result['query']:
            markdown += "**Query:**\n```sql\n" + result['query'] + "\n```\n\n"
        
        if result['error']:
            markdown += "**Error:**\n```\n" + result['error'] + "\n```\n\n"
    
    return markdown

def main():
    """Main function to gather information and output Markdown."""
    try:
        markdown = generate_markdown()
        
        # Print to console
        print(markdown)
        
        # Save to file
        output_path = os.path.join(os.path.dirname(__file__), "technical_info.md")
        with open(output_path, "w") as f:
            f.write(markdown)
        
        logger.info(f"Technical information saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating technical information: {e}")

if __name__ == "__main__":
    main()