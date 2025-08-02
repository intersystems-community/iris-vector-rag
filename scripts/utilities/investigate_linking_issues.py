import argparse
import textwrap

# Attempt to import the project's database connection utility
try:
    from common.iris_connector import get_iris_connection, IRISConnectionError
    DB_CONNECTION_AVAILABLE = True
except ImportError:
    DB_CONNECTION_AVAILABLE = False
    print("WARNING: common.iris_connector module not found. Database operations will be skipped.")
    print("Please ensure common/iris_connector.py is present and correct.")
    # Define a placeholder for IRISConnectionError if the import fails
    class IRISConnectionError(Exception): pass


def get_column_schema_info(cursor, table_name, column_name, schema_name='RAG'):
    """
    Retrieves schema information for a specific column.
    """
    query = f"""
    SELECT
        COLUMN_NAME,
        DATA_TYPE,
        CHARACTER_MAXIMUM_LENGTH,
        IS_NULLABLE,
        COLLATION_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? AND COLUMN_NAME = ?
    """
    try:
        cursor.execute(query, (schema_name, table_name, column_name))
        return cursor.fetchone()
    except Exception as e:
        print(f"Error fetching schema for {schema_name}.{table_name}.{column_name}: {e}")
        # Fallback for older IRIS versions or different catalog names
        query_fallback = f"""
        SELECT
            Name,
            Type,
            MAXLEN,
            AllowNulls
        FROM %Dictionary.CompiledProperty
        WHERE parent = ? AND Name = ?
        """
        # Class name for RAG.SourceDocuments would be RAG.SourceDocuments (if mapped directly)
        # This might need adjustment based on actual class definition if not a direct SQL table.
        class_name = f"{schema_name}.{table_name}"
        try:
            cursor.execute(query_fallback, (class_name, column_name))
            prop_info = cursor.fetchone()
            if prop_info:
                # Map %Dictionary types to SQL-like types (simplified)
                # This is a very basic mapping and might need refinement
                type_mapping = {
                    1: "VARCHAR",  # %String
                    2: "INTEGER",  # %Integer
                    3: "DATE",     # %Date
                    4: "NUMERIC",  # %Numeric
                    # Add more mappings as needed
                }
                return (
                    prop_info[0], # Name
                    type_mapping.get(prop_info[1], f"UnknownType({prop_info[1]})"), # Type
                    prop_info[2], # MAXLEN
                    "YES" if prop_info[3] else "NO", # AllowNulls
                    "N/A" # Collation not directly available here
                )
            return None
        except Exception as e_fallback:
            print(f"Fallback schema query failed for {class_name}.{column_name}: {e_fallback}")
            return None


def print_schema_comparison(cursor):
    """
    Prints a comparison of the schema for doc_id and source_doc_id.
    """
    print("\n--- 1. Database Schema Check ---")

    print("\nFetching schema for RAG.SourceDocuments.doc_id...")
    sd_doc_id_schema = get_column_schema_info(cursor, 'SourceDocuments', 'doc_id')
    if sd_doc_id_schema:
        print(f"  RAG.SourceDocuments.doc_id:")
        print(f"    Column Name: {sd_doc_id_schema[0]}")
        print(f"    Data Type: {sd_doc_id_schema[1]}")
        print(f"    Max Length: {sd_doc_id_schema[2]}")
        print(f"    Is Nullable: {sd_doc_id_schema[3]}")
        print(f"    Collation: {sd_doc_id_schema[4] if len(sd_doc_id_schema) > 4 else 'N/A (check %SQLSTRINGCOLLATION)'}")
    else:
        print("  Could not retrieve schema for RAG.SourceDocuments.doc_id.")

    print("\nFetching schema for RAG.Entities.source_doc_id...")
    e_source_doc_id_schema = get_column_schema_info(cursor, 'Entities', 'source_doc_id')
    if e_source_doc_id_schema:
        print(f"  RAG.Entities.source_doc_id:")
        print(f"    Column Name: {e_source_doc_id_schema[0]}")
        print(f"    Data Type: {e_source_doc_id_schema[1]}")
        print(f"    Max Length: {e_source_doc_id_schema[2]}")
        print(f"    Is Nullable: {e_source_doc_id_schema[3]}")
        print(f"    Collation: {e_source_doc_id_schema[4] if len(e_source_doc_id_schema) > 4 else 'N/A (check %SQLSTRINGCOLLATION)'}")
    else:
        print("  Could not retrieve schema for RAG.Entities.source_doc_id.")

    if sd_doc_id_schema and e_source_doc_id_schema:
        print("\nSchema Comparison:")
        if sd_doc_id_schema[1] != e_source_doc_id_schema[1]:
            print(f"  WARNING: Data type mismatch! SourceDocuments: {sd_doc_id_schema[1]}, Entities: {e_source_doc_id_schema[1]}")
        else:
            print("  Data types appear to match.")

        if sd_doc_id_schema[2] != e_source_doc_id_schema[2]:
            print(f"  WARNING: Max length mismatch! SourceDocuments: {sd_doc_id_schema[2]}, Entities: {e_source_doc_id_schema[2]}")
        else:
            print("  Max lengths appear to match.")
        
        # Note: Collation comparison can be tricky. %SQLSTRINGCOLLATION affects default collation.
        # Explicit collation on columns is less common in IRIS but possible.
        # The INFORMATION_SCHEMA.COLUMNS.COLLATION_NAME should show it if explicitly set.
        # If using %String, it defaults to SQLUPPER which is case-insensitive for comparisons.
        # If types are different (e.g. VARCHAR vs %String mapped to something else), behavior might differ.
        print("  Collation/Case Sensitivity: IRIS %String types are typically case-insensitive for SQL comparisons (SQLUPPER).")
        print("  If explicit collations are set (e.g. EXACT), behavior will differ. Check 'Collation' field above.")
        if len(sd_doc_id_schema) > 4 and len(e_source_doc_id_schema) > 4 and sd_doc_id_schema[4] != e_source_doc_id_schema[4]:
             print(f"  WARNING: Collation mismatch! SourceDocuments: {sd_doc_id_schema[4]}, Entities: {e_source_doc_id_schema[4]}")
        elif len(sd_doc_id_schema) > 4 and len(e_source_doc_id_schema) > 4:
            print("  Explicit collations (if any) appear to match.")
        else:
            print("  Collation information might be partial; further investigation of %SQLSTRINGCOLLATION may be needed if issues persist.")


def sample_data(cursor):
    """
    Samples doc_id and source_doc_id values.
    """
    print("\n--- 2. Data Sampling and Mismatch Identification ---")
    sample_size = 10

    print(f"\nSampling TOP {sample_size} RAG.SourceDocuments.doc_id values...")
    try:
        cursor.execute(f"SELECT TOP {sample_size} doc_id FROM RAG.SourceDocuments WHERE doc_id IS NOT NULL ORDER BY doc_id")
        sd_samples = cursor.fetchall()
        if sd_samples:
            print("  Sample doc_ids from RAG.SourceDocuments:")
            for row in sd_samples:
                print(f"    '{row[0]}'")
        else:
            print("  No doc_id samples found in RAG.SourceDocuments.")
    except Exception as e:
        print(f"  Error sampling from RAG.SourceDocuments: {e}")

    print(f"\nSampling TOP {sample_size} RAG.Entities.source_doc_id values...")
    try:
        cursor.execute(f"SELECT TOP {sample_size} source_doc_id FROM RAG.Entities WHERE source_doc_id IS NOT NULL ORDER BY source_doc_id")
        e_samples = cursor.fetchall()
        if e_samples:
            print("  Sample source_doc_ids from RAG.Entities:")
            for row in e_samples:
                print(f"    '{row[0]}'")
        else:
            print("  No source_doc_id samples found in RAG.Entities.")
    except Exception as e:
        print(f"  Error sampling from RAG.Entities: {e}")

    print("\nIdentifying orphaned entities (TOP 10 by count)...")
    # Using TOP N for IRIS SQL
    query_orphaned = """
    SELECT TOP 10 e.source_doc_id, COUNT(*) as num_orphaned
    FROM RAG.Entities e
    LEFT JOIN RAG.SourceDocuments sd ON e.source_doc_id = sd.doc_id
    WHERE sd.doc_id IS NULL AND e.source_doc_id IS NOT NULL
    GROUP BY e.source_doc_id
    ORDER BY num_orphaned DESC
    """
    try:
        cursor.execute(query_orphaned)
        orphaned_entities = cursor.fetchall()
        if orphaned_entities:
            print("  Orphaned source_doc_id patterns (source_doc_id, count):")
            for row in orphaned_entities:
                print(f"    '{row[0]}' (Count: {row[1]})")
        else:
            print("  No orphaned entities found (or RAG.Entities is empty / all are linked).")
    except Exception as e:
        print(f"  Error identifying orphaned entities: {e}")


def check_doc_id_integrity(cursor):
    """
    Checks for NULL or duplicate doc_id values in RAG.SourceDocuments.
    """
    print("\n--- 3. RAG.SourceDocuments.doc_id Integrity Check ---")

    print("\nChecking for NULL doc_id values in RAG.SourceDocuments...")
    try:
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE doc_id IS NULL")
        null_count = cursor.fetchone()[0]
        print(f"  Number of NULL doc_id values: {null_count}")
        if null_count > 0:
            print("  WARNING: NULL doc_id values found!")
    except Exception as e:
        print(f"  Error checking for NULL doc_ids: {e}")

    print("\nChecking for duplicate doc_id values in RAG.SourceDocuments...")
    # Using TOP N for IRIS SQL
    query_duplicates = """
    SELECT TOP 10 doc_id, COUNT(*) as count_num
    FROM RAG.SourceDocuments
    WHERE doc_id IS NOT NULL
    GROUP BY doc_id
    HAVING COUNT(*) > 1
    ORDER BY count_num DESC
    """
    try:
        cursor.execute(query_duplicates)
        duplicates = cursor.fetchall()
        if duplicates:
            print("  Duplicate doc_id values found (doc_id, count):")
            for row in duplicates:
                print(f"    '{row[0]}' (Count: {row[1]})")
            print("  WARNING: Duplicate doc_id values exist!")
        else:
            print("  No duplicate doc_id values found.")
    except Exception as e:
        print(f"  Error checking for duplicate doc_ids: {e}")

def main():
    if not DB_CONNECTION_AVAILABLE:
        print("Exiting due to missing database connection utility (common.iris_connector).")
        return

    parser = argparse.ArgumentParser(
        description="Investigate entity-document linking issues in the RAG database.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # common.iris_connector uses environment variables (IRIS_HOST, IRIS_PORT, etc.)
    # or a config dictionary if passed to get_iris_connection.
    # For this script, we'll rely on environment variables by calling get_iris_connection() without args.

    args = parser.parse_args()

    conn = None
    try:
        print("Attempting to connect to the database using common.iris_connector...")
        # get_iris_connection can take a config dict, but here we rely on its env var handling
        conn = get_iris_connection()
        cursor = conn.cursor()
        print("Successfully connected to the database.")

        print_schema_comparison(cursor)
        sample_data(cursor)
        check_doc_id_integrity(cursor)

        print("\n--- 4. Preliminary Findings & Next Steps ---")
        print(textwrap.dedent("""
        Based on the output above, consider the following:

        Nature of ID Mismatch:
        - Data Type Mismatch: Do `doc_id` and `source_doc_id` have different SQL data types?
        - Length Mismatch: Is one column shorter than the values stored in the other?
        - Case Sensitivity/Collation: IRIS default (%String/VARCHAR with SQLUPPER) is case-insensitive.
          If `EXACT` collation or different types are used, this could be an issue.
          Examine the 'Collation' fields and sample data for case differences.
        - Formatting Differences:
          - Leading/trailing spaces: Check sampled values carefully.
          - Prefixes/Suffixes: Are there patterns like 'PMC' prefix in one but not the other?
          - Special characters or encoding issues.
        - NULLs or Duplicates:
          - NULL `doc_id` in `SourceDocuments` means those documents can't be linked.
          - Duplicate `doc_id` in `SourceDocuments` can cause ambiguous links.
          - NULL `source_doc_id` in `Entities` means those entities are inherently unlinked.

        Is it a schema issue, data formatting issue, or something else?
        - Schema Issue: Indicated by type/length/collation mismatches reported in Section 1.
        - Data Formatting Issue: Indicated by differences in actual sampled values (Section 2)
          or orphaned entities whose IDs look *almost* like valid `doc_id`s.
        - Data Integrity Issue: Indicated by NULLs/duplicates in `SourceDocuments` (Section 3).

        Concrete Solution Ideas (depends on findings):
        - Schema Change: `ALTER TABLE` to align types, lengths, or collations.
          (Requires careful planning, especially with existing data).
        - Data Cleaning: `UPDATE` statements to trim spaces, standardize case, add/remove prefixes.
          (e.g., `UPDATE RAG.Entities SET source_doc_id = UPPER(source_doc_id)` or
           `UPDATE RAG.Entities SET source_doc_id = LTRIM(RTRIM(source_doc_id))`).
        - Fix Data Ingestion: Modify the source of `RAG.Entities.source_doc_id` or
          `RAG.SourceDocuments.doc_id` to ensure they are generated/stored consistently.
        - Handle NULLs/Duplicates: Delete or correct records with NULL/duplicate primary keys.

        This script provides diagnostic information. The actual solution will require careful
        analysis of this output.
        """))

    except IRISConnectionError as e_conn:
        print(f"\nDatabase connection error: {e_conn}")
        print("Please ensure your IRIS connection environment variables (e.g., IRIS_HOST, IRIS_PORT, IRIS_NAMESPACE, IRIS_USERNAME, IRIS_PASSWORD) are correctly set.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    main()