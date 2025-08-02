import argparse
import logging
import re
import sys
import os
from typing import List, Dict, Tuple, Optional, Any

# Add project root to sys.path to allow imports from common
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from common.iris_connector import get_iris_connection, IRISConnectionError
    import jaydebeapi
except ImportError as e:
    print(f"Error importing common.iris_connector or jaydebeapi: {e}. Ensure the common module is in PYTHONPATH and jaydebeapi is installed.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Regex to capture PMC followed by numbers, potentially with prefixes/suffixes
PMC_PATTERN = re.compile(r'(?:[^a-z0-9]|^)(PMC\d+)(?:[^a-z0-9]|$)', re.IGNORECASE)
# Regex to capture standalone numbers that could be PMC IDs
NUMERIC_PATTERN = re.compile(r'^\d+$')

def standardize_doc_id(original_id: str) -> Tuple[Optional[str], str]:
    """
    Standardizes a document ID to the 'PMC' + numbers format.

    Args:
        original_id: The original document ID string.

    Returns:
        A tuple containing the standardized ID (or None if not standardizable)
        and a status message ('standardized', 'numeric_to_pmc', 'prefixed_pmc', 'skipped_unclear', 'already_standard').
    """
    if not original_id or not isinstance(original_id, str):
        return None, "skipped_invalid_input"

    stripped_id = original_id.strip()

    # Check if already in standard PMC format (e.g., "PMC12345")
    if re.fullmatch(r'PMC\d+', stripped_id):
        return stripped_id, "already_standard"

    # Case 1: Extract PMC ID if embedded (e.g., 'good_PMC1', 'doc_PMC12345')
    pmc_match = PMC_PATTERN.search(stripped_id)
    if pmc_match:
        extracted_pmc_id = pmc_match.group(1).upper()
        # Ensure it's exactly PMC + numbers
        if re.fullmatch(r'PMC\d+', extracted_pmc_id):
             return extracted_pmc_id, "prefixed_pmc"

    # Case 2: Convert numeric IDs (e.g., '12345' -> 'PMC12345')
    if NUMERIC_PATTERN.fullmatch(stripped_id):
        return f"PMC{stripped_id}", "numeric_to_pmc"

    # Case 3: Handle 'DOCA' or other non-standardizable formats
    # For now, we skip these as per instructions.
    # A more sophisticated mapping could be added here.
    if "DOCA" in stripped_id.upper(): # Simple check for DOCA
        return None, "skipped_unclear_DOCA"

    # If no specific rule matched, but PMC was found, try to use it.
    # This handles cases like "PMC12345_extra_stuff" -> "PMC12345"
    # This is a bit more aggressive than the initial PMC_PATTERN check alone.
    if pmc_match: # Re-check pmc_match from earlier
        extracted_pmc_id = pmc_match.group(1).upper()
        if re.fullmatch(r'PMC\d+', extracted_pmc_id):
            return extracted_pmc_id, "extracted_pmc_suffix"


    logger.debug(f"Could not standardize '{original_id}'. It will be skipped or handled by a default rule if any.")
    return None, "skipped_unclear"


def get_db_connection(args: argparse.Namespace) -> Optional[jaydebeapi.Connection]:
    """Establishes and returns a database connection."""
    try:
        config = {
            "db_host": args.host,
            "db_port": args.port,
            "db_namespace": args.namespace,
            "db_user": args.user,
            "db_password": args.password,
        }
        logger.info(f"Attempting to connect to IRIS with config: {{'db_host': '{args.host}', 'db_port': {args.port}, 'db_namespace': '{args.namespace}', 'db_user': '{args.user}'}}")
        conn = get_iris_connection(config=config)
        logger.info("Successfully connected to IRIS.")
        return conn
    except IRISConnectionError as e:
        logger.error(f"Database connection failed: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during DB connection: {e}")
        return None

def fetch_unique_doc_ids(conn: jaydebeapi.Connection) -> List[str]:
    """Fetches all unique doc_id values from RAG.SourceDocuments."""
    unique_ids: List[str] = []
    try:
        with conn.cursor() as cursor:
            # Using TOP for IRIS SQL compatibility, assuming we want all distinct IDs
            # If the number of distinct IDs is very large, consider pagination or sampling
            sql = "SELECT DISTINCT doc_id FROM RAG.SourceDocuments WHERE doc_id IS NOT NULL"
            logger.info(f"Executing query: {sql}")
            cursor.execute(sql)
            rows = cursor.fetchall()
            unique_ids = [row[0] for row in rows if row[0]] # Ensure not None
            logger.info(f"Fetched {len(unique_ids)} unique doc_ids.")
    except jaydebeapi.Error as e:
        logger.error(f"Error fetching unique doc_ids: {e}")
    return unique_ids

def analyze_and_preview_changes(doc_ids: List[str]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Analyzes doc_ids, standardizes them, and prepares a preview.
    Returns a list of proposed changes and a summary of transformation types.
    """
    proposed_changes: List[Dict[str, str]] = []
    transformation_summary: Dict[str, int] = {}

    for original_id in doc_ids:
        standardized_id, status = standardize_doc_id(original_id)
        transformation_summary[status] = transformation_summary.get(status, 0) + 1
        if standardized_id and standardized_id != original_id:
            proposed_changes.append({
                "original_id": original_id,
                "new_id": standardized_id,
                "status": status
            })
        elif not standardized_id:
             # Log IDs that couldn't be standardized, even if no change is proposed
            logger.debug(f"ID '{original_id}' resulted in status '{status}' and will not be changed.")


    return proposed_changes, transformation_summary

def apply_doc_id_changes(conn: jaydebeapi.Connection, changes: List[Dict[str, str]], dry_run: bool = False) -> List[Dict[str, Any]]:
    """
    Applies the doc_id changes to the RAG.SourceDocuments table.
    Logs each transformation.
    """
    update_log: List[Dict[str, Any]] = []
    updated_count = 0
    failed_count = 0

    if dry_run:
        logger.info("[DRY RUN] No changes will be applied to the database.")

    try:
        with conn.cursor() as cursor:
            for change in changes:
                original_id = change["original_id"]
                new_id = change["new_id"]
                status = change["status"]
                log_entry = {
                    "original_id": original_id,
                    "new_id": new_id,
                    "status": status,
                    "applied": False,
                    "error": None
                }
                if not dry_run:
                    try:
                        # IMPORTANT: Use placeholders to prevent SQL injection
                        # Update RAG.SourceDocuments
                        sql_source_docs = "UPDATE RAG.SourceDocuments SET doc_id = ? WHERE doc_id = ?"
                        logger.debug(f"Executing: {sql_source_docs} on RAG.SourceDocuments with params ('{new_id}', '{original_id}')")
                        cursor.execute(sql_source_docs, (new_id, original_id))

                        # Update RAG.Entities
                        sql_entities = "UPDATE RAG.Entities SET source_doc_id = ? WHERE source_doc_id = ?"
                        logger.debug(f"Executing: {sql_entities} on RAG.Entities with params ('{new_id}', '{original_id}')")
                        cursor.execute(sql_entities, (new_id, original_id))
                        
                        # conn.commit() # Commit per change or at the end? For safety, commit at end or in batches.
                        # For now, let's assume autocommit is off or commit will be handled by main.
                        log_entry["applied"] = True
                        updated_count +=1
                    except jaydebeapi.Error as e:
                        logger.error(f"Error updating doc_id '{original_id}' to '{new_id}': {e}")
                        log_entry["error"] = str(e)
                        failed_count += 1
                        # conn.rollback() # Rollback this specific error if transactions are managed per operation
                else: # dry_run
                    log_entry["applied"] = "dry_run_skipped"

                update_log.append(log_entry)

        if not dry_run and failed_count == 0:
            logger.info("Committing all successful changes.")
            conn.commit()
        elif not dry_run and failed_count > 0:
            logger.warning(f"Rolling back changes due to {failed_count} errors during update.")
            conn.rollback()

    except jaydebeapi.Error as e:
        logger.error(f"A database error occurred during the update process: {e}")
        if not dry_run:
            conn.rollback() # Rollback any pending changes if the overall process fails
        # Add a general error log entry if needed
        update_log.append({
            "original_id": "GENERAL_ERROR",
            "new_id": None,
            "status": "transaction_error",
            "applied": False,
            "error": str(e)
        })


    logger.info(f"Update process complete. Updated: {updated_count}, Failed: {failed_count}, Dry run: {dry_run}")
    return update_log


def run_post_cleanup_diagnostics(conn: jaydebeapi.Connection):
    """Runs diagnostic queries after cleanup."""
    if not conn:
        logger.error("No database connection available for post-cleanup diagnostics.")
        return

    try:
        with conn.cursor() as cursor:
            logger.info("\n--- Verifying Entity-Document Linking (Post-Cleanup) ---")
            
            # Check orphaned entities
            query_orphaned = """
            SELECT TOP 10 e.source_doc_id, COUNT(*) as num_orphaned
            FROM RAG.Entities e
            LEFT JOIN RAG.SourceDocuments sd ON e.source_doc_id = sd.doc_id
            WHERE sd.doc_id IS NULL AND e.source_doc_id IS NOT NULL
            GROUP BY e.source_doc_id
            ORDER BY num_orphaned DESC
            """
            logger.info(f"Executing orphaned entities query: {query_orphaned}")
            cursor.execute(query_orphaned)
            orphaned_entities = cursor.fetchall()
            if orphaned_entities:
                logger.info("Orphaned source_doc_id patterns (source_doc_id, count):")
                for row in orphaned_entities:
                    logger.info(f"  '{row[0]}' (Count: {row[1]})")
            else:
                logger.info("No orphaned entities found post-cleanup (or RAG.Entities is empty / all are linked). This is good!")

            # Count total entities and linked entities
            query_total_entities = "SELECT COUNT(*) FROM RAG.Entities WHERE source_doc_id IS NOT NULL"
            cursor.execute(query_total_entities)
            total_entities = cursor.fetchone()
            if total_entities:
                logger.info(f"Total entities with non-NULL source_doc_id: {total_entities[0]}")

            query_linked_entities = """
            SELECT COUNT(DISTINCT e.ID)
            FROM RAG.Entities e
            JOIN RAG.SourceDocuments sd ON e.source_doc_id = sd.doc_id
            WHERE e.source_doc_id IS NOT NULL
            """
            # Assuming RAG.Entities has a primary key named ID for distinct count.
            # If not, COUNT(*) on the JOIN might be sufficient if one entity row maps to one doc.
            # Or COUNT(e.source_doc_id) if we are interested in how many entity records link.
            logger.info(f"Executing linked entities query: {query_linked_entities}")
            cursor.execute(query_linked_entities)
            linked_entities = cursor.fetchone()
            if linked_entities:
                logger.info(f"Number of entities now linked to a document: {linked_entities[0]}")
                if total_entities and total_entities[0] > 0:
                    percentage_linked = (linked_entities[0] / total_entities[0]) * 100
                    logger.info(f"Percentage of entities linked: {percentage_linked:.2f}%")


            logger.info("\n--- RAG.SourceDocuments.doc_id Integrity Check (Post-Cleanup) ---")
            logger.info("Checking for NULL doc_id values in RAG.SourceDocuments...")
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE doc_id IS NULL")
            null_count = cursor.fetchone()[0]
            logger.info(f"  Number of NULL doc_id values: {null_count}")
            if null_count > 0:
                logger.warning("  WARNING: NULL doc_id values still found post-cleanup!")

            logger.info("Checking for duplicate doc_id values in RAG.SourceDocuments...")
            query_duplicates = """
            SELECT TOP 10 doc_id, COUNT(*) as count_num
            FROM RAG.SourceDocuments
            WHERE doc_id IS NOT NULL
            GROUP BY doc_id
            HAVING COUNT(*) > 1
            ORDER BY count_num DESC
            """
            cursor.execute(query_duplicates)
            duplicates = cursor.fetchall()
            if duplicates:
                logger.warning("  Duplicate doc_id values found post-cleanup (doc_id, count):")
                for row in duplicates:
                    logger.warning(f"    '{row[0]}' (Count: {row[1]})")
            else:
                logger.info("  No duplicate doc_id values found post-cleanup.")

    except jaydebeapi.Error as e:
        logger.error(f"An error occurred during post-cleanup diagnostics: {e}")
    except Exception as e_gen:
        logger.error(f"An unexpected error during post-cleanup diagnostics: {e_gen}")


def main():
    parser = argparse.ArgumentParser(description="Clean up doc_id inconsistencies in RAG.SourceDocuments.")
    parser.add_argument("--host", default=os.environ.get("IRIS_HOST", "localhost"), help="IRIS host")
    parser.add_argument("--port", type=int, default=int(os.environ.get("IRIS_PORT", "1972")), help="IRIS port")
    parser.add_argument("--namespace", default=os.environ.get("IRIS_NAMESPACE", "USER"), help="IRIS namespace")
    parser.add_argument("--user", default=os.environ.get("IRIS_USERNAME", "SuperUser"), help="IRIS username")
    parser.add_argument("--password", default=os.environ.get("IRIS_PASSWORD", "SYS"), help="IRIS password")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without applying to DB.")
    parser.add_argument("--yes", action="store_true", help="Automatically confirm and apply changes without prompting.")
    parser.add_argument("--log-file", default="doc_id_cleanup_log.csv", help="File to log transformations.")

    args = parser.parse_args()

    conn = get_db_connection(args)
    if not conn:
        sys.exit(1)

    try:
        logger.info("Fetching unique document IDs...")
        unique_ids = fetch_unique_doc_ids(conn)
        if not unique_ids:
            logger.info("No document IDs found or able to be fetched. Exiting.")
            sys.exit(0)

        logger.info("Analyzing document IDs and preparing preview of changes...")
        proposed_changes, transformation_summary = analyze_and_preview_changes(unique_ids)

        logger.info("\n--- Transformation Summary ---")
        for status, count in transformation_summary.items():
            logger.info(f"{status}: {count}")

        if not proposed_changes:
            logger.info("\nNo changes are proposed based on the current standardization rules. Exiting.")
            logger.info("Running diagnostics even though no changes were proposed for RAG.SourceDocuments...")
            run_post_cleanup_diagnostics(conn)
            sys.exit(0)

        logger.info(f"\n--- Proposed Changes (Preview - {len(proposed_changes)} items) ---")
        for i, change in enumerate(proposed_changes):
            if i < 20: # Preview first 20 changes
                logger.info(f"Original: '{change['original_id']}' -> New: '{change['new_id']}' (Status: {change['status']})")
            elif i == 20:
                logger.info("... (and more, logging first 20)")


        if args.dry_run:
            logger.info("\n[DRY RUN MODE] No changes will be applied to the database.")
            # Simulate logging for dry run
            dry_run_log = [{"original_id": c["original_id"], "new_id": c["new_id"], "status": c["status"], "applied": "dry_run_skipped", "error": None} for c in proposed_changes]
            # Here you could write dry_run_log to args.log_file if desired
            logger.info(f"Dry run complete. {len(dry_run_log)} changes would be logged to {args.log_file}.")
            sys.exit(0)

        if not args.yes:
            confirm = input(f"\nProceed with applying these {len(proposed_changes)} changes to the database? (yes/no): ")
            if confirm.lower() != 'yes':
                logger.info("Changes aborted by user.")
                sys.exit(0)

        logger.info("Applying changes to the database...")
        update_log = apply_doc_id_changes(conn, proposed_changes, dry_run=False) # dry_run is False here

        # Write log to CSV
        try:
            import csv
            with open(args.log_file, 'w', newline='') as f:
                if update_log:
                    writer = csv.DictWriter(f, fieldnames=update_log[0].keys())
                    writer.writeheader()
                    writer.writerows(update_log)
            logger.info(f"Transformation log written to {args.log_file}")
        except Exception as e_csv:
            logger.error(f"Failed to write log to CSV file {args.log_file}: {e_csv}")


        logger.info("\n--- Running Post-Cleanup Diagnostics ---")
        run_post_cleanup_diagnostics(conn)

        logger.info("\nCleanup script finished. Next steps:")
        logger.info("1. Review the diagnostic output above to verify linking improvements.")
        logger.info("2. Test the GraphRAG pipeline again.")

    finally:
        if conn:
            try:
                conn.close()
                logger.info("Database connection closed.")
            except jaydebeapi.Error as e:
                logger.error(f"Error closing database connection: {e}")

if __name__ == "__main__":
    main()
