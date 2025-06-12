import textwrap
try:
    from common.iris_connector import get_iris_connection, IRISConnectionError
    DB_CONNECTION_AVAILABLE = True
except ImportError:
    DB_CONNECTION_AVAILABLE = False
    print("WARNING: common.iris_connector module not found. Database operations will be skipped.")
    print("Please ensure common/iris_connector.py is present and correct.")
    class IRISConnectionError(Exception): pass

def execute_query(cursor, query, params=None):
    try:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor.fetchall()
    except Exception as e:
        print(f"Error executing query: {query}\n{e}")
        return None

def main():
    if not DB_CONNECTION_AVAILABLE:
        print("Exiting due to missing database connection utility (common.iris_connector).")
        return

    conn = None
    try:
        print("Attempting to connect to the database using common.iris_connector...")
        conn = get_iris_connection()
        cursor = conn.cursor()
        print("Successfully connected to the database.")

        print("\n--- RAG.SourceDocuments Details ---")
        total_docs = execute_query(cursor, "SELECT COUNT(*) as total_docs FROM RAG.SourceDocuments;")
        if total_docs:
            print(f"1. Total documents in RAG.SourceDocuments: {total_docs[0][0]}")

        top_20_source_docs = execute_query(cursor, "SELECT TOP 20 doc_id FROM RAG.SourceDocuments ORDER BY doc_id;")
        if top_20_source_docs:
            print("\n2. Sample of TOP 20 doc_ids from RAG.SourceDocuments (ordered):")
            for row in top_20_source_docs:
                print(f"   '{row[0]}'")
        
        highest_pmc_source_doc = execute_query(cursor, "SELECT TOP 1 doc_id FROM RAG.SourceDocuments WHERE doc_id LIKE 'PMC%' ORDER BY doc_id DESC;")
        if highest_pmc_source_doc:
            print(f"\n3. Highest PMC doc_id in RAG.SourceDocuments: '{highest_pmc_source_doc[0][0]}'")
        else:
            print("\n3. No PMC doc_ids found in RAG.SourceDocuments or table is empty.")

        print("\n--- RAG.Entities Details ---")
        total_entities = execute_query(cursor, "SELECT COUNT(*) FROM RAG.Entities;")
        if total_entities:
            print(f"4a. Total rows in RAG.Entities: {total_entities[0][0]}")

        top_20_entity_source_docs = execute_query(cursor, "SELECT DISTINCT TOP 20 source_doc_id FROM RAG.Entities WHERE source_doc_id LIKE 'PMC%' ORDER BY source_doc_id;")
        if top_20_entity_source_docs:
            print("4b. Sample of TOP 20 distinct source_doc_ids from RAG.Entities (PMC only, ordered):")
            for row in top_20_entity_source_docs:
                print(f"   '{row[0]}'")
        
        min_pmc_entity_doc = execute_query(cursor, "SELECT TOP 1 source_doc_id FROM RAG.Entities WHERE source_doc_id LIKE 'PMC%' ORDER BY source_doc_id ASC;")
        max_pmc_entity_doc = execute_query(cursor, "SELECT TOP 1 source_doc_id FROM RAG.Entities WHERE source_doc_id LIKE 'PMC%' ORDER BY source_doc_id DESC;")
        
        min_val_entities = "N/A"
        if min_pmc_entity_doc and min_pmc_entity_doc[0]:
            min_val_entities = min_pmc_entity_doc[0][0]
        
        max_val_entities = "N/A"
        if max_pmc_entity_doc and max_pmc_entity_doc[0]:
            max_val_entities = max_pmc_entity_doc[0][0]

        print(f"\n4c. Range of PMC source_doc_ids in RAG.Entities:")
        print(f"    Lowest: '{min_val_entities}'")
        print(f"    Highest: '{max_val_entities}'")

        print("\n--- RAG.Entities_V2 Details ---")
        total_entities_v2 = execute_query(cursor, "SELECT COUNT(*) FROM RAG.Entities_V2;")
        if total_entities_v2:
            print(f"5a. Total rows in RAG.Entities_V2: {total_entities_v2[0][0]}")
        else:
            print("5a. RAG.Entities_V2 does not exist or is empty.")


        if total_entities_v2 and total_entities_v2[0][0] > 0 :
            top_20_entity_v2_source_docs = execute_query(cursor, "SELECT DISTINCT TOP 20 source_doc_id FROM RAG.Entities_V2 WHERE source_doc_id LIKE 'PMC%' ORDER BY source_doc_id;")
            if top_20_entity_v2_source_docs:
                print("5b. Sample of TOP 20 distinct source_doc_ids from RAG.Entities_V2 (PMC only, ordered):")
                for row in top_20_entity_v2_source_docs:
                    print(f"   '{row[0]}'")
            
            min_pmc_entity_v2_doc = execute_query(cursor, "SELECT TOP 1 source_doc_id FROM RAG.Entities_V2 WHERE source_doc_id LIKE 'PMC%' ORDER BY source_doc_id ASC;")
            max_pmc_entity_v2_doc = execute_query(cursor, "SELECT TOP 1 source_doc_id FROM RAG.Entities_V2 WHERE source_doc_id LIKE 'PMC%' ORDER BY source_doc_id DESC;")
            
            min_val_entities_v2 = "N/A"
            if min_pmc_entity_v2_doc and min_pmc_entity_v2_doc[0]:
                min_val_entities_v2 = min_pmc_entity_v2_doc[0][0]
            
            max_val_entities_v2 = "N/A"
            if max_pmc_entity_v2_doc and max_pmc_entity_v2_doc[0]:
                max_val_entities_v2 = max_pmc_entity_v2_doc[0][0]

            print(f"\n5c. Range of PMC source_doc_ids in RAG.Entities_V2:")
            print(f"    Lowest: '{min_val_entities_v2}'")
            print(f"    Highest: '{max_val_entities_v2}'")

    except IRISConnectionError as e_conn:
        print(f"\nDatabase connection error: {e_conn}")
        print("Please ensure your IRIS connection environment variables are correctly set.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    main()