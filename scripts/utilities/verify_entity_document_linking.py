import sys
sys.path.append('.')
from common.iris_connector import get_iris_connection

def verify_linking():
    print("🔍 Verifying entity-document linking...")
    iris = None
    cursor = None
    correct_links = 0
    incorrect_links = 0
    checked_entities = 0

    try:
        iris = get_iris_connection()
        cursor = iris.cursor()

        # Fetch a sample of entities with their source_doc_id
        cursor.execute("""
            SELECT TOP 10 entity_id, entity_name, source_doc_id
            FROM RAG.Entities
            WHERE source_doc_id IS NOT NULL
        """)
        sample_entities = cursor.fetchall()

        if not sample_entities:
            print("No entities found in RAG.Entities to verify.")
            return 0, 0 # entities_extracted, correct_linking_verified (as a boolean)

        print(f"Checking linking for {len(sample_entities)} sample entities...")

        for entity_id, entity_name, source_doc_id_from_entity in sample_entities:
            checked_entities += 1
            # Check if this source_doc_id exists in RAG.SourceDocuments
            cursor.execute("""
                SELECT doc_id
                FROM RAG.SourceDocuments
                WHERE doc_id = ?
            """, (source_doc_id_from_entity,))
            document_match = cursor.fetchone()

            if document_match:
                print(f"  ✅ Correct link: Entity '{entity_name}' (ID: {entity_id}) with source_doc_id '{source_doc_id_from_entity}' exists in RAG.SourceDocuments.")
                correct_links += 1
            else:
                print(f"  ❌ INCORRECT link: Entity '{entity_name}' (ID: {entity_id}) has source_doc_id '{source_doc_id_from_entity}', which was NOT FOUND in RAG.SourceDocuments.")
                incorrect_links += 1
        
        # Get total entity count from the 13 documents
        cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
        total_entities_in_db = cursor.fetchone()[0]
        
        print("\nVerification Summary:")
        print(f"Checked {checked_entities} sample entities.")
        print(f"Correct links: {correct_links}")
        print(f"Incorrect links: {incorrect_links}")
        
        if incorrect_links == 0 and checked_entities > 0:
            print("✅ All checked entities are correctly linked to source documents.")
            return total_entities_in_db, True
        elif checked_entities == 0:
            print("⚠️ No entities with source_doc_id found to check.")
            return total_entities_in_db, False
        else:
            print("❌ Some entities have incorrect links.")
            return total_entities_in_db, False

    except Exception as e:
        print(f"Error during verification: {e}")
        return 0, False
    finally:
        if cursor:
            cursor.close()
        if iris:
            iris.close()

if __name__ == "__main__":
    entities_count, linking_ok = verify_linking()
    print(f"\nTotal entities extracted from 13 documents (from DB): {entities_count}")
    print(f"Linking correct (based on sample): {linking_ok}")