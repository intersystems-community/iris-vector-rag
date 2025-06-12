"""
Execute ObjectScript to import the iFind class
This will enable proper full-text search in HybridIFindRAG
"""

import sys
import os
sys.path.append('.')
from common.iris_connector import get_iris_connection

def import_ifind_class():
    """Import the ObjectScript class using IRIS Python"""
    conn = get_iris_connection()
    
    print("=== Importing ObjectScript Class for iFind ===\n")
    
    try:
        # Get the absolute path to the class file
        class_file = os.path.abspath('objectscript/RAG.SourceDocumentsWithIFind.cls')
        print(f"1. Class file path: {class_file}")
        
        # Check if file exists
        if not os.path.exists(class_file):
            print(f"   ❌ File not found!")
            return False
        
        print(f"   ✅ File exists")
        
        # Try to use IRIS Python to execute ObjectScript
        print("\n2. Attempting to import class via IRIS Python...")
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Try different approaches
        print("\n   Approach 1: Using CALL syntax...")
        try:
            # Try to call $system.OBJ.Load as a stored procedure
            cursor.execute(f"""
                CALL $SYSTEM.OBJ.Load('{class_file}', 'ck')
            """)
            print("   ✅ Import command executed!")
        except Exception as e:
            print(f"   ❌ CALL syntax failed: {e}")
            
            print("\n   Approach 2: Using SELECT with ObjectScript...")
            try:
                # Try using SELECT to execute ObjectScript
                cursor.execute(f"""
                    SELECT $SYSTEM.OBJ.Load('{class_file}', 'ck')
                """)
                result = cursor.fetchone()
                print(f"   ✅ Import result: {result}")
            except Exception as e2:
                print(f"   ❌ SELECT syntax failed: {e2}")
                
                print("\n   Approach 3: Using IRIS Embedded Python...")
                try:
                    # Try to access IRIS directly if available
                    import iris
                    
                    # Get IRIS native API
                    iris_native = iris.connect(
                        hostname='localhost',
                        port=1972,
                        namespace='USER',
                        username='_SYSTEM',
                        password='SYS'
                    )
                    
                    # Execute ObjectScript
                    result = iris_native.classMethodValue(
                        "%SYSTEM.OBJ",
                        "Load",
                        class_file,
                        "ck"
                    )
                    print(f"   ✅ Import via IRIS native: {result}")
                    
                except Exception as e3:
                    print(f"   ❌ IRIS native failed: {e3}")
                    print("\n   ⚠️  Cannot import via Python - manual import required!")
        
        # Check if the class now exists
        print("\n3. Checking if class was imported...")
        cursor.execute("""
            SELECT COUNT(*) 
            FROM %Dictionary.ClassDefinition 
            WHERE Name = 'RAG.SourceDocumentsWithIFind'
        """)
        exists = cursor.fetchone()[0]
        
        if exists > 0:
            print("   ✅ Class RAG.SourceDocumentsWithIFind now exists!")
            
            # Check for the index
            print("\n4. Checking for iFind index...")
            cursor.execute("""
                SELECT COUNT(*)
                FROM %Dictionary.IndexDefinition
                WHERE parent = 'RAG.SourceDocumentsWithIFind'
                AND Name = 'TextContentFTI'
            """)
            index_exists = cursor.fetchone()[0]
            
            if index_exists > 0:
                print("   ✅ TextContentFTI index exists!")
                
                # Test the index
                print("\n5. Testing %FIND search...")
                try:
                    cursor.execute("""
                        SELECT TOP 5 doc_id, title
                        FROM RAG.SourceDocumentsIFind
                        WHERE %ID %FIND search_index(TextContentFTI, 'diabetes')
                    """)
                    results = cursor.fetchall()
                    print(f"   ✅ iFind search works! Found {len(results)} results")
                    
                    for doc_id, title in results[:3]:
                        print(f"      - {doc_id}: {title[:60]}...")
                        
                except Exception as e:
                    print(f"   ❌ iFind search failed: {e}")
            else:
                print("   ❌ TextContentFTI index not found")
        else:
            print("   ❌ Class still does not exist")
            print("\n⚠️  MANUAL IMPORT REQUIRED!")
            print("\nPlease run in IRIS Terminal:")
            print(f'USER> do $system.OBJ.Load("{class_file}","ck")')
            
        cursor.close()
        conn.close()
        
        return exists > 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_manual_import_script():
    """Create a script file with ObjectScript commands"""
    
    print("\n\n=== Creating Manual Import Script ===\n")
    
    script_content = f"""
; ObjectScript commands to import iFind class
; Run these commands in IRIS Terminal

; 1. Switch to USER namespace (if needed)
zn "USER"

; 2. Import the class
do $system.OBJ.Load("{os.path.abspath('objectscript/RAG.SourceDocumentsWithIFind.cls')}","ck")

; 3. Verify the class exists
do $system.OBJ.Exists("RAG.SourceDocumentsWithIFind")

; 4. Check the index
zw ^%Dictionary.IndexDefinitionI("RAG.SourceDocumentsWithIFind","TextContentFTI")

; 5. Test iFind search
&sql(SELECT TOP 5 doc_id, title FROM RAG.SourceDocumentsIFind WHERE %ID %FIND search_index(TextContentFTI, 'diabetes'))
write SQLCODE,!
"""
    
    with open('import_ifind_class.cos', 'w') as f:
        f.write(script_content)
    
    print("Created: import_ifind_class.cos")
    print("\nTo use:")
    print("1. Open IRIS Terminal")
    print("2. Copy and paste the commands from import_ifind_class.cos")
    print("3. Or run: do ^%RI and select the file")

if __name__ == "__main__":
    # Try to import the class
    success = import_ifind_class()
    
    if not success:
        # Create manual import script
        create_manual_import_script()
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Open IRIS Terminal")
        print("2. Run the import command shown above")
        print("3. Then test HybridIFindRAG again")
        print("\nWithout this import, iFind will NOT work!")