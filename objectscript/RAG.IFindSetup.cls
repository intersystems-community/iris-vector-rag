/// Setup class for iFind functionality on RAG.SourceDocuments
Class RAG.IFindSetup
{

/// Create a new table with iFind support since we can't alter existing table
ClassMethod CreateIFindTable() As %Status
{
    Try {
        Write "Creating new table with iFind support...",!
        
        // Create a new table that mirrors SourceDocuments but with iFind
        &sql(CREATE TABLE RAG.SourceDocumentsIFind (
            doc_id VARCHAR(200) PRIMARY KEY,
            title VARCHAR(500),
            text_content LONGVARCHAR,
            authors LONGVARCHAR,
            keywords LONGVARCHAR,
            embedding VARCHAR(32000),
            created_at TIMESTAMP
        ))
        
        If SQLCODE '= 0 {
            Write "Error creating table: SQLCODE=", SQLCODE, " MSG=", %msg,!
            Return $$$ERROR($$$GeneralError, "Failed to create table")
        }
        
        Write "Table created successfully",!
        
        // Now add the iFind index using ALTER TABLE
        Write "Adding iFind index...",!
        &sql(ALTER TABLE RAG.SourceDocumentsIFind ADD FULLTEXT INDEX idx_ifind (text_content))
        
        If SQLCODE '= 0 {
            Write "Error creating iFind index: SQLCODE=", SQLCODE, " MSG=", %msg,!
            // Try alternative syntax
            Write "Trying alternative syntax...",!
            &sql(CREATE FULLTEXT INDEX idx_ifind ON RAG.SourceDocumentsIFind (text_content))
            
            If SQLCODE '= 0 {
                Write "Still failed: SQLCODE=", SQLCODE, " MSG=", %msg,!
                Return $$$ERROR($$$GeneralError, "Failed to create iFind index")
            }
        }
        
        Write "✅ iFind index created successfully!",!
        Return $$$OK
        
    } Catch ex {
        Write "Exception: ", ex.DisplayString(),!
        Return ex.AsStatus()
    }
}

/// Copy data from SourceDocuments to SourceDocumentsIFind
ClassMethod CopyDataToIFindTable() As %Status
{
    Try {
        Write "Copying data to iFind table...",!
        
        &sql(INSERT INTO RAG.SourceDocumentsIFind 
             SELECT * FROM RAG.SourceDocuments)
        
        If SQLCODE = 0 {
            Write "✅ Copied ", %ROWCOUNT, " documents",!
            Return $$$OK
        } Else {
            Write "Error copying data: SQLCODE=", SQLCODE, " MSG=", %msg,!
            Return $$$ERROR($$$GeneralError, "Failed to copy data")
        }
        
    } Catch ex {
        Write "Exception: ", ex.DisplayString(),!
        Return ex.AsStatus()
    }
}

/// Test iFind search using %CONTAINS
ClassMethod TestIFindSearch(searchText As %String) As %Status
{
    Try {
        Write !,"Searching for: ", searchText,!,!
        
        &sql(DECLARE C1 CURSOR FOR
             SELECT TOP 10 doc_id, title
             FROM RAG.SourceDocumentsIFind
             WHERE %CONTAINS(text_content, :searchText))
        
        &sql(OPEN C1)
        
        Set count = 0
        For {
            &sql(FETCH C1 INTO :docId, :title)
            Quit:SQLCODE'=0
            
            Set count = count + 1
            Write count, ". ", docId, " - ", title,!
        }
        
        &sql(CLOSE C1)
        
        If count = 0 {
            Write "No results found",!
        } Else {
            Write !,"Found ", count, " documents",!
        }
        
        Return $$$OK
        
    } Catch ex {
        Write "Error: ", ex.DisplayString(),!
        Return ex.AsStatus()
    }
}

/// Main setup method
ClassMethod Setup() As %Status
{
    Write "=== Setting up iFind for RAG ===",!,!
    
    // Step 1: Create new table with iFind
    Set sc = ..CreateIFindTable()
    If $$$ISERR(sc) Return sc
    
    // Step 2: Copy data
    Set sc = ..CopyDataToIFindTable()
    If $$$ISERR(sc) Return sc
    
    // Step 3: Test
    Write !,"Testing iFind search...",!
    Set sc = ..TestIFindSearch("diabetes")
    
    Write !,"✅ Setup complete!",!
    Write "Update hybrid_ifind_rag/pipeline.py to use:",!
    Write "  FROM RAG.SourceDocumentsIFind",!
    Write "  WHERE %CONTAINS(text_content, ?)",!
    
    Return $$$OK
}

}