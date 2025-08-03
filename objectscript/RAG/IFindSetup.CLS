/// Setup class for iFind functionality on RAG.SourceDocuments
Class RAG.IFindSetup
{

/// Create persistent class table for iFind functionality
ClassMethod CreateIFindTable() As %Status
{
    Try {
        Write "Setting up iFind table using persistent class...",!
        
        // Compile the SourceDocumentsWithIFind class to create table
        Set status = $System.OBJ.Compile("RAG.SourceDocumentsWithIFind.cls", "ck")
        If $$$ISERR(status) {
            Write "Error compiling class: ", $System.Status.GetErrorText(status),!
            Return status
        }
        
        Write "✅ iFind table class compiled successfully",!
        
        // Test that the table was created
        &sql(SELECT COUNT(*) INTO :count FROM RAG.SourceDocumentsWithIFind)
        If SQLCODE = 0 {
            Write "✅ Table created and accessible",!
        } Else {
            Write "Table creation verification failed: SQLCODE=", SQLCODE,!
        }
        
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
        
        // Use INSERT with explicit column mapping for compatibility
        &sql(INSERT INTO RAG.SourceDocumentsWithIFind 
             (doc_id, title, text_content, authors, keywords, embedding, created_at)
             SELECT doc_id, title, text_content, authors, keywords, embedding, created_at
             FROM RAG.SourceDocuments)
        
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

/// Test iFind search using %FIND
ClassMethod TestIFindSearch(searchText As %String) As %Status
{
    Try {
        Write !,"Searching for: ", searchText,!,!
        
        &sql(DECLARE C1 CURSOR FOR
             SELECT TOP 10 doc_id, title
             FROM RAG.SourceDocumentsWithIFind
             WHERE %FIND(text_content, :searchText) > 0)
        
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
    Write "  FROM RAG.SourceDocumentsWithIFind",!
    Write "  WHERE %FIND(text_content, ?) > 0",!
    
    Return $$$OK
}

}