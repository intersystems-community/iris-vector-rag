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
        
        // Check if source table exists first
        &sql(SELECT COUNT(*) INTO :sourceCount FROM RAG.SourceDocuments)
        If SQLCODE '= 0 {
            Write "Source table RAG.SourceDocuments not found, skipping copy",!
            Return $$$OK
        }
        
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


/// Main setup method
ClassMethod Setup() As %Status
{
    Write "=== Setting up iFind for RAG ===",!,!
    
    // Step 1: Create new table with iFind
    Set sc = ..CreateIFindTable()
    If $$$ISERR(sc) Return sc
    
    // Step 2: Copy data if source exists
    Set sc = ..CopyDataToIFindTable()
    If $$$ISERR(sc) Return sc
    
    Write !,"✅ Setup complete!",!
    Write "iFind table ready for use",!
    
    Return $$$OK
}

}
