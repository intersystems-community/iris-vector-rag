/// Extension of SourceDocuments table to add iFind functionality
/// This class adds full-text search capabilities to the existing RAG.SourceDocuments table
Class RAG.SourceDocumentsIFind
{

/// Method to create iFind index on existing SourceDocuments table
/// This needs to be run in IRIS terminal or via Management Portal
ClassMethod CreateIFindIndex() As %Status
{
    Try {
        // First, we need to alter the existing table to add iFind support
        // This requires creating a computed property that references the text_content field
        
        Write "Creating iFind index on RAG.SourceDocuments...",!
        
        // Execute DDL to add the index
        // Note: This assumes the table already exists
        &sql(CREATE INDEX idx_SourceDocuments_iFind ON RAG.SourceDocuments (text_content) TYPE iFind)
        
        If SQLCODE = 0 {
            Write "✅ iFind index created successfully!",!
            Return $$$OK
        } Else {
            Write "❌ Failed to create iFind index. SQLCODE: ", SQLCODE,!
            Write "   Error: ", %msg,!
            Return $$$ERROR($$$GeneralError, "Failed to create iFind index: "_SQLCODE_" - "_%msg)
        }
    } Catch ex {
        Write "❌ Exception creating iFind index: ", ex.DisplayString(),!
        Return ex.AsStatus()
    }
}

/// Method to test iFind search functionality
ClassMethod TestIFindSearch(searchTerms As %String) As %Status
{
    Try {
        Write "Testing iFind search for: ", searchTerms,!,!
        
        // Use %CONTAINS predicate for iFind search
        &sql(DECLARE iFindCursor CURSOR FOR
             SELECT TOP 10 doc_id, title, 
                    %ID AS rank
             FROM RAG.SourceDocuments
             WHERE %CONTAINS(text_content, :searchTerms)
             ORDER BY rank DESC)
        
        &sql(OPEN iFindCursor)
        
        Set count = 0
        For {
            &sql(FETCH iFindCursor INTO :docId, :title, :rank)
            Quit:SQLCODE'=0
            
            Set count = count + 1
            Write count, ". Document: ", docId,!
            Write "   Title: ", title,!
            Write "   Rank: ", rank,!,!
        }
        
        &sql(CLOSE iFindCursor)
        
        If count = 0 {
            Write "No documents found matching: ", searchTerms,!
        } Else {
            Write "Found ", count, " documents",!
        }
        
        Return $$$OK
        
    } Catch ex {
        Write "Error during search: ", ex.DisplayString(),!
        Return ex.AsStatus()
    }
}

/// Method to build/rebuild the iFind index
ClassMethod BuildIFindIndex() As %Status
{
    Try {
        Write "Building iFind index...",!
        
        // Build the index
        &sql(BUILD INDEX idx_SourceDocuments_iFind ON RAG.SourceDocuments)
        
        If SQLCODE = 0 {
            Write "✅ iFind index built successfully!",!
            Return $$$OK
        } Else {
            Write "❌ Failed to build iFind index. SQLCODE: ", SQLCODE,!
            Return $$$ERROR($$$GeneralError, "Failed to build iFind index")
        }
        
    } Catch ex {
        Write "Error building index: ", ex.DisplayString(),!
        Return ex.AsStatus()
    }
}

/// Instructions for applying this iFind functionality
ClassMethod Instructions()
{
    Write !,"=== How to Apply iFind to RAG.SourceDocuments ===",!,!
    
    Write "1. Connect to IRIS terminal or Management Portal SQL interface",!
    Write "2. Switch to the namespace containing your RAG schema",!
    Write "3. Run the following commands:",!,!
    
    Write "   DO ##class(RAG.SourceDocumentsIFind).CreateIFindIndex()",!
    Write "   DO ##class(RAG.SourceDocumentsIFind).BuildIFindIndex()",!,!
    
    Write "4. Test the search:",!
    Write "   DO ##class(RAG.SourceDocumentsIFind).TestIFindSearch(""diabetes treatment"")",!,!
    
    Write "5. Update hybrid_ifind_rag/pipeline.py to use %CONTAINS:",!
    Write "   Change: WHERE text_content LIKE ?",!
    Write "   To:     WHERE %CONTAINS(text_content, ?)",!,!
    
    Write "Note: iFind indexes require IRIS to be configured with appropriate language models.",!
    Write "      The index will work best with English text by default.",!
}

}