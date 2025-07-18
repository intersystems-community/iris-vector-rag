/// Custom iFind implementation for keyword search in RAG pipeline
/// This class extends %SQL.AbstractFind to provide bitmap-based keyword matching
/// for the Hybrid iFind+Graph+Vector RAG technique.
Class RAGDemo.KeywordFinder Extends %SQL.AbstractFind
{

/// Internal storage for keywords to search
Property Keywords As %String [ MultiDimensional ];

/// Internal storage for document bitmap chunks
Property BitmapChunks As %String [ MultiDimensional ];

/// Current keyword set for matching
Property CurrentKeywords As %String;

/// Initialize keyword search with query terms
Method Initialize(keywords As %String) As %Status
{
    Set ..CurrentKeywords = keywords
    
    // Parse keywords (comma or space separated)
    Set keywordList = $LISTFROMSTRING(keywords, " ")
    Set keywordCount = $LISTLENGTH(keywordList)
    
    // Store keywords for matching
    For i=1:1:keywordCount {
        Set keyword = $ZSTRIP($LIST(keywordList, i), "<>W")
        Continue:keyword=""
        Set ..Keywords(keyword) = ""
    }
    
    // Load bitmap chunks for these keywords
    Do ..LoadBitmapChunks()
    
    Return $$$OK
}

/// Load bitmap chunks for current keywords from database
Method LoadBitmapChunks() As %Status
{
    Set keyword = ""
    For {
        Set keyword = $ORDER(..Keywords(keyword))
        Quit:keyword=""
        
        // Query bitmap chunks for this keyword
        &sql(DECLARE C1 CURSOR FOR
             SELECT chunk_number, bitmap_data
             FROM keyword_bitmap_chunks
             WHERE keyword = :keyword
             ORDER BY chunk_number)
        
        &sql(OPEN C1)
        
        For {
            &sql(FETCH C1 INTO :chunkNum, :bitmapData)
            Quit:SQLCODE'=0
            
            Set ..BitmapChunks(keyword, chunkNum) = bitmapData
        }
        
        &sql(CLOSE C1)
    }
    
    Return $$$OK
}

/// Check if document ID exists in keyword set
Method ContainsItem(docId As %String) As %Boolean
{
    // Check if document contains any of our keywords
    Set keyword = ""
    For {
        Set keyword = $ORDER(..Keywords(keyword))
        Quit:keyword=""
        
        // Simple check: look for keyword in document content
        &sql(SELECT COUNT(*) INTO :count
             FROM keyword_index
             WHERE document_id = :docId AND keyword = :keyword)
        
        If count > 0 Return 1
    }
    
    Return 0
}

/// Get bitmap chunk for chunk number
Method GetChunk(chunkNum As %Integer) As %String
{
    // Return combined bitmap for all keywords at this chunk
    Set combinedBitmap = ""
    Set keyword = ""
    
    For {
        Set keyword = $ORDER(..Keywords(keyword))
        Quit:keyword=""
        
        Set bitmap = $GET(..BitmapChunks(keyword, chunkNum))
        If bitmap '= "" {
            // Combine bitmaps (OR operation)
            Set combinedBitmap = ..CombineBitmaps(combinedBitmap, bitmap)
        }
    }
    
    Return combinedBitmap
}

/// Get next chunk after specified chunk number
Method NextChunk(ByRef chunkNum As %Integer) As %String
{
    Set nextChunk = chunkNum + 1
    Set found = 0
    
    // Find next chunk that has data for any keyword
    For chunk=nextChunk:1:9999 {
        Set keyword = ""
        For {
            Set keyword = $ORDER(..Keywords(keyword))
            Quit:keyword=""
            
            If $DATA(..BitmapChunks(keyword, chunk)) {
                Set chunkNum = chunk
                Set found = 1
                Quit
            }
        }
        Quit:found
    }
    
    If 'found Set chunkNum = ""
    Return $SELECT(found: ..GetChunk(chunkNum), 1: "")
}

/// Get previous chunk before specified chunk number
Method PreviousChunk(ByRef chunkNum As %Integer) As %String
{
    Set prevChunk = chunkNum - 1
    Set found = 0
    
    // Find previous chunk that has data for any keyword
    For chunk=prevChunk:-1:1 {
        Set keyword = ""
        For {
            Set keyword = $ORDER(..Keywords(keyword))
            Quit:keyword=""
            
            If $DATA(..BitmapChunks(keyword, chunk)) {
                Set chunkNum = chunk
                Set found = 1
                Quit
            }
        }
        Quit:found
    }
    
    If 'found Set chunkNum = ""
    Return $SELECT(found: ..GetChunk(chunkNum), 1: "")
}

/// Combine two bitmap strings using OR operation
Method CombineBitmaps(bitmap1 As %String, bitmap2 As %String) As %String
{
    // Simple bitmap combination - in production would use proper bitmap operations
    If bitmap1 = "" Return bitmap2
    If bitmap2 = "" Return bitmap1
    
    // For now, just concatenate - real implementation would do bitwise OR
    Return bitmap1_","_bitmap2
}

/// Factory method to create keyword finder for specific keywords
ClassMethod CreateKeywordFinder(keywords As %String) As RAGDemo.KeywordFinder
{
    Set finder = ..%New()
    Do finder.Initialize(keywords)
    Return finder
}

}