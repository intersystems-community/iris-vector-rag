/// Service for processing and indexing keywords for iFind integration
/// This class handles document keyword extraction, indexing, and bitmap generation
/// for the Hybrid iFind+Graph+Vector RAG technique.
Class RAGDemo.KeywordProcessor Extends %RegisteredObject
{

/// Extract and index keywords from document content
ClassMethod IndexDocument(docId As %Integer, content As %String) As %Status
{
    Set sc = $$$OK
    
    Try {
        // Extract keywords from content
        Set keywords = ..ExtractKeywords(content)
        
        // Clear existing keywords for this document
        &sql(DELETE FROM keyword_index WHERE document_id = :docId)
        
        // Index each keyword
        Set keywordList = $LISTFROMSTRING(keywords, ",")
        Set keywordCount = $LISTLENGTH(keywordList)
        
        For i=1:1:keywordCount {
            Set keyword = $ZSTRIP($LIST(keywordList, i), "<>W")
            Continue:keyword=""
            Continue:$LENGTH(keyword)<3  // Skip short keywords
            
            // Calculate frequency and positions
            Set frequency = ..CalculateKeywordFrequency(content, keyword)
            Set positions = ..FindKeywordPositions(content, keyword)
            
            // Insert keyword index entry
            &sql(INSERT INTO keyword_index 
                 (document_id, keyword, frequency, position_data)
                 VALUES (:docId, :keyword, :frequency, :positions))
            
            If SQLCODE < 0 {
                Set sc = $$$ERROR($$$SQLError, SQLCODE, $GET(%msg))
                Quit
            }
        }
        
        // Update bitmap chunks for affected keywords
        If $$$ISOK(sc) {
            Set sc = ..UpdateBitmapChunks(keywords)
        }
        
    } Catch ex {
        Set sc = ex.AsStatus()
    }
    
    Return sc
}

/// Extract meaningful keywords from document content
ClassMethod ExtractKeywords(content As %String) As %String
{
    Set keywords = ""
    Set content = $ZCONVERT(content, "L")  // Convert to lowercase
    
    // Define stop words
    Set stopWords = ",the,a,an,and,or,but,in,on,at,to,for,of,with,by,is,are,was,were,be,been,have,has,had,do,does,did,will,would,could,should,may,might,can,what,how,when,where,why,who,this,that,these,those,i,you,he,she,it,we,they,me,him,her,us,them,my,your,his,its,our,their,"
    
    // Extract words using simple pattern matching
    Set pos = 1
    Set wordPattern = "[A-Za-z0-9]+"
    
    While pos <= $LENGTH(content) {
        Set match = $LOCATE(content, wordPattern, pos)
        If match = 0 Quit
        
        Set word = $EXTRACT(content, $PIECE(match, ",", 1), $PIECE(match, ",", 2))
        Set pos = $PIECE(match, ",", 2) + 1
        
        // Filter out stop words and short words
        If $LENGTH(word) < 3 Continue
        If ","_stopWords_"," [ (","_word_",") Continue
        
        // Add to keyword list if not already present
        If keywords '[ word {
            Set keywords = keywords_$SELECT(keywords="": "", 1: ",")_word
        }
    }
    
    Return keywords
}

/// Calculate frequency of keyword in content
ClassMethod CalculateKeywordFrequency(content As %String, keyword As %String) As %Integer
{
    Set frequency = 0
    Set content = $ZCONVERT(content, "L")
    Set keyword = $ZCONVERT(keyword, "L")
    
    Set pos = 1
    While pos <= $LENGTH(content) {
        Set found = $FIND(content, keyword, pos)
        If found = 0 Quit
        
        Set frequency = frequency + 1
        Set pos = found
    }
    
    Return frequency
}

/// Find positions of keyword in content
ClassMethod FindKeywordPositions(content As %String, keyword As %String) As %String
{
    Set positions = ""
    Set content = $ZCONVERT(content, "L")
    Set keyword = $ZCONVERT(keyword, "L")
    
    Set pos = 1
    While pos <= $LENGTH(content) {
        Set found = $FIND(content, keyword, pos)
        If found = 0 Quit
        
        Set startPos = found - $LENGTH(keyword)
        Set positions = positions_$SELECT(positions="": "", 1: ",")_startPos
        Set pos = found
    }
    
    Return positions
}

/// Build bitmap chunks for efficient searching
ClassMethod BuildBitmapChunks(keyword As %String) As %Status
{
    Set sc = $$$OK
    
    Try {
        // Clear existing bitmap chunks for this keyword
        &sql(DELETE FROM keyword_bitmap_chunks WHERE keyword = :keyword)
        
        // Get all documents containing this keyword
        &sql(DECLARE C1 CURSOR FOR
             SELECT document_id, frequency
             FROM keyword_index
             WHERE keyword = :keyword
             ORDER BY document_id)
        
        &sql(OPEN C1)
        
        Set chunkSize = 1000  // Documents per chunk
        Set currentChunk = 1
        Set chunkData = ""
        Set docCount = 0
        
        For {
            &sql(FETCH C1 INTO :docId, :frequency)
            Quit:SQLCODE'=0
            
            // Add document to current chunk
            Set chunkData = chunkData_$SELECT(chunkData="": "", 1: ",")_docId_":"_frequency
            Set docCount = docCount + 1
            
            // If chunk is full, save it and start new chunk
            If docCount >= chunkSize {
                &sql(INSERT INTO keyword_bitmap_chunks
                     (keyword, chunk_number, bitmap_data)
                     VALUES (:keyword, :currentChunk, :chunkData))
                
                If SQLCODE < 0 {
                    Set sc = $$$ERROR($$$SQLError, SQLCODE, $GET(%msg))
                    Quit
                }
                
                Set currentChunk = currentChunk + 1
                Set chunkData = ""
                Set docCount = 0
            }
        }
        
        // Save final chunk if it has data
        If chunkData '= "" {
            &sql(INSERT INTO keyword_bitmap_chunks
                 (keyword, chunk_number, bitmap_data)
                 VALUES (:keyword, :currentChunk, :chunkData))
            
            If SQLCODE < 0 {
                Set sc = $$$ERROR($$$SQLError, SQLCODE, $GET(%msg))
            }
        }
        
        &sql(CLOSE C1)
        
    } Catch ex {
        Set sc = ex.AsStatus()
    }
    
    Return sc
}

/// Update bitmap chunks for multiple keywords
ClassMethod UpdateBitmapChunks(keywords As %String) As %Status
{
    Set sc = $$$OK
    
    Set keywordList = $LISTFROMSTRING(keywords, ",")
    Set keywordCount = $LISTLENGTH(keywordList)
    
    For i=1:1:keywordCount {
        Set keyword = $ZSTRIP($LIST(keywordList, i), "<>W")
        Continue:keyword=""
        
        Set sc = ..BuildBitmapChunks(keyword)
        If $$$ISERR(sc) Quit
    }
    
    Return sc
}

/// Rebuild all keyword indexes and bitmap chunks
ClassMethod RebuildAllIndexes() As %Status
{
    Set sc = $$$OK
    
    Try {
        Write "Rebuilding keyword indexes...", !
        
        // Clear all existing keyword data
        &sql(DELETE FROM keyword_bitmap_chunks)
        &sql(DELETE FROM keyword_index)
        
        // Process all documents
        &sql(DECLARE C2 CURSOR FOR
             SELECT id, content
             FROM documents
             WHERE content IS NOT NULL
             ORDER BY id)
        
        &sql(OPEN C2)
        
        Set docCount = 0
        For {
            &sql(FETCH C2 INTO :docId, :content)
            Quit:SQLCODE'=0
            
            Set sc = ..IndexDocument(docId, content)
            If $$$ISERR(sc) {
                Write "Error indexing document ", docId, ": ", $SYSTEM.Status.GetErrorText(sc), !
                Continue
            }
            
            Set docCount = docCount + 1
            If docCount # 100 = 0 {
                Write "Processed ", docCount, " documents...", !
            }
        }
        
        &sql(CLOSE C2)
        
        Write "Completed indexing ", docCount, " documents", !
        
    } Catch ex {
        Set sc = ex.AsStatus()
        Write "Error during rebuild: ", $SYSTEM.Status.GetErrorText(sc), !
    }
    
    Return sc
}

/// Get keyword statistics
ClassMethod GetKeywordStats() As %Status
{
    Write "Keyword Index Statistics:", !
    Write "========================", !
    
    // Total keywords
    &sql(SELECT COUNT(DISTINCT keyword) INTO :totalKeywords FROM keyword_index)
    Write "Total unique keywords: ", totalKeywords, !
    
    // Total keyword entries
    &sql(SELECT COUNT(*) INTO :totalEntries FROM keyword_index)
    Write "Total keyword entries: ", totalEntries, !
    
    // Total bitmap chunks
    &sql(SELECT COUNT(*) INTO :totalChunks FROM keyword_bitmap_chunks)
    Write "Total bitmap chunks: ", totalChunks, !
    
    // Top keywords by frequency
    Write !, "Top 10 keywords by document count:", !
    &sql(DECLARE C3 CURSOR FOR
         SELECT keyword, COUNT(*) as doc_count, SUM(frequency) as total_freq
         FROM keyword_index
         GROUP BY keyword
         ORDER BY doc_count DESC, total_freq DESC)
    
    &sql(OPEN C3)
    
    Set rank = 0
    For {
        &sql(FETCH C3 INTO :keyword, :docCount, :totalFreq)
        Quit:SQLCODE'=0
        
        Set rank = rank + 1
        Write rank, ". ", keyword, " (", docCount, " docs, ", totalFreq, " total occurrences)", !
        
        Quit:rank>=10
    }
    
    &sql(CLOSE C3)
    
    Return $$$OK
}

}