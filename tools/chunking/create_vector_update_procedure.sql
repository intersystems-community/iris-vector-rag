-- Create a stored procedure to update vector columns
CREATE OR REPLACE PROCEDURE RAG.UpdateChunkVector(
    IN chunk_id VARCHAR(255)
)
LANGUAGE OBJECTSCRIPT
{
    NEW SQLCODE, %msg
    
    // Get the embedding string for this chunk
    &sql(SELECT embedding INTO :embeddingStr
         FROM RAG.DocumentChunks_V2
         WHERE chunk_id = :chunk_id)
    
    IF SQLCODE'=0 {
        QUIT SQLCODE
    }
    
    // Convert to vector and update
    TRY {
        SET vectorData = ##class(%Library.Vector).%New("DOUBLE", 384)
        SET embeddingList = $LISTFROMSTRING(embeddingStr, ",")
        
        FOR i=1:1:$LISTLENGTH(embeddingList) {
            DO vectorData.SetAt($DOUBLE($LIST(embeddingList, i)), i)
        }
        
        &sql(UPDATE RAG.DocumentChunks_V2
             SET chunk_embedding_vector = :vectorData
             WHERE chunk_id = :chunk_id)
        
    } CATCH ex {
        SET SQLCODE = ex.AsSQLCODE()
        SET %msg = ex.DisplayString()
    }
    
    QUIT SQLCODE
}