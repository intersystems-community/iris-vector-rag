-- common/colbert_udf.sql
-- SQL function definitions for ColBERT MaxSim operations in IRIS

-- MaxSim UDF for ColBERT
-- This function calculates the MaxSim score between query token embeddings 
-- and the token embeddings of a document

CREATE OR REPLACE FUNCTION AppLib.ColbertMaxSimScore(docId VARCHAR(255), queryEmbeddingsJson VARCHAR(MAX)) 
RETURNS DOUBLE LANGUAGE OBJECTSCRIPT {
    Set score = 0
    
    // Parse the query embeddings JSON into a local array
    Set queryEmbArr = ##class(%DynamicArray).%FromJSON(queryEmbeddingsJson)
    Set queryEmbCount = queryEmbArr.%Size()
    
    // Get document token embeddings from the DocumentTokenEmbeddings table
    Set sql = "SELECT token_embedding FROM DocumentTokenEmbeddings WHERE doc_id = ?"
    Set rs = ##class(%SQL.Statement).%ExecDirect(, sql, docId)
    
    If (rs.%SQLCODE < 0) {
        // SQL error occurred
        Return 0
    }
    
    // Initialize array to store max similarity for each query token
    For i=0:1:queryEmbCount-1 {
        Set maxSims(i) = 0
    }
    
    // For each document token
    While rs.%Next() {
        // Get token embedding as string and convert to array of doubles
        Set tokenEmbStr = rs.%Get("token_embedding")
        Set tokenEmbArr = ##class(%DynamicArray).%FromJSON(tokenEmbStr)
        
        // For each query token embedding
        For q=0:1:queryEmbCount-1 {
            Set queryEmb = queryEmbArr.%Get(q)
            
            // Calculate cosine similarity
            Set dotProduct = 0
            Set normDoc = 0
            Set normQuery = 0
            
            For d=0:1:tokenEmbArr.%Size()-1 {
                Set docVal = tokenEmbArr.%Get(d)
                Set queryVal = queryEmb.%Get(d)
                
                Set dotProduct = dotProduct + (docVal * queryVal)
                Set normDoc = normDoc + (docVal * docVal)
                Set normQuery = normQuery + (queryVal * queryVal)
            }
            
            // Avoid division by zero
            If (normDoc = 0) || (normQuery = 0) {
                Set similarity = 0
            } Else {
                Set similarity = dotProduct / (($ZSQR(normDoc) * $ZSQR(normQuery)))
            }
            
            // Update max similarity for this query token if higher
            If similarity > maxSims(q) {
                Set maxSims(q) = similarity
            }
        }
    }
    
    // Sum up the max similarities for each query token
    For i=0:1:queryEmbCount-1 {
        Set score = score + maxSims(i)
    }
    
    Return score
}

-- Simplified vector operations wrapper functions
-- These are utility functions for vector operations in SQL

CREATE OR REPLACE FUNCTION AppLib.StringToVector(vectorStr VARCHAR(MAX)) 
RETURNS VECTOR LANGUAGE OBJECTSCRIPT {
    // Remove brackets and split by commas
    Set str = $REPLACE(vectorStr, "[", "")
    Set str = $REPLACE(str, "]", "")
    Set values = $LISTFROMSTRING(str, ",")
    
    // Convert to vector format
    Set vector = ""
    Set count = $LISTLENGTH(values)
    
    For i=1:1:count {
        Set value = $LISTGET(values, i)
        // Append to vector string
        If i>1 Set vector = vector _ ","
        Set vector = vector _ $NORMALIZE(value)
    }
    
    Return vector
}

CREATE OR REPLACE FUNCTION AppLib.DotProduct(vec1 VECTOR, vec2 VECTOR) 
RETURNS DOUBLE LANGUAGE OBJECTSCRIPT {
    // Extract values from vectors
    Set values1 = $LISTFROMSTRING(vec1, ",")
    Set values2 = $LISTFROMSTRING(vec2, ",")
    Set count = $LISTLENGTH(values1)
    
    // Calculate dot product
    Set product = 0
    For i=1:1:count {
        Set val1 = $LISTGET(values1, i)
        Set val2 = $LISTGET(values2, i)
        Set product = product + (val1 * val2)
    }
    
    Return product
}

-- Register the functions for CREATE OR REPLACE to work
-- (May require administrative privileges)
GRANT EXECUTE ON FUNCTION AppLib.ColbertMaxSimScore TO PUBLIC;
GRANT EXECUTE ON FUNCTION AppLib.StringToVector TO PUBLIC;
GRANT EXECUTE ON FUNCTION AppLib.DotProduct TO PUBLIC;
