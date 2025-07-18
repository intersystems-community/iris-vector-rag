Class RAG.VectorMigration Extends %RegisteredObject
{

/// Convert a vector stored in VARCHAR format to a string list format
/// that can be parsed by TO_VECTOR() function
/// @param vectorValue The value from the VARCHAR column containing vector data
/// @return String representation like "[1.0,2.0,3.0,...]" or comma-separated values
ClassMethod GetVectorAsStringFromVarchar(vectorValue As %String) As %String [ SqlProc ]
{
    // Handle null or empty input
    If (vectorValue = "") || ($IsObject(vectorValue) = 0) {
        Return ""
    }
    
    Try {
        // If the vectorValue contains @$vector handle, we need to resolve it
        // This might be an OID reference to a vector stored elsewhere
        If $Find(vectorValue, "@$vector") {
            // Try to resolve the vector reference
            // The format might be something like "OIDHEX@$vector"
            Set oid = $Piece(vectorValue, "@", 1)
            
            // Try to get the actual vector data using the OID
            // This is speculative - the exact method depends on how vectors are stored internally
            Set vectorObj = ##class(%Library.DynamicObject).%OpenId(oid)
            If $IsObject(vectorObj) {
                // If it's a dynamic object, try to get the vector data
                Set vectorData = vectorObj.%Get("vector")
                If vectorData '= "" {
                    Return ..FormatVectorAsString(vectorData)
                }
            }
            
            // Alternative approach: try to interpret as a global reference
            // Vector data might be stored in a global with the OID as subscript
            Set globalRef = "^RAG.VectorData("""_oid_""")"
            If $Data(@globalRef) {
                Set vectorData = $Get(@globalRef)
                If vectorData '= "" {
                    Return ..FormatVectorAsString(vectorData)
                }
            }
            
            // Another approach: try to use %SQL.Statement to query the vector
            // This might work if the vector is actually accessible via SQL in some form
            Set stmt = ##class(%SQL.Statement).%New()
            Set sql = "SELECT document_embedding_vector FROM RAG.SourceDocuments WHERE document_embedding_vector = ?"
            Set status = stmt.%Prepare(sql)
            If $$$ISOK(status) {
                Set result = stmt.%Execute(vectorValue)
                If result.%Next() {
                    Set actualVector = result.%Get("document_embedding_vector")
                    // Try to extract numeric components if possible
                    Return ..ExtractNumericComponents(actualVector)
                }
            }
        }
        
        // If not a @$vector handle, try to parse as direct vector data
        Return ..FormatVectorAsString(vectorValue)
        
    } Catch ex {
        // Log the error and return empty string
        Do ##class(%SYS.System).WriteToConsoleLog("Error in GetVectorAsStringFromVarchar: "_ex.DisplayString())
        Return ""
    }
}

/// Format vector data as a string that TO_VECTOR can parse
/// @param vectorData The raw vector data in various possible formats
/// @return Formatted string like "[1.0,2.0,3.0,...]"
ClassMethod FormatVectorAsString(vectorData As %String) As %String [ Private ]
{
    // If already in bracket format, return as-is
    If ($Extract(vectorData, 1) = "[") && ($Extract(vectorData, *) = "]") {
        Return vectorData
    }
    
    // If comma-separated without brackets, add brackets
    If $Find(vectorData, ",") {
        Return "["_vectorData_"]"
    }
    
    // If space-separated, convert to comma-separated with brackets
    If $Find(vectorData, " ") {
        Set commaFormat = $Replace(vectorData, " ", ",")
        Return "["_commaFormat_"]"
    }
    
    // If it's a %List, convert to string format
    If $ListValid(vectorData) {
        Set result = "["
        For i=1:1:$ListLength(vectorData) {
            If i > 1 Set result = result_","
            Set result = result_$List(vectorData, i)
        }
        Set result = result_"]"
        Return result
    }
    
    // Default: assume it's already a valid format
    Return vectorData
}

/// Extract numeric components from complex vector representations
/// @param vectorValue The vector value that might contain numeric data
/// @return String representation of numeric components
ClassMethod ExtractNumericComponents(vectorValue As %String) As %String [ Private ]
{
    Set result = ""
    Set numbers = ""
    
    // Try to extract floating point numbers using regex-like pattern matching
    Set len = $Length(vectorValue)
    Set currentNum = ""
    Set inNumber = 0
    
    For i=1:1:len {
        Set char = $Extract(vectorValue, i)
        
        // Check if character is part of a number (digit, decimal point, minus sign)
        If (char?1N) || (char = ".") || (char = "-") || (char = "e") || (char = "E") || (char = "+") {
            Set currentNum = currentNum_char
            Set inNumber = 1
        } Else {
            // End of number
            If inNumber && (currentNum '= "") {
                If numbers '= "" Set numbers = numbers_","
                Set numbers = numbers_currentNum
                Set currentNum = ""
                Set inNumber = 0
            }
        }
    }
    
    // Handle last number if string ends with a number
    If inNumber && (currentNum '= "") {
        If numbers '= "" Set numbers = numbers_","
        Set numbers = numbers_currentNum
    }
    
    If numbers '= "" {
        Return "["_numbers_"]"
    }
    
    Return ""
}

/// Alternative method that tries to use IRIS vector functions directly
/// @param vectorValue The value from the VARCHAR column
/// @return String representation suitable for TO_VECTOR
ClassMethod GetVectorUsingIRISFunctions(vectorValue As %String) As %String [ SqlProc ]
{
    Try {
        // Try using IRIS built-in vector functions if they exist
        // This is speculative based on what IRIS might provide
        
        // Method 1: Try VECTOR_ELEMENTS if it exists
        Set sql = "SELECT VECTOR_ELEMENTS(?) AS elements"
        Set stmt = ##class(%SQL.Statement).%New()
        Set status = stmt.%Prepare(sql)
        If $$$ISOK(status) {
            Set result = stmt.%Execute(vectorValue)
            If result.%Next() {
                Set elements = result.%Get("elements")
                If elements '= "" Return elements
            }
        }
        
        // Method 2: Try VECTOR_TO_JSON if it exists
        Set sql = "SELECT VECTOR_TO_JSON(?) AS json_vector"
        Set stmt = ##class(%SQL.Statement).%New()
        Set status = stmt.%Prepare(sql)
        If $$$ISOK(status) {
            Set result = stmt.%Execute(vectorValue)
            If result.%Next() {
                Set jsonVector = result.%Get("json_vector")
                If jsonVector '= "" {
                    // Parse JSON to extract array
                    Set dynObj = ##class(%DynamicAbstractObject).%FromJSON(jsonVector)
                    If $IsObject(dynObj) {
                        Return dynObj.%ToJSON()
                    }
                }
            }
        }
        
        // Method 3: Try casting to different vector types to see what works
        Set sql = "SELECT CAST(? AS VECTOR(FLOAT, 128)) AS vector_double"
        Set stmt = ##class(%SQL.Statement).%New()
        Set status = stmt.%Prepare(sql)
        If $$$ISOK(status) {
            Set result = stmt.%Execute(vectorValue)
            If result.%Next() {
                Set vectorDouble = result.%Get("vector_double")
                If vectorDouble '= "" {
                    // If this works, the vector might be accessible this way
                    Return ..FormatVectorAsString(vectorDouble)
                }
            }
        }
        
    } Catch ex {
        Do ##class(%SYS.System).WriteToConsoleLog("Error in GetVectorUsingIRISFunctions: "_ex.DisplayString())
    }
    
    Return ""
}

}