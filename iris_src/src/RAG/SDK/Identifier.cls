/// What the SDK lets into the SQL it builds by concatenation.
///
/// IRIS cannot bind a table or column name as a parameter, so every name the SDK puts in a
/// statement -- the document table, the id, text, embedding and metadata columns, a schema --
/// has to be checked first, and a row count has to be a real integer. These names come from
/// `^RAG.SDK.Config`, from arguments and from `SetDefaultTable`, none of which is trusted to
/// have been checked by whoever wrote it.
Class RAG.SDK.Identifier
{

/// 1 when pName is one SQL identifier: a letter or % then letters, digits and underscores.
ClassMethod IsName(pName As %String) As %Boolean
{
    Set tSC = $$$OK
    Set tValid = 0
    If ($Length(pName) > 0) && ($Length(pName) <= 128) {
        Set tValid = (pName ? 1(1A,1"%").(1AN,1"_"))
    }
    Quit tValid
}

/// 1 when pTable is a table name of one to three identifiers joined by dots.
ClassMethod IsTable(pTable As %String) As %Boolean
{
    Set tSC = $$$OK
    Set tParts = $Length(pTable, ".")
    If (pTable = "") || (tParts > 3) Return 0
    For tI = 1:1:tParts {
        If '..IsName($Piece(pTable, ".", tI)) Return 0
    }
    Quit 1
}

/// Ok when pName is an identifier, otherwise an error naming pRole (for example "text column").
ClassMethod CheckName(pName As %String, pRole As %String) As %Status
{
    Set tSC = $$$OK
    If '..IsName(pName) {
        Set tSC = $$$ERROR($$$GeneralError, "invalid identifier for the "_pRole_": '"_pName_"' (letters, digits and underscores only)")
    }
    Quit tSC
}

/// Ok when pTable is a table name, otherwise an error that says so.
ClassMethod CheckTable(pTable As %String) As %Status
{
    Set tSC = $$$OK
    If '..IsTable(pTable) {
        Set tSC = $$$ERROR($$$GeneralError, "invalid identifier for the table: '"_pTable_"' (one to three names of letters, digits and underscores, joined by dots)")
    }
    Quit tSC
}

/// pValue as a positive integer, or 0 when it is not one. A number that arrives as text from
/// a SQL procedure or a Python caller is accepted; anything else -- 0, a fraction, a
/// negative, a string with SQL in it -- is 0, and callers refuse 0.
ClassMethod TopK(pValue As %String) As %Integer
{
    Set tSC = $$$OK
    Set tTopK = 0
    If (pValue ? 1.9N) && (+pValue > 0) Set tTopK = +pValue
    Quit tTopK
}

}
