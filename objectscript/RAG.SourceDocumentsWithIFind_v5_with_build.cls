Class RAG.SourceDocumentsWithIFind Extends %Persistent [ ClassType = persistent, DdlAllowed, SqlTableName = SourceDocumentsIFind ]
{

Property doc_id As %Library.String(MAXLEN = 255) [ Required ]

Property title As %Library.String(MAXLEN = 1000)

Property text_content As %Stream.GlobalCharacter // iFind can index %Stream.GlobalCharacter directly

Property embedding As %Library.String(MAXLEN = 0) // Unlimited length for embedding string

Property created_at As %Library.TimeStamp

Index DocIdIndex On doc_id [ Unique ]

Index TextContentFTI On (text_content) As %iFind.Index.Basic(LANGUAGE = "en", LOWER = 1)

Method %BuildIndices() As %Status
{
    // Check if the specific iFind index exists
    // $SYSTEM.SQL.TableExists("RAG.SourceDocumentsIFind") - to check table
    // How to check if a specific index like TextContentFTI exists for this table?
    // For now, let's assume we always want to try building it or let IRIS handle if it's already there.
    // A more robust check would be: IF ##class(%Dictionary.CompiledIndex).%ExistsId("RAG.SourceDocumentsWithIFind||TextContentFTI")

    Write "Attempting to build/rebuild TextContentFTI index for RAG.SourceDocumentsWithIFind...",!
    
    // Option 1: Call system build for all indices of this class (should pick up TextContentFTI)
    // Set sc = ..%BuildIndices() -- This would be recursive if not careful or if this IS the override.
    // Let's use a more direct system utility if possible, or ensure this is the intended override.

    // Option 2: Use $SYSTEM.SQL.TuneTable (already tried, but maybe more reliable after class is definitely loaded)
    // Do $SYSTEM.SQL.TuneTable("RAG.SourceDocumentsIFind","/build")
    // Set sc = $$$OK 

    // Option 3: Specific index build using $SYSTEM.INDEX.Build (as per user feedback)
    // Note: $SYSTEM.INDEX.Build typically takes (IndexName, TableName, Flags)
    // The index name is TextContentFTI. The table name is RAG.SourceDocumentsIFind.
    // However, iFind indices on stream properties are often managed slightly differently.
    // Let's try the most common system method for persistent classes first.
    // If this is an override of a system method, it should call SUPER::%BuildIndices()
    // or directly manage its own indices.

    // For an iFind index defined in the class, IRIS usually handles its build upon class compilation
    // or via TuneTable. If we need to force it, it's often via purging and rebuilding.

    // Let's try a simple approach first: ensure the class is compiled, then TuneTable.
    // The class is already compiled by $SYSTEM.OBJ.Load().
    // So, the main action here is to ensure TuneTable is called effectively.
    // The feedback suggested $SYSTEM.INDEX.Build("TEXTCONTENTFTI", "RAG.SourceDocumentsWithIFind")

    Set sc = $SYSTEM.INDEX.Build("RAG.SourceDocumentsIFind", "TextContentFTI", "/check=0 /nolock") // Build specific index
    
    If sc = 1 {
        Write "Call to $SYSTEM.INDEX.Build for TextContentFTI completed successfully (returned 1).", !
        // Further check if it actually exists now via SQL might be needed
    } Else {
        Write "Call to $SYSTEM.INDEX.Build for TextContentFTI returned ", sc, ". Error: ", $SYSTEM.Status.GetErrorText(sc),!
    }
    Return sc
}

Storage Default
{
<Data name="SourceDocumentsIFindDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>doc_id</Value>
</Value>
<Value name="3">
<Value>title</Value>
</Value>
<Value name="4">
<Value>text_content</Value>
</Value>
<Value name="5">
<Value>embedding</Value>
</Value>
<Value name="6">
<Value>created_at</Value>
</Value>
</Data>
<DataLocation>^RAG.SourceDocsIFindD</DataLocation>
<DefaultData>SourceDocumentsIFindDefaultData</DefaultData>
<IdLocation>^RAG.SourceDocsIFindD</IdLocation>
<IndexLocation>^RAG.SourceDocsIFindI</IndexLocation>
<StreamLocation>^RAG.SourceDocsIFindS</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}