# iFind Implementation Notes & Current Issues

## Objective
To implement a full-text search capability for the HybridIFindRAG pipeline using InterSystems IRIS's iFind functionality. This involves creating an ObjectScript class (`RAG.SourceDocumentsWithIFind`) with an `%iFind.Index.Basic` on the `text_content` property.

## Final Attempted Class Definition
The last version of the class definition, which includes a `%BuildIndices` method and incorporates syntax corrections based on examples and feedback, is stored in `objectscript/RAG.SourceDocumentsWithIFind_v5_with_build.cls`.

```objectscript
// Content of objectscript/RAG.SourceDocumentsWithIFind_v5_with_build.cls
Class RAG.SourceDocumentsWithIFind Extends %Persistent [ ClassType = persistent, DdlAllowed, SqlTableName = SourceDocumentsIFind ]
{

Property doc_id As %Library.String(MAXLEN = 255) [ Required ]

Property title As %Library.String(MAXLEN = 1000)

Property text_content As %Stream.GlobalCharacter // iFind can index %Stream.GlobalCharacter directly

Property embedding As %Library.String(MAXLEN = 0) // Corrected: unlimited length

Property created_at As %Library.TimeStamp

Index DocIdIndex On doc_id [ Unique ]

Index TextContentFTI On (text_content) As %iFind.Index.Basic(LANGUAGE = "en", LOWER = 1) // Added LANGUAGE and LOWER

Method %BuildIndices() As %Status
{
    Write "Attempting to build/rebuild TextContentFTI index for RAG.SourceDocumentsWithIFind...",!
    
    Set sc = $SYSTEM.INDEX.Build("RAG.SourceDocumentsWithIFind", "TextContentFTI", "/check=0 /nolock") // Build specific index
    
    If sc = 1 {
        Write "Call to $SYSTEM.INDEX.Build for TextContentFTI completed successfully (returned 1).", !
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
```

## Attempts to Load and Build Index Programmatically

The following sequence was attempted after a clean Docker container restart (`iris_db_rag_licensed` using image `containers.intersystems.com/intersystems/iris-arm64:2025.1`):

1.  **Copy class file to container:**
    `docker cp objectscript/RAG.SourceDocumentsWithIFind_v5_with_build.cls iris_db_rag_licensed:/tmp/RAG.SourceDocumentsWithIFind_v5.cls`
    *Result: Successful.*

2.  **Load class definition:**
    Command: `echo 'Set sc = $SYSTEM.OBJ.Load("/tmp/RAG.SourceDocumentsWithIFind_v5.cls") If sc { Write "Class RAG.SourceDocumentsWithIFind (v5) loaded successfully!", ! } Else { Write "Error loading class RAG.SourceDocumentsWithIFind (v5).", ! Write "Error Details: ", $SYSTEM.Status.GetErrorText($SYSTEM.Status.GetLastErrorCode()), ! } Halt' | docker exec -i iris_db_rag_licensed iris session IRIS -U USER`
    *Output: `Load finished successfully. Class RAG.SourceDocumentsWithIFind (v5) loaded successfully!`*

3.  **Attempt to call `%BuildIndices` method (in a new session):**
    Command: `echo 'Set sc = ##class(RAG.SourceDocumentsWithIFind).%BuildIndices() If sc = 1 { Write "%BuildIndices successful (returned 1)", ! } Else { Write "%BuildIndices failed. Error: ", $SYSTEM.Status.GetErrorText(sc), ! } Halt' | docker exec -i iris_db_rag_licensed iris session IRIS -U USER`
    *Output: `<CLASS DOES NOT EXIST> *RAG.SourceDocumentsWithIFind`*

## Contradictory Observations

*   `$SYSTEM.OBJ.Load()` reports successful compilation of the class.
*   Earlier tests with `##class(%Dictionary.ClassDefinition).%ExistsId("RAG.SourceDocumentsWithIFind")` returned `1` after a successful load, confirming the class definition was registered in the dictionary.
*   However, attempts to invoke class methods like `##class(RAG.SourceDocumentsWithIFind).%BuildIndices()` or even `##class(RAG.SourceDocumentsWithIFind).%Kill()` in subsequent non-interactive sessions fail with `<CLASS DOES NOT EXIST>`.

This indicates a fundamental issue with class resolution or visibility in the IRIS environment when using piped `docker exec` commands for sequential operations, or a subtle, unreported compilation issue that prevents the class from being fully usable.

## Final Test Result for HybridIFindRAG Pipeline

The Python test script (`test_hybrid_ifind_rag.py`) consistently fails with:
`ERROR:hybrid_ifind_rag.pipeline:Error in iFind search: [SQLCODE: <-151>:<Index is not found within tables used by this statement>] ... [%msg: <Index TEXTCONTENTFTI not found within tables used by this statement>]`

This confirms the `TextContentFTI` index is not usable by the SQL engine.

## Conclusion & Recommendation

The iFind functionality for the HybridIFindRAG pipeline is currently **BLOCKED**.

Due to persistent issues with either the IRIS environment's handling of class compilation/resolution across non-interactive sessions or subtle, un-surfaced errors in the class/index definition, the iFind index `TextContentFTI` cannot be reliably created and made available to the SQL engine through scripted `docker exec` commands.

**Manual Intervention Required:**
An IRIS developer or administrator with direct access to the IRIS instance (e.g., via IRIS Studio or an interactive IRIS terminal) must perform the following:
1.  Load the class definition from `objectscript/RAG.SourceDocumentsWithIFind_v5_with_build.cls`.
2.  Ensure it compiles without any errors (inspecting compiler output thoroughly).
3.  Explicitly build all indices for the class, particularly `TextContentFTI`. This might involve calling `##class(RAG.SourceDocumentsWithIFind).%BuildIndices()` or using `$SYSTEM.SQL.TuneTable("RAG.SourceDocumentsWithIFind","/build")` from an interactive session.
4.  Verify the index `TextContentFTI` exists and is queryable using direct SQL commands within IRIS.

Until these manual steps are successfully performed and verified, the iFind component of the HybridIFindRAG pipeline will remain non-functional. For current evaluation purposes, it's recommended to focus on the other 6 RAG techniques.