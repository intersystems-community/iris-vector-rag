/// Tests for RAG.SDK.Search as an external consumer calls it: a table the SDK did not
/// create, a predicate the caller owns, and rows that have to carry their own attributes.
///
/// Two fixture tables exist because the element type of the embedding column is the thing
/// under test in half of these: FloatDocs is what every Python path in this repo creates,
/// DoubleDocs is what RAG.SDK.Schema.Initialize creates. DoubleDocs deliberately has no
/// metadata column, so "table without metadata" is covered without a third table.
///
/// No method here is embedded Python. Embedded Python does not compile on this project's
/// container -- see specs/085-sdk-search-filter/research.md finding 6.
Class RAG.SDK.Test.SearchTest Extends %UnitTest.TestCase
{

Parameter FLOATTABLE = "RAGTest085.FloatDocs";

Parameter DOUBLETABLE = "RAGTest085.DoubleDocs";

Parameter VECDIM = 8;

Method OnBeforeAllTests() As %Status
{
    Do ..SaveConfig()
    Do ..MakeFixture(..#FLOATTABLE, "FLOAT", 1)
    Do ..MakeFixture(..#DOUBLETABLE, "DOUBLE", 0)
    Return $$$OK
}

Method OnAfterAllTests() As %Status
{
    Do ..DropFixture(..#FLOATTABLE)
    Do ..DropFixture(..#DOUBLETABLE)
    Do ..RestoreConfig()
    Return $$$OK
}

/// Every test that repoints the SDK leaves the config as it found it, so a test that
/// forgets to set a key cannot pass on the previous test's setting.
Method TearDown() As %Status
{
    Kill ^RAG.SDK.Config
    Return $$$OK
}

// --- User Story 3: search works against a FLOAT embedding column ---------------------

/// Every Python path in this repo creates VECTOR(FLOAT, N). Before this feature the SDK
/// hardcoded TO_VECTOR(?,DOUBLE,N) and IRIS refused the comparison with SQLCODE -259, so
/// the caller got an error object where it expected an array.
Method TestVectorSearchAgainstFloatColumn()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set json = ##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 3)
    Do $$$AssertEquals($Extract(json, 1), "[", "FLOAT search returns an array, not "_json)
    Set hits = [].%FromJSON(json)
    Do $$$AssertEquals(hits.%Size(), 3, "three hits returned")
    Do $$$AssertEquals(hits.%Get(0).id, "b-1", "nearest row first")
    Do $$$AssertTrue(+hits.%Get(0).score > +hits.%Get(2).score, "scores descend")
}

/// The DOUBLE case worked before this feature and has to keep working: resolving the type
/// must not become a new way to break the tables the SDK creates itself.
Method TestVectorSearchAgainstDoubleColumn()
{
    Do ..PointAt(..#DOUBLETABLE)
    Set json = ##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 3)
    Do $$$AssertEquals($Extract(json, 1), "[", "DOUBLE search returns an array, not "_json)
    Set hits = [].%FromJSON(json)
    Do $$$AssertEquals(hits.%Size(), 3, "three hits returned")
    Do $$$AssertEquals(hits.%Get(0).id, "b-1", "nearest row first")
}

/// A consumer's table stores its text in a plain VARCHAR, not a stream. The SDK used to
/// open every text value as a stream OID unconditionally, which threw <LIST> and took the
/// whole search down for exactly the tables this feature exists to support.
Method TestVectorSearchPlainVarcharTextColumn()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set json = ##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 1)
    Do $$$AssertEquals($Extract(json, 1), "[", "varchar text column does not break the search, got "_json)
    Set hits = [].%FromJSON(json)
    Do $$$AssertEquals(hits.%Get(0).text, "beta apex alpha document", "the row's text comes back verbatim")
}

Method TestGetVectorTypeResolvesElementType()
{
    Do $$$AssertEquals(##class(RAG.SDK.Search).GetVectorType(..#FLOATTABLE, "embedding"), "FLOAT", "float fixture reports FLOAT")
    Do $$$AssertEquals(##class(RAG.SDK.Search).GetVectorType(..#DOUBLETABLE, "embedding"), "DOUBLE", "double fixture reports DOUBLE")
}

/// The declared width of the column, read from the dictionary. Counting commas in a value
/// read back out of the column reports 1 for every width, because %GetData on a VECTOR
/// column returns the packed vector and not the text that went in.
Method TestGetVectorDimResolvesDeclaredWidth()
{
    Do $$$AssertEquals(##class(RAG.SDK.Search).GetVectorDim(..#FLOATTABLE, "embedding"), ..#VECDIM, "float fixture reports its declared width")
    Do $$$AssertEquals(##class(RAG.SDK.Search).GetVectorDim(..#DOUBLETABLE, "embedding"), ..#VECDIM, "double fixture reports its declared width")
    Do $$$AssertEquals(##class(RAG.SDK.Search).GetVectorDim(..#FLOATTABLE, "nosuchcol"), 0, "an unresolvable column reports 0")
}

/// An unresolvable type has to say which table and column it could not resolve. A caller
/// looking at "" or a bare -259 cannot tell a typo from a driver problem.
Method TestVectorSearchUnresolvableTypeNamesTableAndColumn()
{
    Do $$$AssertEquals(##class(RAG.SDK.Search).GetVectorType(..#FLOATTABLE, "nosuchcol"), "", "unknown column resolves to nothing")

    Set ^RAG.SDK.Config("default_table") = ..#FLOATTABLE
    Set ^RAG.SDK.Config("default_id_col") = "doc_id"
    Set ^RAG.SDK.Config("default_text_col") = "text_content"
    Set ^RAG.SDK.Config("default_emb_col") = "nosuchcol"
    Set json = ##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 3)
    Set obj = {}.%FromJSON(json)
    Do $$$AssertNotEquals(obj.error, "", "an error object comes back")
    Do $$$AssertTrue($Find(obj.error, ..#FLOATTABLE) > 0, "the error names the table: "_obj.error)
    Do $$$AssertTrue($Find(obj.error, "nosuchcol") > 0, "the error names the column: "_obj.error)
}

// --- User Story 1: a caller-supplied predicate scopes the search ----------------------

/// The unscoped nearest three rows all belong to agent-b, so a scoped search that returns
/// three agent-a rows can only have honoured the predicate.
Method TestVectorSearchWhereClauseScopesResults()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set json = ##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 3, "agent_id = 'agent-a'")
    Do $$$AssertEquals($Extract(json, 1), "[", "scoped search returns an array, not "_json)
    Set hits = [].%FromJSON(json)
    Do $$$AssertEquals(hits.%Size(), 3, "three rows are in scope and three come back")
    For i = 0:1:2 {
        Do $$$AssertEquals($Extract(hits.%Get(i).id, 1, 2), "a-", "hit "_i_" is an agent-a row, got "_hits.%Get(i).id)
    }
    Do $$$AssertEquals(hits.%Get(0).id, "a-1", "nearest in-scope row first")
}

Method TestVectorSearchEmptyWhereClauseUnchanged()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set omitted = ##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 3)
    Set explicit = ##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 3, "")
    Do $$$AssertEquals(omitted, explicit, "an empty predicate is the same as no predicate")
    Set hits = [].%FromJSON(omitted)
    Do $$$AssertEquals(hits.%Get(0).id, "b-1", "unscoped search still returns the globally nearest row")
}

/// A predicate no row satisfies is a legitimate answer, not a failure.
Method TestVectorSearchWhereClauseMatchingNothing()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set json = ##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 3, "agent_id = 'nobody'")
    Do $$$AssertEquals(json, "[]", "no matches is an empty array, got "_json)
}

/// The opposite case: a predicate the database rejects must not look like "no matches".
Method TestVectorSearchInvalidWhereClauseNamesFailure()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set json = ##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 3, "nosuchcol = 1")
    Do $$$AssertEquals($Extract(json, 1), "{", "a broken predicate returns an error object, got "_json)
    Set obj = {}.%FromJSON(json)
    Do $$$AssertNotEquals(obj.error, "", "the error is named")
}

/// The fragment is a predicate, not a statement. Anything that could end the statement or
/// comment out the rest of it is refused before it reaches %Prepare.
Method TestVectorSearchWhereClauseRejectsStatementBreak()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set bad(1) = "agent_id = 'agent-a'; DROP TABLE "_..#FLOATTABLE
    Set bad(2) = "agent_id = 'agent-a' -- rest of the clause"
    Set bad(3) = "agent_id = 'agent-a' /* rest of the clause */"
    For i = 1:1:3 {
        Set json = ##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 3, bad(i))
        Do $$$AssertEquals($Extract(json, 1), "{", "predicate "_i_" is refused, got "_json)
        Set obj = {}.%FromJSON(json)
        Do $$$AssertTrue($Find(obj.error, "single predicate") > 0, "the refusal says why: "_obj.error)
    }
    Do $$$AssertTrue(##class(%SYSTEM.SQL.Schema).TableExists(..#FLOATTABLE), "the fixture table survived")
}

/// The IVF index is built over a whole table and its Search() has no predicate argument, so
/// a scoped search has to take the SQL path. The index name here does not exist: if the fast
/// path were taken at all, this test would see an error instead of scoped rows.
Method TestVectorSearchWhereClauseIgnoresIvfFastPath()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set ^RAG.SDK.Config("ivf_index_name") = "no-such-ivf-index"
    Set json = ##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 3, "agent_id = 'agent-a'")
    Do $$$AssertEquals($Extract(json, 1), "[", "scoped search bypasses the IVF path, got "_json)
    Set hits = [].%FromJSON(json)
    Do $$$AssertEquals(hits.%Size(), 3, "three scoped rows come back")
    Do $$$AssertEquals($Extract(hits.%Get(0).id, 1, 2), "a-", "and they are the scoped ones")
}

Method TestTextSearchWhereClauseScopesResults()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set json = ##class(RAG.SDK.Search).TextSearch("alpha", 10, "agent_id = 'agent-a'")
    Do $$$AssertEquals($Extract(json, 1), "[", "scoped text search returns an array, got "_json)
    Set hits = [].%FromJSON(json)
    Do $$$AssertTrue(hits.%Size() > 0, "the scoped text search matches something")
    For i = 0:1:(hits.%Size() - 1) {
        Do $$$AssertEquals($Extract(hits.%Get(i).id, 1, 2), "a-", "text hit "_i_" is in scope, got "_hits.%Get(i).id)
    }
}

Method TestHybridSearchWhereClauseScopesResults()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set json = ##class(RAG.SDK.Search).HybridSearch(..QueryVec(), "alpha", 3, "RRF", "agent_id = 'agent-a'")
    Do $$$AssertEquals($Extract(json, 1), "[", "scoped hybrid search returns an array, got "_json)
    Set hits = [].%FromJSON(json)
    Do $$$AssertEquals(hits.%Size(), 3, "three fused rows come back")
    For i = 0:1:(hits.%Size() - 1) {
        Do $$$AssertEquals($Extract(hits.%Get(i).id, 1, 2), "a-", "fused hit "_i_" is in scope, got "_hits.%Get(i).id)
    }
}

// --- User Story 4: fused results keep the text both legs already read ------------------

/// Both legs select the text column and both throw it away when they fuse. A caller that
/// wants the text of a hybrid hit has to query for it again, once per hit.
Method TestHybridSearchRrfCarriesText()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set json = ##class(RAG.SDK.Search).HybridSearch(..QueryVec(), "alpha", 3, "RRF", "agent_id = 'agent-a'")
    Do $$$AssertEquals($Extract(json, 1), "[", "RRF hybrid returns an array, got "_json)
    Set hits = [].%FromJSON(json)
    Do $$$AssertEquals(hits.%Size(), 3, "three fused rows come back")
    For i = 0:1:(hits.%Size() - 1) {
        Do $$$AssertNotEquals(hits.%Get(i).text, "", "fused hit "_hits.%Get(i).id_" carries its text")
    }
    Set top = hits.%Get(0)
    Do $$$AssertEquals(top.text, "alpha first document", "and the text is the row's own")
    Do $$$AssertEquals(top.metadata.owner, "agent-a", "the row's metadata comes through fusion too")
}

Method TestHybridSearchLinearCarriesText()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set json = ##class(RAG.SDK.Search).HybridSearch(..QueryVec(), "alpha", 3, "linear", "agent_id = 'agent-a'")
    Do $$$AssertEquals($Extract(json, 1), "[", "linear hybrid returns an array, got "_json)
    Set hits = [].%FromJSON(json)
    Do $$$AssertEquals(hits.%Size(), 3, "three fused rows come back")
    For i = 0:1:(hits.%Size() - 1) {
        Do $$$AssertNotEquals(hits.%Get(i).text, "", "fused hit "_hits.%Get(i).id_" carries its text")
    }
    Do $$$AssertEquals(hits.%Get(0).text, "alpha first document", "and the text is the row's own")
}

/// A row only one leg found has to carry its text as well: the text leg is the only place
/// "fifth" appears, and the vector leg's rows are all nearer than it.
Method TestHybridSearchSingleLegRowCarriesText()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set json = ##class(RAG.SDK.Search).HybridSearch(..QueryVec(), "fifth", 3, "RRF", "agent_id = 'agent-a'")
    Do $$$AssertEquals($Extract(json, 1), "[", "hybrid returns an array, got "_json)
    Set hit = ..FindHit([].%FromJSON(json), "a-5")
    Do $$$AssertTrue($IsObject(hit), "the text-only row is in the fused results: "_json)
    If '$IsObject(hit) Return
    Do $$$AssertEquals(hit.text, "alpha fifth document", "and it carries its text")
}

/// The hit with the given id, or "" when the array has none.
ClassMethod FindHit(arr As %DynamicArray, id As %String) As %RawString [ Private ]
{
    For i = 0:1:(arr.%Size() - 1) {
        If arr.%Get(i).id = id Return arr.%Get(i)
    }
    Return ""
}

// --- User Story 2: rows carry their own attributes ------------------------------------

/// The point of the metadata passthrough: one search, no follow-up query per hit.
Method TestVectorSearchReturnsMetadata()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set json = ##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 1, "agent_id = 'agent-a'")
    Do $$$AssertEquals($Extract(json, 1), "[", "search returns an array, got "_json)
    Set hit = [].%FromJSON(json).%Get(0)
    Do $$$AssertEquals(hit.id, "a-1", "the nearest in-scope row")
    Do $$$AssertTrue($IsObject(hit.metadata), "the hit carries a metadata object")
    Do $$$AssertEquals(hit.metadata.owner, "agent-a", "the row's own metadata came back")
    Do $$$AssertEquals(+hit.metadata.rank, 1, "including its non-string values")
}

/// A table with no metadata column is not an error, and must not grow an empty key.
Method TestVectorSearchNoMetadataColumnOmitsKey()
{
    Do ..PointAt(..#DOUBLETABLE)
    Set json = ##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 1)
    Do $$$AssertEquals($Extract(json, 1), "[", "search still succeeds, got "_json)
    Set hit = [].%FromJSON(json).%Get(0)
    Do $$$AssertEquals(hit.%IsDefined("metadata"), 0, "no metadata key on the hit")
    Do $$$AssertEquals(hit.id, "b-1", "and the hit is otherwise unchanged")
}

/// Metadata that is not JSON is the caller's data, not a reason to fail their search.
Method TestVectorSearchNonJsonMetadataCarriedThrough()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set json = ##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 1, "doc_id = 'a-5'")
    Do $$$AssertEquals($Extract(json, 1), "[", "search succeeds, got "_json)
    Set hit = [].%FromJSON(json).%Get(0)
    Do $$$AssertEquals(hit.id, "a-5", "the row with unparseable metadata")
    Do $$$AssertEquals(hit.metadata, "not json at all", "its metadata came back as text")
}

Method TestAttachTableRecordsMetadataColumn()
{
    Set json = ##class(RAG.SDK.Bridge).AttachTable(..#FLOATTABLE, "doc_id", "text_content", "embedding", "sdk085-fixture", "attrs")
    Set obj = {}.%FromJSON(json)
    Do $$$AssertEquals(obj.%Get("error"), "", "attaching the fixture succeeds: "_json)
    Do $$$AssertEquals(obj.%Get("metadata_col"), "attrs", "the metadata column is recorded")
    Do $$$AssertEquals(obj.%Get("vector_type"), "FLOAT", "so is the vector element type")
    Do $$$AssertEquals(obj.dimension, ..#VECDIM, "and the detected dimension is right")

    Set bad = {}.%FromJSON(##class(RAG.SDK.Bridge).AttachTable(..#FLOATTABLE, "doc_id", "text_content", "embedding", "sdk085-fixture", "nosuchcol"))
    Do $$$AssertNotEquals(bad.%Get("error"), "", "a metadata column that does not exist is refused")

    Do ..CleanMapping("sdk085-fixture")
}

/// Omitting the metadata column has to clear a previously configured one. Leaving a stale
/// setting behind would make the SDK select a column the new table does not have.
Method TestSetDefaultTableClearsMetadataColumn()
{
    Do ##class(RAG.SDK.Bridge).SetDefaultTable(..#FLOATTABLE, "doc_id", "text_content", "embedding", "attrs")
    Set obj = {}.%FromJSON(##class(RAG.SDK.Bridge).GetDefaultTable())
    Do $$$AssertEquals(obj.%Get("metadata_col"), "attrs", "the metadata column is reported")

    Do ##class(RAG.SDK.Bridge).SetDefaultTable(..#DOUBLETABLE, "doc_id", "text_content", "embedding")
    Set obj2 = {}.%FromJSON(##class(RAG.SDK.Bridge).GetDefaultTable())
    Do $$$AssertEquals(obj2.%Get("metadata_col"), "", "omitting it clears the previous setting")
    Do $$$AssertEquals($Get(^RAG.SDK.Config("default_meta_col"), ""), "", "and clears the config key itself")
}

// --- Feature 070's declared coverage, which was never written -------------------------

/// 070 marked this method delivered. It was not, so the SDK shipped with no test that the
/// nearest row is the one returned first.
Method TestVectorSearchReturnsTopResult()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set json = ##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 5)
    Do $$$AssertEquals($Extract(json, 1), "[", "search returns an array, got "_json)
    Set hits = [].%FromJSON(json)
    Do $$$AssertEquals(hits.%Get(0).id, "b-1", "the highest-seed row is first")
    For i = 1:1:(hits.%Size() - 1) {
        Do $$$AssertTrue(+hits.%Get(i - 1).score >= +hits.%Get(i).score, "scores do not rise down the list at hit "_i)
    }
}

/// The dimension of the search comes from the query vector, not from configuration. A query
/// too short to be a vector is not searchable and comes back empty rather than as an error.
Method TestVectorSearchDimensionDetection()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Do $$$AssertEquals(##class(RAG.SDK.Search).VectorSearch("1", 3), "[]", "a one-element query is not a vector")
    Set json = ##class(RAG.SDK.Search).VectorSearch(..QueryVec(..#VECDIM), 3)
    Do $$$AssertEquals($Extract(json, 1), "[", "a query of the column's width searches, got "_json)
    Set wrong = ##class(RAG.SDK.Search).VectorSearch(..QueryVec(..#VECDIM - 1), 3)
    Do $$$AssertEquals($Extract(wrong, 1), "{", "a query of the wrong width is a named error, got "_wrong)
}

Method TestVectorSearchTopKRespected()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Do $$$AssertEquals([].%FromJSON(##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 2)).%Size(), 2, "two asked for, two returned")
    Do $$$AssertEquals([].%FromJSON(##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 5)).%Size(), 5, "five asked for, five returned")
    // The fixture has 8 rows, so asking for more than the table holds is capped by the data.
    Do $$$AssertEquals([].%FromJSON(##class(RAG.SDK.Search).VectorSearch(..QueryVec(), 50)).%Size(), 8, "more asked for than exists returns what exists")
}

/// The unscoped text search 070 declared: a term that appears in one row returns that row.
Method TestTextSearch()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set json = ##class(RAG.SDK.Search).TextSearch("fifth", 5)
    Do $$$AssertEquals($Extract(json, 1), "[", "text search returns an array, got "_json)
    Set hits = [].%FromJSON(json)
    Do $$$AssertEquals(hits.%Size(), 1, "one row contains the term")
    Do $$$AssertEquals(hits.%Get(0).id, "a-5", "and it is the row that contains it")
    Do $$$AssertEquals(hits.%Get(0).text, "alpha fifth document", "carrying its text")
}

/// The two fusion strategies have to be distinguishable, which is what 070 asked for and
/// what a shared `FuseRRF`/`FuseLinear` bug would hide.
///
/// "fifth" appears in one row only, and that row is the farthest from the query, so the
/// vector leg does not reach it. RRF gives a first-place text hit the same weight as a
/// first-place vector hit, so the row surfaces; linear weights it 0.3 against cosine scores
/// near 1.0, so it does not.
Method TestHybridStrategiesDiffer()
{
    Do ..PointAt(..#FLOATTABLE, "attrs")
    Set rrf = [].%FromJSON(##class(RAG.SDK.Search).HybridSearch(..QueryVec(), "fifth", 3, "RRF"))
    Set lin = [].%FromJSON(##class(RAG.SDK.Search).HybridSearch(..QueryVec(), "fifth", 3, "linear"))
    Do $$$AssertTrue($IsObject(..FindHit(rrf, "a-5")), "RRF surfaces the text-only row")
    Do $$$AssertTrue('$IsObject(..FindHit(lin, "a-5")), "linear does not, at alpha 0.7")
    Do $$$AssertEquals(lin.%Get(0).id, "b-1", "linear leads with the nearest vector row")
}

/// With nothing configured the SDK falls back to the table it creates itself, and that table
/// is the only one it may assume has a metadata column.
Method TestDefaultTableFallback()
{
    Kill ^RAG.SDK.Config
    Set cfg = ##class(RAG.SDK.Search).GetTableConfig()
    Do $$$AssertEquals(cfg.%Get("table"), "RAG.SourceDocuments", "the built-in table is the fallback")
    Do $$$AssertEquals(cfg.%Get("id_col"), "doc_id", "with its own id column")
    Do $$$AssertEquals(cfg.%Get("text_col"), "text_content", "and text column")
    Do $$$AssertEquals(cfg.%Get("embedding_col"), "embedding", "and embedding column")
    Do $$$AssertEquals(cfg.%Get("metadata_col"), "metadata", "and its known metadata column")
    Set reported = {}.%FromJSON(##class(RAG.SDK.Bridge).GetDefaultTable())
    Do $$$AssertEquals(reported.table, "RAG.SourceDocuments", "and the bridge reports the same table")
    Do $$$AssertEquals(reported.%Get("metadata_col"), "metadata", "and the same metadata column")
}

/// AttachTable writes to Graph_KG.table_mappings when that table exists, so a test that
/// calls it removes its own row again.
Method CleanMapping(label As %String) [ Private ]
{
    If '##class(%SYSTEM.SQL.Schema).TableExists("Graph_KG.table_mappings") Return
    Set st = ##class(%SQL.Statement).%New()
    Set sc = st.%Prepare("DELETE FROM Graph_KG.table_mappings WHERE label = ?")
    If $$$ISERR(sc) Return
    Do st.%Execute(label)
}

/// The whole config subtree is stashed rather than a known list of keys: the SDK reads
/// ivf_index_name and bm25_index_name as well as the default_* keys, and a future key
/// would otherwise leak out of this suite silently. A process-private global is used
/// because the value has to outlive the method that saves it.
Method SaveConfig() [ Private ]
{
    Kill ^||RAGSDKConfigBackup
    Merge ^||RAGSDKConfigBackup = ^RAG.SDK.Config
}

Method RestoreConfig() [ Private ]
{
    Kill ^RAG.SDK.Config
    Merge ^RAG.SDK.Config = ^||RAGSDKConfigBackup
    Kill ^||RAGSDKConfigBackup
}

/// Point the SDK at a fixture table. metaCol = "" means "this table has no metadata column".
Method PointAt(table As %String, metaCol As %String = "") [ Private ]
{
    Set ^RAG.SDK.Config("default_table") = table
    Set ^RAG.SDK.Config("default_id_col") = "doc_id"
    Set ^RAG.SDK.Config("default_text_col") = "text_content"
    Set ^RAG.SDK.Config("default_emb_col") = "embedding"
    If metaCol = "" {
        Kill ^RAG.SDK.Config("default_meta_col")
    } Else {
        Set ^RAG.SDK.Config("default_meta_col") = metaCol
    }
}

/// A vector whose cosine similarity to QueryVec() rises with seed: component 1 is the
/// seed and the rest are 1, so similarity is seed / sqrt(seed*seed + dim - 1).
ClassMethod MakeVec(seed As %Integer, dim As %Integer = 8) As %String
{
    Set vec = seed
    For i = 2:1:dim { Set vec = vec_",1" }
    Return vec
}

/// The query every test searches with: (1, 0, 0, ...).
ClassMethod QueryVec(dim As %Integer = 8) As %String
{
    Set vec = 1
    For i = 2:1:dim { Set vec = vec_",0" }
    Return vec
}

/// Rows for both owners, ordered so that the unscoped nearest three all belong to agent-b
/// and agent-a still has more than three rows of its own. A scoped search that returns
/// agent-b rows, or fewer than three rows, therefore fails rather than passing by accident.
///
/// The last agent-a row carries unparseable metadata on purpose.
ClassMethod FixtureRows(ByRef rows) As %Integer
{
    Kill rows
    Set rows(1) = $ListBuild("b-1", "agent-b", "beta apex alpha document", 100, "{""owner"":""agent-b"",""rank"":1}")
    Set rows(2) = $ListBuild("b-2", "agent-b", "beta second alpha document", 90, "{""owner"":""agent-b"",""rank"":2}")
    Set rows(3) = $ListBuild("b-3", "agent-b", "beta third alpha document", 80, "{""owner"":""agent-b"",""rank"":3}")
    Set rows(4) = $ListBuild("a-1", "agent-a", "alpha first document", 50, "{""owner"":""agent-a"",""rank"":1}")
    Set rows(5) = $ListBuild("a-2", "agent-a", "alpha second document", 40, "{""owner"":""agent-a"",""rank"":2}")
    Set rows(6) = $ListBuild("a-3", "agent-a", "alpha third document", 30, "{""owner"":""agent-a"",""rank"":3}")
    Set rows(7) = $ListBuild("a-4", "agent-a", "alpha fourth document", 20, "{""owner"":""agent-a"",""rank"":4}")
    Set rows(8) = $ListBuild("a-5", "agent-a", "alpha fifth document", 10, "not json at all")
    Return 8
}

/// Create a fixture table with the given vector element type and insert every row through
/// TO_VECTOR, which is the only way a VECTOR column can be written from SQL.
Method MakeFixture(table As %String, vectorType As %String, withMetadata As %Boolean) [ Private ]
{
    Do ..DropFixture(table)
    Set ddl = "CREATE TABLE "_table_" (doc_id VARCHAR(64) NOT NULL PRIMARY KEY, "
        _"text_content VARCHAR(2000), embedding VECTOR("_vectorType_","_..#VECDIM_"), "
        _"agent_id VARCHAR(64)"
    If withMetadata Set ddl = ddl_", attrs VARCHAR(2000)"
    Set ddl = ddl_")"
    Set code = ..RunSQL(ddl)
    If code < 0 {
        Do $$$AssertFailure("could not create fixture "_table_": SQLCODE "_code)
        Return
    }

    Set cols = "doc_id, text_content, embedding, agent_id"
    Set vals = "?, ?, TO_VECTOR(?,"_vectorType_","_..#VECDIM_"), ?"
    If withMetadata {
        Set cols = cols_", attrs"
        Set vals = vals_", ?"
    }
    Set st = ##class(%SQL.Statement).%New()
    Set sc = st.%Prepare("INSERT INTO "_table_" ("_cols_") VALUES ("_vals_")")
    If $$$ISERR(sc) {
        Do $$$AssertFailure("could not prepare fixture insert: "_$SYSTEM.Status.GetErrorText(sc))
        Return
    }
    Set count = ..FixtureRows(.rows)
    For i = 1:1:count {
        Set row = rows(i)
        Set id = $List(row, 1), text = $List(row, 3)
        Set vec = ..MakeVec($List(row, 4), ..#VECDIM), owner = $List(row, 2)
        If withMetadata {
            Set rs = st.%Execute(id, text, vec, owner, $List(row, 5))
        } Else {
            Set rs = st.%Execute(id, text, vec, owner)
        }
        If rs.%SQLCODE '= 0 {
            Do $$$AssertFailure("fixture insert of "_id_" failed: "_rs.%Message)
            Return
        }
    }
}

Method DropFixture(table As %String) [ Private ]
{
    If ##class(%SYSTEM.SQL.Schema).TableExists(table) Do ..RunSQL("DROP TABLE "_table)
}

/// SQLCODE of a statement run for its effect, or -1 when it would not even prepare.
Method RunSQL(sql As %String) As %Integer [ Private ]
{
    Set st = ##class(%SQL.Statement).%New()
    Set sc = st.%Prepare(sql)
    If $$$ISERR(sc) Return -1
    Set rs = st.%Execute()
    Return rs.%SQLCODE
}

}
