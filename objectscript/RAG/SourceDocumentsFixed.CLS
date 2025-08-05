Class RAG.SourceDocumentsFixed Extends %Persistent
{
    /// Document identifier
    Property doc_id As %String(MAXLEN=255);
    /// Document title
    Property title As %String(MAXLEN=500);
    /// Full text content - limited to 5000 chars for ObjectScript compilation
    Property text_content As %String(MAXLEN=5000);
    /// Document authors
    Property authors As %String(MAXLEN=1000);
    /// Document keywords
    Property keywords As %String(MAXLEN=1000);
    /// Abstract/summary
    Property abstract As %String(MAXLEN=2000);
}
