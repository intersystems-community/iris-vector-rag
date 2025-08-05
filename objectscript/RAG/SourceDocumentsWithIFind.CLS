Class RAG.SourceDocumentsWithIFind Extends %Persistent
{
    /// Document identifier for HybridIFind pipeline
    Property doc_id As %String(MAXLEN=255);
    /// Document title
    Property title As %String(MAXLEN=500);
    /// Document content for iFind search - limited to 5000 chars for ObjectScript compilation
    Property text_content As %String(MAXLEN=5000);
    /// Document authors
    Property authors As %String(MAXLEN=1000);
    /// Document keywords
    Property keywords As %String(MAXLEN=1000);
}
