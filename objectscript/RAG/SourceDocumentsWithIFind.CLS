Class RAG.SourceDocumentsWithIFind Extends %Persistent [ SqlTableName = SourceDocumentsIFind ]
{
    Property doc_id As %String(MAXLEN = 255) [ Required ];
    Property title As %String(MAXLEN = 1000);
    Property text_content As %Stream.GlobalCharacter;
    Property embedding As %String(MAXLEN = "");
    Property created_at As %TimeStamp;
    
    Index DocIdIndex On doc_id [ Unique ];
    
    // This is the key - %iFind.Index.Basic on the STREAM field
    Index TextContentFTI On (text_content) As %iFind.Index.Basic(
        LANGUAGE = "en",
        LOWER = 1,
        INDEXOPTION = 2  // Enable stemming and decompounding
    );
    
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
        <DataLocation>^RAG.SourceDocumentsIFindD</DataLocation>
        <DefaultData>SourceDocumentsIFindDefaultData</DefaultData>
        <IdLocation>^RAG.SourceDocumentsIFindD</IdLocation>
        <IndexLocation>^RAG.SourceDocumentsIFindI</IndexLocation>
        <StreamLocation>^RAG.SourceDocumentsIFindS</StreamLocation>
        <Type>%Storage.Persistent</Type>
    }
}