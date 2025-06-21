Class RAG.SourceDocumentsWithIFind Extends %Persistent [ ClassType = persistent, DdlAllowed, SqlTableName = SourceDocumentsIFind ]
{

Property doc_id As %Library.String(MAXLEN = 255) [ Required ]

Property title As %Library.String(MAXLEN = 1000)

Property text_content As %Stream.GlobalCharacter // iFind can index %Stream.GlobalCharacter directly

Property embedding As %Library.String(MAXLEN = 0) // Unlimited length for embedding string

Property created_at As %Library.TimeStamp

Index DocIdIndex On doc_id [ Unique ]

Index TextContentFTI On (text_content) As %iFind.Index.Basic // Simple iFind index

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
<DataLocation>^RAG.SourceDocsIFindD</DataLocation> // Adjusted global name
<DefaultData>SourceDocumentsIFindDefaultData</DefaultData>
<IdLocation>^RAG.SourceDocsIFindD</IdLocation> // Adjusted global name
<IndexLocation>^RAG.SourceDocsIFindI</IndexLocation> // Adjusted global name
<StreamLocation>^RAG.SourceDocsIFindS</StreamLocation> // Adjusted global name
<Type>%Storage.Persistent</Type>
}

}