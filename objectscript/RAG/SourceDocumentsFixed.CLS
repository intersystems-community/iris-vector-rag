/// RAG SourceDocuments table with iFind support - FIXED VERSION
/// This class provides the main SourceDocuments table structure that supports iFind functionality
/// Key change: Uses %String instead of %Stream.GlobalCharacter to avoid stream field limitations
Class RAG.SourceDocumentsFixed Extends %Persistent
{

/// Document identifier
Property doc_id As %String(MAXLEN = 255) [ Required ];

/// Document title
Property title As %String(MAXLEN = 1000);

/// Full text content - Using %Stream for proper IRIS handling
Property text_content As %Stream.GlobalCharacter;

/// Document abstract - Using %String instead of %Stream
Property abstract As %String(MAXLEN = 4000);

/// Document authors
Property authors As %String(MAXLEN = 1000);

/// Document keywords  
Property keywords As %String(MAXLEN = 1000);

/// Document metadata as JSON string
Property metadata As %String(MAXLEN = 4000);

/// Vector embedding as stream for large embeddings
Property embedding As %Stream.GlobalCharacter;

/// Creation timestamp
Property created_at As %TimeStamp [ InitialExpression = {$ZDateTime($Horolog,3)} ];

/// Index on document ID for fast lookups
Index DocIdIndex On doc_id [ Unique ];

/// Index on creation timestamp for temporal queries
Index CreatedAtIndex On created_at;

/// Index on title for text search
Index TitleIndex On title;

/// Storage definition for the class
Storage Default
{
<Data name="SourceDocumentsFixedDefaultData">
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
<Value>abstract</Value>
</Value>
<Value name="6">
<Value>authors</Value>
</Value>
<Value name="7">
<Value>keywords</Value>
</Value>
<Value name="8">
<Value>metadata</Value>
</Value>
<Value name="9">
<Value>embedding</Value>
</Value>
<Value name="10">
<Value>created_at</Value>
</Value>
</Data>
<DataLocation>^RAG.SourceDocumentsFixedD</DataLocation>
<DefaultData>SourceDocumentsFixedDefaultData</DefaultData>
<IdLocation>^RAG.SourceDocumentsFixedD</IdLocation>
<IndexLocation>^RAG.SourceDocumentsFixedI</IndexLocation>
<StreamLocation>^RAG.SourceDocumentsFixedS</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}
