Class RAG.SourceDocumentsWithIFind Extends %Persistent [ SqlTableName = SourceDocumentsIFind ]
{

Property doc_id As %String(MAXLEN = 255) [ Required ]

Property title As %String(MAXLEN = 1000)

Property text_content As %Stream.GlobalCharacter

Property embedding As %String(MAXLEN = "")

Property created_at As %TimeStamp

Index DocIdIndex On doc_id [ Unique ]

Index TextContentFTI On (text_content) As %iFind.Index.Basic

}