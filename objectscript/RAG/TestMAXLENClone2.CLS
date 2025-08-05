Class RAG.TestMAXLENClone2 Extends %Persistent
{
    /// Test different MAXLEN syntax variants (clone of working TestMAXLENVariants)
    Property test1 As %String(MAXLEN=255);
    Property test2 As %String(MAXLEN = "255");
    Property test3 As %String(MAXLEN="255");
}
