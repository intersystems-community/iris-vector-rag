Class RAG.TestMAXLENVariants Extends %Persistent
{
    /// Test different MAXLEN syntax variants
    Property test1 As %String(MAXLEN=255);
    Property test2 As %String(MAXLEN = "255");
    Property test3 As %String(MAXLEN="255");
}
