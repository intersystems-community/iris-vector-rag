Class RAG.TestMultipleMAXLEN Extends %Persistent
{
    /// Test if multiple MAXLEN properties work like TestMAXLENVariants
    Property prop1 As %String(MAXLEN=255);
    Property prop2 As %String(MAXLEN=500);
    Property prop3 As %String(MAXLEN=1000);
}
