/// Lightweight key/value store backed by a global for tracking bootstrap state.
/// Used by initialize_schema() to record one-time setup steps (e.g. BuildKG completion).
Class Graph.KG.Meta Extends %RegisteredObject [ SqlSchemaName = Graph_KG ]
{

/// Get a metadata value by key.
/// @param key  Metadata key (e.g. "kg_built")
/// @returns    Value string, or "" if not set
ClassMethod Get(key As %String) As %String
{
    Return $Get(^Graph.KG.Meta(key), "")
}

/// Set a metadata value.
/// @param key    Metadata key
/// @param value  Value to store
/// @returns $$$OK
ClassMethod Set(key As %String, value As %String) As %Status
{
    Set ^Graph.KG.Meta(key) = value
    Quit $$$OK
}

/// Delete a metadata key.
ClassMethod Delete(key As %String) As %Status
{
    Kill ^Graph.KG.Meta(key)
    Quit $$$OK
}

/// Check if a key exists and has a truthy value.
ClassMethod IsSet(key As %String) As %Boolean
{
    Set val = $Get(^Graph.KG.Meta(key), "")
    Return (val '= "") && (val '= "0")
}

/// Clear all metadata (for testing/reset).
ClassMethod PurgeAll() As %Status
{
    Kill ^Graph.KG.Meta
    Quit $$$OK
}

ClassMethod GetNKG(s1 As %String = "", s2 As %String = "", s3 As %String = "", s4 As %String = "") As %String
{
	If s4 '= "" Return $Get(^NKG(s1, s2, s3, s4), "")
	If s3 '= "" Return $Get(^NKG(s1, s2, s3), "")
	If s2 '= "" Return $Get(^NKG(s1, s2), "")
	If s1 '= "" Return $Get(^NKG(s1), "")
	Return ""
}

ClassMethod GetKG(s1 As %String = "", s2 As %String = "", s3 As %String = "", s4 As %String = "", s5 As %String = "", s6 As %String = "") As %String
{
	If s6 '= "" Return $Get(^KG(s1, s2, s3, s4, s5, s6), "")
	If s5 '= "" Return $Get(^KG(s1, s2, s3, s4, s5), "")
	If s4 '= "" Return $Get(^KG(s1, s2, s3, s4), "")
	If s3 '= "" Return $Get(^KG(s1, s2, s3), "")
	If s2 '= "" Return $Get(^KG(s1, s2), "")
	If s1 '= "" Return $Get(^KG(s1), "")
	Return ""
}

ClassMethod IVG_PDISC(vals As %String, p As %Double) As %String [ SqlProc ]
{
	Set arr = ##class(%Library.DynamicArray).%FromJSON(vals)
	Set n = arr.%Size()
	If n = 0 { Quit "" }
	Set i = n * p \ 1
	If i >= n { Set i = n - 1 }
	Quit arr.%Get(i)
}

ClassMethod IVG_PCONT(vals As %String, p As %Double) As %Double [ SqlProc ]
{
	Set arr = ##class(%Library.DynamicArray).%FromJSON(vals)
	Set n = arr.%Size()
	If n = 0 { Quit 0 }
	Set lower = n * p \ 1
	If lower >= n - 1 { Quit arr.%Get(n-1) + 0 }
	Set frac = (n * p) - lower
	Quit arr.%Get(lower) + frac * (arr.%Get(lower+1) - arr.%Get(lower))
}

}
