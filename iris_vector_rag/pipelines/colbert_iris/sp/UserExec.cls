/// User.Exec — Minimal ObjectScript execution helper for Python SDK.
/// Allows Python to run arbitrary ObjectScript via classMethodVoid.
Class User.Exec Extends %RegisteredObject
{

ClassMethod Run(cmd As %String) As %String
{
    Xecute cmd
    Return ""
}

}
