/// Setup helper for iris-vector-rag: unexpire passwords over TCP with no `iris session`.
/// Loaded into %SYS by scripts/dev/tcp_setup.py; never shipped.
/// Cortex XDR kills processes that pipe ObjectScript into `docker exec iris session`,
/// so password unexpiry runs through the Native API and this helper class instead.
Class IVRDev.Setup [ Abstract ]
{

/// Replaced by the source's hash when tcp_setup.py loads the class; it reloads when the hash differs.
Parameter SOURCEHASH = "";

ClassMethod SourceHash() As %String
{
    Return ..#SOURCEHASH
}

/// Unexpire all user passwords matching pPattern (default "*" = all users).
ClassMethod UnExpireUserPasswords(pPattern As %String = "*") As %String
{
    Set tSC = ##class(Security.Users).UnExpireUserPasswords(pPattern)
    If $$$ISERR(tSC) {
        Return "FAIL unexpire: "_$System.Status.GetErrorText(tSC)
    }
    Return ""
}

/// Run ObjectScript lines in pNamespace: each line with Xecute, errors written inline.
/// Returns: "" on success, "FAIL <step>: <detail>" on setup error.
/// The output (what the lines wrote, with errors) is written to pOutputGlobal as a stream.
ClassMethod RunLines(pLines As %String, pNamespace As %String, pOutputGlobal As %String) As %String
{
    If pNamespace = "" {
        Return "FAIL: namespace required"
    }

    // Create output stream
    Set tStream = ##class(%Stream.GlobalCharacter).%New()
    Set tStream.Name = pOutputGlobal

    // Try to switch to the namespace
    Try {
        New $NAMESPACE
        Set $NAMESPACE = pNamespace
    } Catch ex {
        Return "FAIL ns: "_ex.DisplayString()
    }

    // Execute each line
    For i=1:1:$Length(pLines, $Char(10)) {
        Set tLine = $Piece(pLines, $Char(10), i)
        If tLine = "" Continue
        If $ZStrip(tLine, "<>W") = "" Continue

        Try {
            Xecute tLine
        } Catch ex {
            Do tStream.WriteLine("ERROR line "_i_": "_ex.DisplayString())
        }
    }

    Do tStream.%Save()
    Return ""
}

}
