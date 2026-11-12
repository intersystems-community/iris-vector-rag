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
/// Returns: "OK^<output>" or "FAIL^<error>". Output is what the lines wrote (including errors).
/// Output is captured by redirecting the current device to a global, then reading it back.
ClassMethod RunLines(pLines As %String, pNamespace As %String) As %String
{
    If pNamespace = "" {
        Return "FAIL^namespace required"
    }

    // Switch to the target namespace
    New $NAMESPACE
    Set tOldNS = $NAMESPACE
    Try {
        Set $NAMESPACE = pNamespace
    } Catch ex {
        Return "FAIL^"_ex.DisplayString()
    }

    // Capture output by redirecting IO
    Set tOutputGlobal = "^IVRDev.RunOutput."_$JOB
    Kill @tOutputGlobal
    Set tOldIO = $IO
    Open tOutputGlobal:("":0) Use tOutputGlobal

    // Execute each line
    Set tHadError = 0
    For i=1:1:$Length(pLines, $Char(10)) {
        Set tLine = $Piece(pLines, $Char(10), i)
        If tLine = "" Continue
        If $ZStrip(tLine, "<>W") = "" Continue
        If $ZCVT(tLine, "L") = "halt" Continue

        Try {
            Xecute tLine
        } Catch ex {
            Write "ERROR line "_i_": "_ex.DisplayString(),!
            Set tHadError = 1
        }
    }

    // Restore IO and collect output
    Use tOldIO
    Close tOutputGlobal
    Set tOutput = ""
    Set tKey = $Order(@tOutputGlobal@(""))
    While tKey '= "" {
        Set tOutput = tOutput_@tOutputGlobal@(tKey)
        Set tKey = $Order(@tOutputGlobal@(tKey))
    }
    Kill @tOutputGlobal

    Return "OK^"_tOutput
}

}
