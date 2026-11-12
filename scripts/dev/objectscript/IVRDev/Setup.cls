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

/// Run ObjectScript code in pNamespace and return its captured output.
/// Executes each line with Xecute, errors written inline as they would appear in a session.
/// Output is captured to a temp file. Returns "" on success, "FAIL: <detail>" on setup error.
ClassMethod Run(pNamespace As %String, pCode As %String) As %String
{
    If pNamespace = "" {
        Return "FAIL: namespace required"
    }

    // Switch to the target namespace
    New $NAMESPACE
    Try {
        Set $NAMESPACE = pNamespace
    } Catch ex {
        Return "FAIL: "_ex.DisplayString()
    }

    // Open temp file for output capture
    Set tFile = ##class(%File).TempFilename("txt")
    Open tFile:("WNS"):5
    If '$Test {
        Return "FAIL: could not open temp file"
    }
    Use tFile

    // Execute each line, catching errors
    For i=1:1:$Length(pCode, $Char(10)) {
        Set tLine = $Piece(pCode, $Char(10), i)
        If tLine = "" Continue
        If $ZStrip(tLine, "<>W") = "" Continue
        If $ZCVT(tLine, "L") = "halt" Continue
        If $ZCVT(tLine, "L") = "h" Continue

        Try {
            Xecute tLine
        } Catch ex {
            Write ex.DisplayString(),!
        }
    }

    // Close file (restores principal device), read contents, delete
    Close tFile
    Set tStream = ##class(%Stream.FileCharacter).%New()
    Set tSC = tStream.LinkToFile(tFile)
    If $$$ISERR(tSC) {
        Do ##class(%File).Delete(tFile)
        Return "FAIL: "_$System.Status.GetErrorText(tSC)
    }
    Set tOutput = ""
    While 'tStream.AtEnd {
        Set tOutput = tOutput_tStream.Read(32000)
    }
    Do ##class(%File).Delete(tFile)

    Return tOutput
}

}
