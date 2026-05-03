Class Graph.KG.Snapshot Extends %RegisteredObject
{

ClassMethod ExportGlobals(outputDir As %String, globalListJson As %String) As %String
{
	Do ##class(%Library.File).CreateDirectoryChain(outputDir)
	Set globalList = ##class(%DynamicArray).%FromJSON(globalListJson)
	Set exportedFiles = ##class(%DynamicArray).%New()
	Set gCount = globalList.%Size()
	For i = 0:1:(gCount-1) {
		Set entry = globalList.%Get(i)
		Set globalName = entry.%Get("name")
		If globalName = "" Continue
		Set exportFile = outputDir _ globalName _ ".gof"
		Set sc = ##class(%GlobalEdit).Export("^" _ globalName, exportFile)
		If $$$ISOK(sc) {
			Do exportedFiles.%Push(exportFile)
		}
	}
	Quit exportedFiles.%ToJSON()
}

ClassMethod ImportGlobals(inputDir As %String) As %String
{
	Set imported = 0
	Set rs = ##class(%ResultSet).%New("%Library.File:FileSet")
	Do rs.Execute(inputDir, "*.gof")
	While rs.Next() {
		Set fileName = rs.Get("Name")
		Set sc = ##class(%GlobalEdit).Import(fileName)
		If $$$ISOK(sc) Set imported = imported + 1
	}
	Quit "{""imported"":"_imported_"}"
}

ClassMethod ReadFileChunk(filePath As %String, offset As %Integer = 0, chunkSize As %Integer = 1048576) As %String
{
	Set stream = ##class(%Stream.FileCharacter).%New()
	Do stream.LinkToFile(filePath)
	Do stream.Rewind()
	If offset > 0 {
		Do stream.Read(offset)
	}
	Set chunk = stream.Read(chunkSize)
	Quit chunk
}

ClassMethod WriteFileChunk(filePath As %String, content As %String, append As %Integer = 0)
{
	Set stream = ##class(%Stream.FileCharacter).%New()
	Do stream.LinkToFile(filePath)
	If 'append Do stream.Rewind()
	If append Do stream.MoveToEnd()
	Do stream.Write(content)
	Do stream.%Save()
}

ClassMethod GetFileSize(filePath As %String) As %Integer
{
	Quit ##class(%Library.File).GetFileSize(filePath)
}

ClassMethod DeleteDir(dirPath As %String) As %Integer
{
	Set sc = ##class(%Library.File).RemoveDirectoryTree(dirPath)
	Quit +sc
}

ClassMethod FileExists(filePath As %String) As %Integer
{
	Quit ##class(%Library.File).Exists(filePath)
}

}
