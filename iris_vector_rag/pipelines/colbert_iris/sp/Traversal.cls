/// Pure ObjectScript BFS implementation for maximum performance and stability.
/// Spec 187: implementation split into focused SIBLING classes (TraversalBuild /
/// TraversalBFS / TraversalPaths / TraversalKHop). This class is a thin compatibility
/// facade that DELEGATES to them. Sibling (not nested) names + explicit delegators are
/// REQUIRED: the %SYS.DBSRV Native-API bridge does not resolve INHERITED ClassMethods
/// on this legacy class name (empirically confirmed — see specs/187). Delegators keep
/// every existing ##class(Graph.KG.Traversal).X() and classMethodValue caller working.
Class Graph.KG.Traversal Extends %RegisteredObject [ SqlSchemaName = Graph_KG ]
{

ClassMethod DijkstraProc(srcId As %String, dstId As %String, weightProp As %String = "", maxCost As %Double = 9999, maxHops As %Integer = 10, direction As %String = "out") As %String [ SqlProc, SqlName = DijkstraPath ]
{
	Quit ##class(Graph.KG.TraversalPaths).DijkstraJson(srcId, dstId, weightProp, maxCost, maxHops, direction)
}

ClassMethod BuildKG() As %Status
{
	Quit ##class(Graph.KG.TraversalBuild).BuildKG()
}

ClassMethod InitNKGSkeleton() As %Status
{
	Quit ##class(Graph.KG.TraversalBuild).InitNKGSkeleton()
}

ClassMethod NKGEdgeCount() As %Integer
{
	Quit ##class(Graph.KG.TraversalBuild).NKGEdgeCount()
}

ClassMethod NKGNodeCount() As %Integer
{
	Quit ##class(Graph.KG.TraversalBuild).NKGNodeCount()
}

ClassMethod BuildNKG() As %Status
{
	Quit ##class(Graph.KG.TraversalBuild).BuildNKG()
}

ClassMethod BackfillDegp() As %Integer
{
	Quit ##class(Graph.KG.TraversalBuild).BackfillDegp()
}

ClassMethod Build2HopStats() As %Integer
{
	Quit ##class(Graph.KG.TraversalBuild).Build2HopStats()
}

ClassMethod Build2HopExactStats() As %Integer
{
	Quit ##class(Graph.KG.TraversalBuild).Build2HopExactStats()
}

ClassMethod KGEdgeCount(maxCount As %Integer = 10000) As %Integer
{
	Quit ##class(Graph.KG.TraversalBuild).KGEdgeCount(maxCount)
}

ClassMethod NKGPopulated() As %Integer
{
	Quit ##class(Graph.KG.TraversalBuild).NKGPopulated()
}

ClassMethod BFS(srcId As %String, preds As %DynamicArray = "", maxHops As %Integer = 2, dstLabel As %String = "") As %DynamicArray
{
	Quit ##class(Graph.KG.TraversalBFS).BFS(srcId, preds, maxHops, dstLabel)
}

ClassMethod BFSFast(srcId As %String, preds As %DynamicArray = "", maxHops As %Integer = 2, dstLabel As %String = "", direction As %String = "out", maxResults As %Integer = 0) As %Integer
{
	Quit ##class(Graph.KG.TraversalBFS).BFSFast(srcId, preds, maxHops, dstLabel, direction, maxResults)
}

ClassMethod BFSJSON(srcId As %String, preds As %DynamicArray, maxHops As %Integer = 2, dstLabel As %String = "") As %DynamicArray
{
	Quit ##class(Graph.KG.TraversalBFS).BFSJSON(srcId, preds, maxHops, dstLabel)
}

ClassMethod BFSFastJson(srcId As %String, predsJson As %String = "", maxHops As %Integer = 2, dstLabel As %String = "", direction As %String = "out") As %String
{
	Quit ##class(Graph.KG.TraversalBFS).BFSFastJson(srcId, predsJson, maxHops, dstLabel, direction)
}

ClassMethod BFSFastJsonDirect(srcId As %String, predsJson As %String = "", maxHops As %Integer = 2, dstLabel As %String = "", direction As %String = "out") As %String
{
	Quit ##class(Graph.KG.TraversalBFS).BFSFastJsonDirect(srcId, predsJson, maxHops, dstLabel, direction)
}

ClassMethod BFSFastJsonChunked(srcId As %String, predsJson As %String = "", maxHops As %Integer = 2, dstLabel As %String = "", direction As %String = "out", maxResults As %Integer = 0) As %String
{
	Quit ##class(Graph.KG.TraversalBFS).BFSFastJsonChunked(srcId, predsJson, maxHops, dstLabel, direction, maxResults)
}

ClassMethod BFSFastJsonSorted(srcId As %String, predsJson As %String = "", maxHops As %Integer = 2, dstLabel As %String = "", direction As %String = "out", maxResults As %Integer = 0) As %String
{
	Quit ##class(Graph.KG.TraversalBFS).BFSFastJsonSorted(srcId, predsJson, maxHops, dstLabel, direction, maxResults)
}

ClassMethod BFSFastCountDistinct(srcId As %String, predsJson As %String = "", maxHops As %Integer = 2, dstLabel As %String = "", direction As %String = "out") As %Integer
{
	Quit ##class(Graph.KG.TraversalBFS).BFSFastCountDistinct(srcId, predsJson, maxHops, dstLabel, direction)
}

ClassMethod ReadBFSChunk(tag As %String, chunkNum As %Integer) As %String
{
	Quit ##class(Graph.KG.TraversalBFS).ReadBFSChunk(tag, chunkNum)
}

ClassMethod ReadBFSResults(tag As %String) As %String
{
	Quit ##class(Graph.KG.TraversalBFS).ReadBFSResults(tag)
}

ClassMethod ReadBFSPage(tag As %String, cursorStep As %String, cursorO As %String, pageSize As %Integer = 500) As %String
{
	Quit ##class(Graph.KG.TraversalBFS).ReadBFSPage(tag, cursorStep, cursorO, pageSize)
}

ClassMethod ShortestPathJson(srcId As %String, dstId As %String, maxHops As %Integer = 5, predsJson As %String = "", direction As %String = "both", findAll As %Integer = 0) As %String
{
	Quit ##class(Graph.KG.TraversalPaths).ShortestPathJson(srcId, dstId, maxHops, predsJson, direction, findAll)
}

ClassMethod DijkstraJson(srcId As %String, dstId As %String, weightProp As %String = "", maxCost As %Double = 9999, maxHops As %Integer = 10, direction As %String = "out") As %String
{
	Quit ##class(Graph.KG.TraversalPaths).DijkstraJson(srcId, dstId, weightProp, maxCost, maxHops, direction)
}

ClassMethod KHopCount(srcId As %String, pred As %String = "") As %Integer
{
	Quit ##class(Graph.KG.TraversalKHop).KHopCount(srcId, pred)
}

ClassMethod KHopNeighborIds(srcId As %String, pred As %String = "") As %String
{
	Quit ##class(Graph.KG.TraversalKHop).KHopNeighborIds(srcId, pred)
}

ClassMethod KHop2Count(srcId As %String, pred As %String = "") As %Integer
{
	Quit ##class(Graph.KG.TraversalKHop).KHop2Count(srcId, pred)
}

ClassMethod KHop2NeighborIds(srcId As %String, pred As %String = "", maxResults As %Integer = 0) As %String
{
	Quit ##class(Graph.KG.TraversalKHop).KHop2NeighborIds(srcId, pred, maxResults)
}

ClassMethod KHop2CountFast(srcId As %String, pred As %String = "") As %Integer
{
	Quit ##class(Graph.KG.TraversalKHop).KHop2CountFast(srcId, pred)
}

ClassMethod KHop2CountExact(srcId As %String, pred As %String = "") As %Integer
{
	Quit ##class(Graph.KG.TraversalKHop).KHop2CountExact(srcId, pred)
}
}
