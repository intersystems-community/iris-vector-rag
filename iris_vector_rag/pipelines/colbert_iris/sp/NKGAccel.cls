/// Graph.KG.NKGAccel - NICHE Knowledge Graph accelerated analytics.
/// Seed-local algorithms run in ObjectScript at $ORDER speed.
/// Full-graph algorithms export adjacency for Rust consumption.
///
/// Auto-detects ^NKG (integer-encoded) vs ^KG (string-subscripted).
/// NKGAccel adjacency-index facade (spec 187). Delegates to sibling classes.
/// Sibling (not nested) names required — inherited ClassMethods don't resolve
/// via %SYS.DBSRV native bridge when class name == package prefix.
Class Graph.KG.NKGAccel Extends (%RegisteredObject, Graph.KG.NKGAccelLoader, Graph.KG.NKGAccelAdjacency, Graph.KG.NKGAccelTraversal, Graph.KG.NKGAccelCentrality)
{

/// Detect whether ^NKG is available. Cache per-process.

/// Clear cached detection (call after BuildKG or schema change)

/// Load arno callout library and register all function IDs for this process.
/// Call once at startup: Do ##class(Graph.KG.NKGAccel).Load("/path/to/libarno_callout.so")
/// Returns 1 on success, 0 if library not found or load failed.
/// Safe to call multiple times - no-ops if already loaded in this process.


/// Returns 1 if arno is loaded and bfs function is available in this process.

// ============================================================
// T015: ExportAdjacency - full graph for Rust consumption
// ============================================================

/// Export full graph adjacency as integer-indexed lines.
/// Format: srcIdx:dstIdx1,dstIdx2,...\n
/// Returns string directly if < 3MB, otherwise chunks to ^||adj.

/// Export from ^NKG (integer subscripts - zero string overhead)

/// Export from ^KG (string subscripts - builds node index on the fly)

// ============================================================
// T019: KHopNeighbors - seed-local BFS at $ORDER speed
// ============================================================

/// k-hop BFS from a seed node. Returns JSON with visited nodes and distances.
/// Runs entirely in ObjectScript - no Rust, no graph serialization.
/// Cost: ~0.5ms for 2-hop fanout-16 regardless of graph size.



/// Personalized PageRank from seed node(s). Runs in ObjectScript.
/// Returns top-K nodes with scores as JSON.

// ============================================================
// Wrappers that replace CacheGraphJson path for seed-local
// ============================================================

/// K-hop neighborhood - pure ObjectScript, no Rust needed

/// Personalized PageRank - pure ObjectScript on local subgraph

/// Random walk - pure ObjectScript















/// Spec 168 (FR-168-001..007): Closeness centrality via ^NKG BFS.
/// Runs BFS from every node server-side; computes harmonic or classical closeness.
/// Returns "OK:[{""id"":""..."",""score"":0.5}]" or "ERROR:<msg>".

/// Spec 169: Eigenvector centrality via ^NKG power iteration (FR-169-001).
/// L2-normalized power iteration over raw adjacency A (NOT transition matrix D^-1*A).
/// Returns "OK:[{""id"":""..."",""score"":0.5}]" or "ERROR:<msg>".

/// Spec 170: Betweenness centrality via ^NKG Brandes 2001 (FR-170-001).
/// Sampled Brandes betweenness: capped at maxSources (default 200) to bound runtime.
/// For graphs with n <= maxSources, runs exact full Brandes. For larger graphs,
/// samples maxSources random-ish sources and scales scores by n/maxSources.
/// Returns "OK:[{""id"":""..."",""score"":0.5}]" or "ERROR:<msg>".

/// Spec 172: Parallel Brandes worker — called by %SYSTEM.WorkMgr via BetweennessGlobalParallel.
/// Runs Brandes over sources[srcFirst..srcLast] from ^bcParSrc(parentPid, i),
/// writes partial scores to ^bcParScore(parentPid, workerN, nodeIdx).

/// Spec 172: Parallel Brandes dispatcher using %SYSTEM.WorkMgr.
/// Splits sources across numJobs workers, each writing partial scores to
/// ^bcParScore(parentPid, workerN, nodeIdx). Waits via Sync(), then merges.
/// Returns "OK:[{""id"":""..."",""score"":0.5}]" or "ERROR:<msg>".


/// Spec 173: Betweenness centrality on a neighborhood subgraph.
/// Extracts the hops-hop neighborhood around seedId, then runs Brandes
/// restricted to that subgraph. Scales to any total graph size — performance
/// depends only on neighborhood size (~500-5K nodes for biomedical KGs).
/// Returns "OK:[{""id"":""..."",""score"":0.5}]" or "ERROR:<msg>".

/// OS fallback for BetweennessNeighborhood: $BITLOGIC parallel Brandes
/// restricted to ^||nbhNodes(0..n-1). Caller must populate ^||nbhNodes.

ClassMethod UseNKG() As %Boolean
{
	Quit ##class(Graph.KG.NKGAccelLoader).UseNKG()
}
ClassMethod ResetCache()
{
	Do ##class(Graph.KG.NKGAccelLoader).ResetCache()
}
ClassMethod Load(libPath As %String = "/usr/irissys/mgr/libarno_callout.so") As %Boolean
{
	Quit ##class(Graph.KG.NKGAccelLoader).Load(libPath)
}
ClassMethod Unload()
{
	Do ##class(Graph.KG.NKGAccelLoader).Unload()
}
ClassMethod IsLoaded() As %Boolean
{
	Quit ##class(Graph.KG.NKGAccelLoader).IsLoaded()
}
ClassMethod InvalidateAdjCache()
{
	Do ##class(Graph.KG.NKGAccelLoader).InvalidateAdjCache()
}
ClassMethod WarmAdjCache() As %Integer
{
	Quit ##class(Graph.KG.NKGAccelLoader).WarmAdjCache()
}
ClassMethod Capabilities() As %String
{
	Quit ##class(Graph.KG.NKGAccelLoader).Capabilities()
}
ClassMethod BuildNKGRust() As %String
{
	Quit ##class(Graph.KG.NKGAccelLoader).BuildNKGRust()
}
ClassMethod GetFirstNKGNode() As %String
{
	Quit ##class(Graph.KG.NKGAccelLoader).GetFirstNKGNode()
}
ClassMethod ExportAdjacency() As %String
{
	Quit ##class(Graph.KG.NKGAccelAdjacency).ExportAdjacency()
}
ClassMethod ReadLargeOutChunk(tag As %String, chunkNum As %Integer) As %String
{
	Quit ##class(Graph.KG.NKGAccelAdjacency).ReadLargeOutChunk(tag, chunkNum)
}
ClassMethod ReadKHopResults(tag As %String) As %String
{
	Quit ##class(Graph.KG.NKGAccelAdjacency).ReadKHopResults(tag)
}
ClassMethod KHopNeighbors(seedId As %String, k As %Integer = 2, maxNodes As %Integer = 1000) As %String
{
	Quit ##class(Graph.KG.NKGAccelTraversal).KHopNeighbors(seedId, k, maxNodes)
}
ClassMethod KHopNeighborsSorted(seedId As %String, k As %Integer = 2, maxNodes As %Integer = 1000) As %String
{
	Quit ##class(Graph.KG.NKGAccelTraversal).KHopNeighborsSorted(seedId, k, maxNodes)
}
ClassMethod KHopJson(seedId As %String, k As %Integer = 2, maxNodes As %Integer = 1000) As %String
{
	Quit ##class(Graph.KG.NKGAccelTraversal).KHopJson(seedId, k, maxNodes)
}
ClassMethod PPRNative(seedId As %String, alpha As %Double = 0.85, maxIter As %Integer = 20, topK As %Integer = 20) As %String
{
	Quit ##class(Graph.KG.NKGAccelTraversal).PPRNative(seedId, alpha, maxIter, topK)
}
ClassMethod PPRJson(seedId As %String, alpha As %Double = 0.85, maxIter As %Integer = 20) As %String
{
	Quit ##class(Graph.KG.NKGAccelTraversal).PPRJson(seedId, alpha, maxIter)
}
ClassMethod RandomWalkJson(seedId As %String, walkLength As %Integer = 20, numWalks As %Integer = 10) As %String
{
	Quit ##class(Graph.KG.NKGAccelTraversal).RandomWalkJson(seedId, walkLength, numWalks)
}
ClassMethod BFSJson(seed As %String, predicatesJson As %String = "[]", maxHops As %Integer = 3, maxResults As %Integer = 0) As %String
{
	Quit ##class(Graph.KG.NKGAccelTraversal).BFSJson(seed, predicatesJson, maxHops, maxResults)
}
ClassMethod ShortestPathNKG(srcId As %String, dstId As %String, maxHops As %Integer = 10) As %String
{
	Quit ##class(Graph.KG.NKGAccelTraversal).ShortestPathNKG(srcId, dstId, maxHops)
}
ClassMethod CountDistinctKHop(srcId As %String, predsJson As %String = "", maxHops As %Integer = 2, direction As %String = "both") As %String
{
	Quit ##class(Graph.KG.NKGAccelTraversal).CountDistinctKHop(srcId, predsJson, maxHops, direction)
}
ClassMethod Build2HopExact() As %Boolean
{
	Quit ##class(Graph.KG.NKGAccelTraversal).Build2HopExact()
}
ClassMethod ClosenessGlobal(formula As %String = "harmonic", direction As %String = "out", maxHops As %Integer = 0, topK As %Integer = 10000) As %String
{
	Quit ##class(Graph.KG.NKGAccelCentrality).ClosenessGlobal(formula, direction, maxHops, topK)
}
ClassMethod ClosenessGlobalMSBFS(formula As %String = "harmonic", direction As %String = "out", maxHops As %Integer = 0, topK As %Integer = 10000) As %String
{
	Quit ##class(Graph.KG.NKGAccelCentrality).ClosenessGlobalMSBFS(formula, direction, maxHops, topK)
}
ClassMethod EigenvectorGlobal(maxIter As %Integer = 30, tol As %Float = 0.0001, topK As %Integer = 10000) As %String
{
	Quit ##class(Graph.KG.NKGAccelCentrality).EigenvectorGlobal(maxIter, tol, topK)
}
ClassMethod BetweennessGlobal(sampleSize As %Integer = 0, topK As %Integer = 10000, maxSources As %Integer = 200) As %String
{
	Quit ##class(Graph.KG.NKGAccelCentrality).BetweennessGlobal(sampleSize, topK, maxSources)
}
ClassMethod BetweennessWorker(parentPid As %Integer, workerN As %Integer, srcFirst As %Integer, srcLast As %Integer, scale As %Double) As %Status
{
	Quit ##class(Graph.KG.NKGAccelCentrality).BetweennessWorker(parentPid, workerN, srcFirst, srcLast, scale)
}
ClassMethod BetweennessGlobalParallel(sampleSize As %Integer = 0, topK As %Integer = 10000, maxSources As %Integer = 200, numJobs As %Integer = 0) As %String
{
	Quit ##class(Graph.KG.NKGAccelCentrality).BetweennessGlobalParallel(sampleSize, topK, maxSources, numJobs)
}
ClassMethod BetweennessNeighborhood(seedId As %String, hops As %Integer = 2, sampleSize As %Integer = 200, topK As %Integer = 20) As %String
{
	Quit ##class(Graph.KG.NKGAccelCentrality).BetweennessNeighborhood(seedId, hops, sampleSize, topK)
}

}
