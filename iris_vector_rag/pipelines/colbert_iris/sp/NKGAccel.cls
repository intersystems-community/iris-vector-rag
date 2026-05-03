/// Graph.KG.NKGAccel — NICHE Knowledge Graph accelerated analytics.
/// Seed-local algorithms run in ObjectScript at $ORDER speed.
/// Full-graph algorithms export adjacency for Rust consumption.
///
/// Auto-detects ^NKG (integer-encoded) vs ^KG (string-subscripted).
Class Graph.KG.NKGAccel Extends %RegisteredObject
{

/// Detect whether ^NKG is available. Cache per-process.
ClassMethod UseNKG() As %Boolean
{
    If $Data(^||NKGAccel("useNKG")) Return ^||NKGAccel("useNKG")
    Set use = ($Data(^NKG("$meta")) > 0)
    Set ^||NKGAccel("useNKG") = use
    Return use
}

/// Clear cached detection (call after BuildKG or schema change)
ClassMethod ResetCache()
{
    Kill ^||NKGAccel
}

/// Load arno callout library and register all function IDs for this process.
/// Call once at startup: Do ##class(Graph.KG.NKGAccel).Load("/path/to/libarno_callout.so")
/// Returns 1 on success, 0 if library not found or load failed.
/// Safe to call multiple times — no-ops if already loaded in this process.
ClassMethod Load(libPath As %String = "/usr/irissys/mgr/libarno_callout.so") As %Boolean
{
    If $Data(^||NKGAccel("dllid")) Return 1
    Set dllid = $ZF(-4, 1, libPath)
    If dllid = 0 Return 0
    Set ^||NKGAccel("dllid") = dllid
    Set ^||NKGAccel("libPath") = libPath
    Set ^||NKGAccel("bfs_fn") = $ZF(-4, 3, dllid, "kg_bfs_global")
    Return 1
}

/// Returns 1 if arno is loaded and bfs function is available in this process.
ClassMethod IsLoaded() As %Boolean
{
    If '$Data(^||NKGAccel("dllid")) Return 0
    Return (^||NKGAccel("bfs_fn") > 0)
}

// ============================================================
// T015: ExportAdjacency — full graph for Rust consumption
// ============================================================

/// Export full graph adjacency as integer-indexed lines.
/// Format: srcIdx:dstIdx1,dstIdx2,...\n
/// Returns string directly if < 3MB, otherwise chunks to ^||adj.
ClassMethod ExportAdjacency() As %String
{
    If ..UseNKG() Return ..ExportAdjacencyNKG()
    Return ..ExportAdjacencyKG()
}

/// Export from ^NKG (integer subscripts — zero string overhead)
ClassMethod ExportAdjacencyNKG() As %String [ Private ]
{
    Set result = "", chunk = "", chunkNum = 0
    Set src = ""
    For {
        Set src = $Order(^NKG(-3, src))
        Quit:src=""
        Set nbrs = "", pred = ""
        For {
            Set pred = $Order(^NKG(-1, src, pred))
            Quit:pred=""
            Set dst = ""
            For {
                Set dst = $Order(^NKG(-1, src, pred, dst))
                Quit:dst=""
                Set:nbrs'="" nbrs = nbrs _ ","
                Set nbrs = nbrs _ dst
            }
        }
        Set line = src _ ":" _ nbrs _ $Char(10)
        If $Length(result) + $Length(line) > 3000000 {
            // Switch to chunked mode
            If chunkNum = 0 {
                Set chunkNum = 1
                Set ^||adj(1) = result
                Set result = ""
            }
            If $Length(chunk) + $Length(line) > 32000 {
                Set chunkNum = chunkNum + 1
                Set ^||adj(chunkNum) = chunk
                Set chunk = ""
            }
            Set chunk = chunk _ line
        } Else {
            Set result = result _ line
        }
    }
    If chunkNum > 0 {
        If chunk '= "" { Set chunkNum = chunkNum + 1, ^||adj(chunkNum) = chunk }
        Set ^||adj = chunkNum
        Return "CHUNKED:" _ chunkNum
    }
    Return result
}

/// Export from ^KG (string subscripts — builds node index on the fly)
ClassMethod ExportAdjacencyKG() As %String [ Private ]
{
    // Pass 1: assign integer indices to all nodes
    Kill ^||nodeIdx
    Set n = 0, node = ""
    For {
        Set node = $Order(^KG("deg", node))
        Quit:node=""
        Set ^||nodeIdx(node) = n
        Set ^||nodeMap(n) = node
        Set n = n + 1
    }
    // Also pick up nodes only in "in" (no outgoing edges)
    Set node = ""
    For {
        Set node = $Order(^KG("in", node))
        Quit:node=""
        If '$Data(^||nodeIdx(node)) {
            Set ^||nodeIdx(node) = n
            Set ^||nodeMap(n) = node
            Set n = n + 1
        }
    }

    // Pass 2: build adjacency
    Set result = "", chunk = "", chunkNum = 0
    Set src = ""
    For {
        Set src = $Order(^KG("out", src))
        Quit:src=""
        Set sIdx = $Get(^||nodeIdx(src))
        If sIdx = "" Continue
        Set nbrs = "", pred = ""
        For {
            Set pred = $Order(^KG("out", src, pred))
            Quit:pred=""
            Set dst = ""
            For {
                Set dst = $Order(^KG("out", src, pred, dst))
                Quit:dst=""
                Set dIdx = $Get(^||nodeIdx(dst))
                If dIdx = "" Continue
                Set:nbrs'="" nbrs = nbrs _ ","
                Set nbrs = nbrs _ dIdx
            }
        }
        Set line = sIdx _ ":" _ nbrs _ $Char(10)
        If $Length(result) + $Length(line) > 3000000 {
            If chunkNum = 0 {
                Set chunkNum = 1
                Set ^||adj(1) = result
                Set result = ""
            }
            If $Length(chunk) + $Length(line) > 32000 {
                Set chunkNum = chunkNum + 1
                Set ^||adj(chunkNum) = chunk
                Set chunk = ""
            }
            Set chunk = chunk _ line
        } Else {
            Set result = result _ line
        }
    }
    If chunkNum > 0 {
        If chunk '= "" { Set chunkNum = chunkNum + 1, ^||adj(chunkNum) = chunk }
        Set ^||adj = chunkNum
        Return "CHUNKED:" _ chunkNum _ "|NODEMAP"
    }
    // Append node map header so Rust can resolve indices to strings
    Set header = "NODEMAP:" _ n _ $Char(10)
    For i = 0:1:n-1 { Set header = header _ i _ "=" _ ^||nodeMap(i) _ $Char(10) }
    Return header _ "---" _ $Char(10) _ result
}

// ============================================================
// T019: KHopNeighbors — seed-local BFS at $ORDER speed
// ============================================================

/// k-hop BFS from a seed node. Returns JSON with visited nodes and distances.
/// Runs entirely in ObjectScript — no Rust, no graph serialization.
/// Cost: ~0.5ms for 2-hop fanout-16 regardless of graph size.
ClassMethod KHopNeighbors(seedId As %String, k As %Integer = 2, maxNodes As %Integer = 1000) As %String
{
    Set useNKG = ..UseNKG()
    If useNKG {
        Set seedIdx = $Get(^NKG("$NI", seedId))
        If seedIdx = "" Return "{""error"":""node not found"",""node"":"""_seedId_"""}"
    }

    Kill ^||visited, ^||frontier, ^||nextFrontier
    If useNKG {
        Set ^||visited(seedIdx) = 0
        Set ^||frontier(seedIdx) = ""
    } Else {
        Set ^||visited(seedId) = 0
        Set ^||frontier(seedId) = ""
    }
    Set totalNodes = 1

    For hop = 1:1:k {
        Kill ^||nextFrontier
        Set node = ""
        For {
            Set node = $Order(^||frontier(node))
            Quit:node=""
            If useNKG {
                Set pred = ""
                For {
                    Set pred = $Order(^NKG(-1, node, pred))
                    Quit:pred=""
                    Set dst = ""
                    For {
                        Set dst = $Order(^NKG(-1, node, pred, dst))
                        Quit:dst=""
                        If '$Data(^||visited(dst)) {
                            Set ^||visited(dst) = hop
                            Set ^||nextFrontier(dst) = ""
                            Set totalNodes = totalNodes + 1
                            If totalNodes >= maxNodes Goto KHDone
                        }
                    }
                }
            } Else {
                Set pred = ""
                For {
                    Set pred = $Order(^KG("out", node, pred))
                    Quit:pred=""
                    Set dst = ""
                    For {
                        Set dst = $Order(^KG("out", node, pred, dst))
                        Quit:dst=""
                        If '$Data(^||visited(dst)) {
                            Set ^||visited(dst) = hop
                            Set ^||nextFrontier(dst) = ""
                            Set totalNodes = totalNodes + 1
                            If totalNodes >= maxNodes Goto KHDone
                        }
                    }
                }
            }
        }
        Kill ^||frontier
        Merge ^||frontier = ^||nextFrontier
    }

KHDone
    // Serialize result
    Set result = "{""seed"":"""_seedId_""",""hops"":"_k_",""totalNodes"":"_totalNodes_",""nodes"":["
    Set node = "", first = 1
    For {
        Set node = $Order(^||visited(node), 1, dist)
        Quit:node=""
        Set:'first result = result _ ","
        Set first = 0
        If useNKG {
            Set nodeStr = $Get(^NKG("$ND", node), node)
        } Else {
            Set nodeStr = node
        }
        Set result = result _ "{""id"":"""_nodeStr_""",""dist"":"_dist_"}"
    }
    Set result = result _ "]}"
    Return result
}

// ============================================================
// T020: PPRNative — Personalized PageRank via power iteration
// ============================================================

/// Personalized PageRank from seed node(s). Runs in ObjectScript.
/// Returns top-K nodes with scores as JSON.
ClassMethod PPRNative(seedId As %String, alpha As %Double = 0.85, maxIter As %Integer = 20, topK As %Integer = 20) As %String
{
    Set useNKG = ..UseNKG()

    // Initialize: collect seed's local neighborhood (2-hop)
    // and run PPR only on the subgraph
    Set khJson = ..KHopNeighbors(seedId, 2, 5000)
    Set kh = ##class(%DynamicObject).%FromJSON(khJson)
    If kh.%IsDefined("error") Return khJson

    // Build local adjacency from the visited nodes
    Kill ^||pprAdj, ^||pprScore, ^||pprNext
    Set nodes = kh.nodes
    Set n = nodes.%Size()

    // Index the local nodes
    Kill ^||localIdx
    For i = 0:1:n-1 {
        Set nodeId = nodes.%Get(i).id
        Set ^||localIdx(nodeId) = i
    }

    // Build local adjacency
    For i = 0:1:n-1 {
        Set nodeId = nodes.%Get(i).id
        Set deg = 0
        If useNKG {
            Set nkgIdx = $Get(^NKG("$NI", nodeId), "")
            If nkgIdx '= "" {
                Set pred = ""
                For {
                    Set pred = $Order(^NKG(-1, nkgIdx, pred))
                    Quit:pred=""
                    Set dst = ""
                    For {
                        Set dst = $Order(^NKG(-1, nkgIdx, pred, dst))
                        Quit:dst=""
                        Set dstStr = $Get(^NKG("$ND", dst), "")
                        If $Data(^||localIdx(dstStr)) {
                            Set ^||pprAdj(i, deg) = ^||localIdx(dstStr)
                            Set deg = deg + 1
                        }
                    }
                }
            }
        } Else {
            Set pred = ""
            For {
                Set pred = $Order(^KG("out", nodeId, pred))
                Quit:pred=""
                Set dst = ""
                For {
                    Set dst = $Order(^KG("out", nodeId, pred, dst))
                    Quit:dst=""
                    If $Data(^||localIdx(dst)) {
                        Set ^||pprAdj(i, deg) = ^||localIdx(dst)
                        Set deg = deg + 1
                    }
                }
            }
        }
        Set ^||pprAdj(i) = deg
    }

    // Find seed index
    Set seedLocalIdx = $Get(^||localIdx(seedId), 0)

    // Initialize scores: teleport vector = 1 for seed, 0 for others
    For i = 0:1:n-1 { Set ^||pprScore(i) = 0 }
    Set ^||pprScore(seedLocalIdx) = 1

    // Power iteration
    For iter = 1:1:maxIter {
        For i = 0:1:n-1 { Set ^||pprNext(i) = 0 }
        For i = 0:1:n-1 {
            Set deg = $Get(^||pprAdj(i), 0)
            If deg = 0 Continue
            Set contrib = ^||pprScore(i) / deg
            For j = 0:1:deg-1 {
                Set nbr = ^||pprAdj(i, j)
                Set ^||pprNext(nbr) = ^||pprNext(nbr) + (alpha * contrib)
            }
        }
        // Add teleport
        For i = 0:1:n-1 {
            Set ^||pprNext(i) = ^||pprNext(i) + ((1 - alpha) * $Select(i=seedLocalIdx:1, 1:0))
        }
        // Swap
        For i = 0:1:n-1 { Set ^||pprScore(i) = ^||pprNext(i) }
    }

    // Collect top-K by score (simple selection)
    Kill ^||ranked
    For i = 0:1:n-1 {
        Set score = ^||pprScore(i)
        Set ^||ranked(-score, i) = ""
    }

    Set result = "{""seed"":"""_seedId_""",""alpha"":"_alpha_",""iterations"":"_maxIter_",""scores"":["
    Set first = 1, count = 0
    Set sk = ""
    For {
        Set sk = $Order(^||ranked(sk))
        Quit:sk=""
        Set idx = ""
        For {
            Set idx = $Order(^||ranked(sk, idx))
            Quit:idx=""
            Set nodeId = nodes.%Get(idx).id
            Set score = -sk
            Set:'first result = result _ ","
            Set first = 0
            Set result = result _ "{""id"":"""_nodeId_""",""score"":"_$FNumber(score,"",6)_"}"
            Set count = count + 1
            If count >= topK Goto PPRDone
        }
    }

PPRDone
    Set result = result _ "]}"
    Return result
}

// ============================================================
// Wrappers that replace CacheGraphJson path for seed-local
// ============================================================

/// K-hop neighborhood — pure ObjectScript, no Rust needed
ClassMethod KHopJson(seedId As %String, k As %Integer = 2, maxNodes As %Integer = 1000) As %String
{
    Return ..KHopNeighbors(seedId, k, maxNodes)
}

/// Personalized PageRank — pure ObjectScript on local subgraph
ClassMethod PPRJson(seedId As %String, alpha As %Double = 0.85, maxIter As %Integer = 20) As %String
{
    Return ..PPRNative(seedId, alpha, maxIter)
}

/// Random walk — pure ObjectScript
ClassMethod RandomWalkJson(seedId As %String, walkLength As %Integer = 20, numWalks As %Integer = 10) As %String
{
    Set useNKG = ..UseNKG()
    Set result = "["
    For w = 1:1:numWalks {
        Set:'(w=1) result = result _ ","
        Set result = result _ "["
        Set current = seedId
        If useNKG { Set currentIdx = $Get(^NKG("$NI", seedId), "") }
        Set result = result _ """" _ seedId _ """"
        For step = 1:1:walkLength {
            // Collect neighbors
            Kill ^||walkNbrs
            Set nbrCount = 0
            If useNKG && (currentIdx '= "") {
                Set pred = "" For { Set pred=$Order(^NKG(-1,currentIdx,pred)) Quit:pred=""  Set dst="" For { Set dst=$Order(^NKG(-1,currentIdx,pred,dst)) Quit:dst=""  Set nbrCount=nbrCount+1, ^||walkNbrs(nbrCount)=dst } }
            } Else {
                Set pred="" For { Set pred=$Order(^KG("out",current,pred)) Quit:pred=""  Set dst="" For { Set dst=$Order(^KG("out",current,pred,dst)) Quit:dst=""  Set nbrCount=nbrCount+1, ^||walkNbrs(nbrCount)=dst } }
            }
            If nbrCount = 0 Quit
            // Pick random neighbor
            Set pick = $Random(nbrCount) + 1
            If useNKG {
                Set currentIdx = ^||walkNbrs(pick)
                Set current = $Get(^NKG("$ND", currentIdx), currentIdx)
            } Else {
                Set current = ^||walkNbrs(pick)
            }
            Set result = result _ ",""" _ current _ """"
        }
        Set result = result _ "]"
    }
    Set result = result _ "]"
    Return result
}

 ClassMethod BFSJson(seed As %String, predicatesJson As %String = "[]", maxHops As %Integer = 3, maxResults As %Integer = 0) As %String
{
    Set dllPath = $Get(^||NKGAccel("libPath"), "/usr/irissys/mgr/libarno_callout.so")
    If '$Data(^||NKGAccel("bfs_fn")) {
        Set dllid = $ZF(-4, 1, dllPath)
        If dllid = 0 {
            Return ##class(Graph.KG.Traversal).BFSFastJson(seed, predicatesJson, maxHops, "")
        }
        Set ^||NKGAccel("dllid") = dllid
        Set ^||NKGAccel("bfs_fn") = $ZF(-4, 3, dllid, "kg_bfs_global")
    }
    Set dllid = ^||NKGAccel("dllid")
    Set fnid = ^||NKGAccel("bfs_fn")
    If fnid = 0 {
        Return ##class(Graph.KG.Traversal).BFSFastJson(seed, predicatesJson, maxHops, "")
    }

    // Per-seed export: only export the subgraph reachable from seed (bounded by maxHops+1)
    // This keeps adjStr small even on 300K+ edge graphs
    Set adjStr = ..ExportAdjacencyFromSeed(seed, maxHops + 1, predicatesJson)
    If $Length(adjStr) = 0 {
        Return ##class(Graph.KG.Traversal).BFSFastJson(seed, predicatesJson, maxHops, "")
    }
    Return $ZF(-5, dllid, fnid, adjStr, seed, predicatesJson, maxHops, maxResults)
}

ClassMethod ExportAdjacencyFromSeed(seed As %String, maxHops As %Integer = 4, predicatesJson As %String = "[]") As %String [ Private ]
{
    Set seedIdx = $Get(^NKG("$NI", seed), "")
    If seedIdx = "" Return ""

    Set preds = ""
    If predicatesJson '= "[]" && (predicatesJson '= "") {
        Try { Set preds = ##class(%DynamicArray).%FromJSON(predicatesJson) } Catch { Set preds = "" }
    }

    // BFS to collect reachable node indices within maxHops
    Kill ^||seedBFS
    Set ^||seedBFS("frontier", seedIdx) = ""
    Set ^||seedBFS("seen", seedIdx) = ""

    For hop = 1:1:maxHops {
        Kill ^||seedBFS("nextFrontier")
        Set s = ""
        For {
            Set s = $Order(^||seedBFS("frontier", s))
            Quit:s=""
            Set pred = ""
            For {
                Set pred = $Order(^NKG(-1, s, pred))
                Quit:pred=""
                If $IsObject(preds) {
                    Set allowed = 0
                    Set predStr = $Get(^NKG("$LS", -pred - 1), "")
                    For pi = 0:1:preds.%Size()-1 {
                        If preds.%Get(pi) = predStr { Set allowed = 1 Quit }
                    }
                    If 'allowed Continue
                }
                Set dst = ""
                For {
                    Set dst = $Order(^NKG(-1, s, pred, dst))
                    Quit:dst=""
                    Set ^||seedBFS("nodes", s) = ""
                    Set ^||seedBFS("nodes", dst) = ""
                    If '$Data(^||seedBFS("seen", dst)) {
                        Set ^||seedBFS("seen", dst) = ""
                        Set ^||seedBFS("nextFrontier", dst) = ""
                    }
                }
            }
            // Inbound traversal for undirected
            Set pred = ""
            For {
                Set pred = $Order(^NKG(-2, s, pred))
                Quit:pred=""
                Set dst = ""
                For {
                    Set dst = $Order(^NKG(-2, s, pred, dst))
                    Quit:dst=""
                    Set ^||seedBFS("nodes", s) = ""
                    Set ^||seedBFS("nodes", dst) = ""
                    If '$Data(^||seedBFS("seen", dst)) {
                        Set ^||seedBFS("seen", dst) = ""
                        Set ^||seedBFS("nextFrontier", dst) = ""
                    }
                }
            }
        }
        If '$Data(^||seedBFS("nextFrontier")) Quit
        Merge ^||seedBFS("frontier") = ^||seedBFS("nextFrontier")
        Kill ^||seedBFS("nextFrontier")
    }

    // Export only edges between collected nodes
    Set result = ""
    Set nodeIdx = ""
    For {
        Set nodeIdx = $Order(^||seedBFS("nodes", nodeIdx))
        Quit:nodeIdx=""
        Set srcStr = $Get(^NKG("$ND", nodeIdx), nodeIdx)
        Set pred = ""
        For {
            Set pred = $Order(^NKG(-1, nodeIdx, pred))
            Quit:pred=""
            Set predStr = $Get(^NKG("$LS", -pred - 1), pred)
            Set dst = ""
            For {
                Set dst = $Order(^NKG(-1, nodeIdx, pred, dst))
                Quit:dst=""
                If '$Data(^||seedBFS("nodes", dst)) Continue
                Set dstStr = $Get(^NKG("$ND", dst), dst)
                Set result = result _ srcStr _ $Char(31) _ predStr _ $Char(31) _ dstStr _ $Char(10)
            }
        }
    }
    Kill ^||seedBFS
    Return result
}

ClassMethod ExportAdjacencyWithPreds() As %String [ Private ]
{
    Set result = "", chunk = "", chunkNum = 0
    Set src = ""
    For {
        Set src = $Order(^NKG(-1, src))
        Quit:src=""
        Set srcStr = $Get(^NKG("$ND", src), src)
        Set pred = ""
        For {
            Set pred = $Order(^NKG(-1, src, pred))
            Quit:pred=""
            Set predStr = $Get(^NKG("$LS", -pred - 1), pred)
            Set dst = ""
            For {
                Set dst = $Order(^NKG(-1, src, pred, dst))
                Quit:dst=""
                Set dstStr = $Get(^NKG("$ND", dst), dst)
                Set line = srcStr _ $Char(31) _ predStr _ $Char(31) _ dstStr _ $Char(10)
                If $Length(result) + $Length(line) > 3000000 {
                    If chunkNum = 0 {
                        Set chunkNum = 1
                        Set ^||adjWP(1) = result
                        Set result = ""
                    }
                    If $Length(chunk) + $Length(line) > 32000 {
                        Set chunkNum = chunkNum + 1
                        Set ^||adjWP(chunkNum) = chunk
                        Set chunk = ""
                    }
                    Set chunk = chunk _ line
                } Else {
                    Set result = result _ line
                }
            }
        }
    }
    If chunkNum > 0 {
        If chunk '= "" { Set chunkNum = chunkNum + 1, ^||adjWP(chunkNum) = chunk }
        Set ^||adjWP = chunkNum
        Return "CHUNKED:" _ chunkNum
    }
    Return result
}

ClassMethod Capabilities() As %String
{
    Set cap = {}
    Do cap.%Set("version", "0.1.0")
    Do cap.%Set("nkg_accel", 1, "boolean")
    Do cap.%Set("nkg_data", ..UseNKG(), "boolean")
    Do cap.%Set("global_set", ##class(%Dictionary.ClassDefinition).%ExistsId("User.GlobalSet"), "boolean")
    Do cap.%Set("vec_index", ##class(%Dictionary.ClassDefinition).%ExistsId("Graph.KG.VecIndex"), "boolean")
    Set hasRust = 0
    Try {
        If ##class(Graph.KG.ArnoAccel).IsAvailable() Set hasRust = 1
    } Catch {}
    Do cap.%Set("rust_callout", hasRust, "boolean")
    Set algos = ["khop", "ppr", "random_walk", "export_adjacency", "bfs"]
    Do cap.%Set("algorithms", algos)
    Set rustAlgos = []
    If hasRust {
        Do rustAlgos.%Push("pagerank")
        Do rustAlgos.%Push("wcc")
        Do rustAlgos.%Push("cdlp")
        Do rustAlgos.%Push("bfs")
    }
    Do cap.%Set("rust_algorithms", rustAlgos)
    Do cap.%Set("bfs", 1, "boolean")
    Return cap.%ToJSON()
}

}
