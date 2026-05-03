/// Pure ObjectScript BFS implementation for maximum performance and stability
Class Graph.KG.Traversal Extends %RegisteredObject [ SqlSchemaName = Graph_KG ]
{

/// Rebuilds the ^KG functional index from SQL tables with high-performance batching
ClassMethod BuildKG() As %Status
{
	Try {
		// Disable journaling for this process to maximize speed (requires %Admin_Operate)
		// Do $System.Process.SetJournaling(0) 
		
		Kill ^KG
		TStart
		
		// 1. Labels
		&sql(DECLARE c1 CURSOR FOR SELECT s, label FROM Graph_KG.rdf_labels)
		&sql(OPEN c1)
		For {
			&sql(FETCH c1 INTO :s, :label)
			Quit:SQLCODE'=0
			Set ^KG("label", label, s) = ""
		}
		&sql(CLOSE c1)
		
		// 2. Props
		&sql(DECLARE c2 CURSOR FOR SELECT s, key, val FROM Graph_KG.rdf_props)
		&sql(OPEN c2)
		For {
			&sql(FETCH c2 INTO :s, :key, :val)
			Quit:SQLCODE'=0
			Set ^KG("prop", s, key) = val
		}
		&sql(CLOSE c2)
		
		// 3. Edges & Stats (including weights)
		Set count = 0
		&sql(DECLARE c3 CURSOR FOR SELECT s, p, o_id, qualifiers FROM Graph_KG.rdf_edges)
		&sql(OPEN c3)
		For {
			&sql(FETCH c3 INTO :s, :p, :o, :qual)
			Quit:SQLCODE'=0
			
			If (s = "") || (p = "") || (o = "") Continue
			
			Set weight = 1.0
			If qual '= "" {
				Try {
					Set obj = ##class(%DynamicObject).%FromJSON(qual)
					If (obj '= "") && (obj.weight '= "") Set weight = obj.weight
				} Catch {}
			}
			
			Set ^KG("out", 0, s, p, o) = weight
			Set ^KG("in", 0, o, p, s) = weight
			Set tmp = $Increment(^KG("deg", s))
			Set tmp = $Increment(^KG("degp", s, p))
			
			Set count = count + 1
			// Commit every 50k records to manage transaction buffer
			If count # 50000 = 0 {
				TCommit
				TStart
			}
		}
		&sql(CLOSE c3)
		
		TCommit

		// BuildNKG requires Graph.KG.GraphIndex — optional, non-fatal if missing
		Try { Do ..BuildNKG() } Catch {}

		Return $$$OK
	} Catch ex {
		TRollback
		Return ex.AsStatus()
	}
}

ClassMethod BuildNKG() As %Status
{
	Kill ^NKG
	Do ##class(Graph.KG.GraphIndex).InitStructuralLabels()
	
	Set edgeCount = 0
	Set src = ""
	For {
		Set src = $Order(^KG("out", 0, src))
		Quit:src=""
		Set sIdx = ##class(Graph.KG.GraphIndex).InternNode(src)
		Set pred = ""
		For {
			Set pred = $Order(^KG("out", 0, src, pred))
			Quit:pred=""
			Set pIdx = ##class(Graph.KG.GraphIndex).InternLabel(pred)
			Set dst = ""
			For {
				Set dst = $Order(^KG("out", 0, src, pred, dst), 1, weight)
				Quit:dst=""
				Set oIdx = ##class(Graph.KG.GraphIndex).InternNode(dst)
				If '$Data(^NKG(-1, sIdx, -(pIdx+1), oIdx)) {
					Set ^NKG(-1, sIdx, -(pIdx+1), oIdx) = weight
					Set ^NKG(-2, oIdx, -(pIdx+1), sIdx) = weight
					Set edgeCount = edgeCount + 1
				}
			}
		}
		Set ^NKG(-3, sIdx) = $Get(^KG("deg", src))
	}
	Set ^NKG("$meta", "edgeCount") = edgeCount
	Set tmp = $Increment(^NKG("$meta", "version"))
	Return $$$OK
}

/// Helper for single predicate traversal
ClassMethod TraverseWithPredicate(s As %String, p As %String, hop As %Integer, maxHops As %Integer, nextP As %String, dstLabel As %String, ByRef pathId As %Integer, results As %DynamicArray, ByRef nextFrontier, ByRef seen) [ Internal, Private ]
{
    Set o = ""
    For {
        Set o = $Order(^KG("out", 0, s, p, o))
        If o = "" Quit
        
        // Filter by label if requested
        If (dstLabel '= "") && ('$Data(^KG("label", dstLabel, o))) Continue
        
        // T014: O(1) Hop Rejection
        If (hop < maxHops) && (nextP '= "") && ('$Data(^KG("degp", o, nextP))) Continue
        
        Set pathId = pathId + 1
        Set obj = ##class(%DynamicObject).%New()
        Set obj.id = pathId
        Set obj.step = hop
        Set obj.s = s
        Set obj.p = p
        Set obj.o = o
        Set obj.w = $Get(^KG("out", 0, s, p, o), 1.0)
        Do results.%Push(obj)
        
        If '$Data(seen(o)) Set nextFrontier(o) = ""
    }
}

/// Helper for all predicates traversal
ClassMethod TraverseAllPredicates(s As %String, hop As %Integer, maxHops As %Integer, nextP As %String, dstLabel As %String, ByRef pathId As %Integer, results As %DynamicArray, ByRef nextFrontier, ByRef seen) [ Internal, Private ]
{
    Set p = ""
    For {
        Set p = $Order(^KG("out", 0, s, p))
        If p = "" Quit
        Do ..TraverseWithPredicate(s, p, hop, maxHops, nextP, dstLabel, .pathId, results, .nextFrontier, .seen)
    }
}

/// Optimized BFS using Process-Private Globals and $ListBuild to bypass object bottleneck
/// @param direction "out" (default), "in", or "both" for undirected BFS
/// @returns Number of edges found. Results are in ^||BFS.Results(i) = $lb(s, p, o, w, step)
ClassMethod BFSFast(srcId As %String, preds As %DynamicArray = "", maxHops As %Integer = 2, dstLabel As %String = "", direction As %String = "out") As %Integer
{
    Kill ^||BFS.Results
    If (srcId = "") || (maxHops <= 0) Return 0
    
    Set count = 0
    Set seen(srcId) = ""
    Set frontier(srcId) = ""
    
    For hop = 1:1:maxHops {
        Kill nextFrontier
        Set s = ""
        Set fixedP = $Select($IsObject(preds):preds.%Get(hop-1), 1:"")
        Set nextP = $Select($IsObject(preds):preds.%Get(hop), 1:"")
        
        For {
            Set s = $Order(frontier(s))
            If s = "" Quit
            
            If direction = "out" || (direction = "both") {
                If fixedP '= "" {
                    Do ..TraverseWithPredicateFast(s, fixedP, hop, maxHops, nextP, dstLabel, .count, .nextFrontier, .seen)
                } Else {
                    Do ..TraverseAllPredicatesFast(s, hop, maxHops, nextP, dstLabel, .count, .nextFrontier, .seen)
                }
            }
            
            If direction = "in" || (direction = "both") {
                If fixedP '= "" {
                    Do ..TraverseWithPredicateFastInbound(s, fixedP, hop, .count, .nextFrontier, .seen)
                } Else {
                    Do ..TraverseAllPredicatesFastInbound(s, hop, .count, .nextFrontier, .seen)
                }
            }
        }
        
        If '$Data(nextFrontier) Quit
        Merge frontier = nextFrontier
        Set node = ""
        For {
            Set node = $Order(nextFrontier(node))
            If node = "" Quit
            Set seen(node) = ""
        }
    }
    Return count
}

ClassMethod TraverseWithPredicateFast(s As %String, p As %String, hop As %Integer, maxHops As %Integer, nextP As %String, dstLabel As %String, ByRef count As %Integer, ByRef nextFrontier, ByRef seen) [ Internal, Private ]
{
    Set o = ""
    For {
        Set o = $Order(^KG("out", 0, s, p, o))
        If o = "" Quit
        If (dstLabel '= "") && ('$Data(^KG("label", dstLabel, o))) Continue
        If (hop < maxHops) && (nextP '= "") && ('$Data(^KG("degp", o, nextP))) Continue
        
        Set count = count + 1
        Set ^||BFS.Results(count) = $ListBuild(s, p, o, $Get(^KG("out", 0, s, p, o), 1.0), hop)
        
        If '$Data(seen(o)) Set nextFrontier(o) = ""
    }
}

ClassMethod TraverseAllPredicatesFast(s As %String, hop As %Integer, maxHops As %Integer, nextP As %String, dstLabel As %String, ByRef count As %Integer, ByRef nextFrontier, ByRef seen) [ Internal, Private ]
{
    Set p = ""
    For {
        Set p = $Order(^KG("out", 0, s, p))
        If p = "" Quit
        Do ..TraverseWithPredicateFast(s, p, hop, maxHops, nextP, dstLabel, .count, .nextFrontier, .seen)
    }
}

/// Inbound BFS: traverse ^KG("in", 0, s, p, o) where s is the target, o is the source.
/// Result stored as (o, p, s) to preserve original edge direction.
ClassMethod TraverseWithPredicateFastInbound(s As %String, p As %String, hop As %Integer, ByRef count As %Integer, ByRef nextFrontier, ByRef seen) [ Internal, Private ]
{
    Set o = ""
    For {
        Set o = $Order(^KG("in", 0, s, p, o))
        If o = "" Quit
        Set count = count + 1
        Set ^||BFS.Results(count) = $ListBuild(s, p, o, $Get(^KG("out", 0, o, p, s), 1.0), hop)
        If '$Data(seen(o)) Set nextFrontier(o) = ""
    }
}

ClassMethod TraverseAllPredicatesFastInbound(s As %String, hop As %Integer, ByRef count As %Integer, ByRef nextFrontier, ByRef seen) [ Internal, Private ]
{
    Set p = ""
    For {
        Set p = $Order(^KG("in", 0, s, p))
        If p = "" Quit
        Do ..TraverseWithPredicateFastInbound(s, p, hop, .count, .nextFrontier, .seen)
    }
}

/// Main BFS implementation
ClassMethod BFS(srcId As %String, preds As %DynamicArray = "", maxHops As %Integer = 2, dstLabel As %String = "") As %DynamicArray
{
    Set results = ##class(%DynamicArray).%New()
    If (srcId = "") || (maxHops <= 0) Return results
    
    Set pathId = 0
    Set seen(srcId) = ""
    Set frontier(srcId) = ""
    
    For hop = 1:1:maxHops {
        Kill nextFrontier
        Set s = ""
        Set fixedP = $Select($IsObject(preds):preds.%Get(hop-1), 1:"")
        Set nextP = $Select($IsObject(preds):preds.%Get(hop), 1:"")
        
        For {
            Set s = $Order(frontier(s))
            If s = "" Quit
            
            If fixedP '= "" {
                Do ..TraverseWithPredicate(s, fixedP, hop, maxHops, nextP, dstLabel, .pathId, results, .nextFrontier, .seen)
            } Else {
                Do ..TraverseAllPredicates(s, hop, maxHops, nextP, dstLabel, .pathId, results, .nextFrontier, .seen)
            }
        }
        
        If '$Data(nextFrontier) Quit
        Merge frontier = nextFrontier
        
        Set node = ""
        For {
            Set node = $Order(nextFrontier(node))
            If node = "" Quit
            Set seen(node) = ""
        }
    }
    Return results
}

/// Python implementation for US1 readability targets
ClassMethod TraverseWithPredicatePy(s As %String, p As %String, hop As %Integer, dstLabel As %String) [ Internal, Private, Language = python ]
{
    import iris
    kg = iris.gref("^KG")
    matches = []
    o = ""
    while True:
        o = kg.order(["out", s, p, o])
        if not o: break
        if dstLabel and not kg.data(["label", dstLabel, o]):
            continue
        weight = kg.get(["out", s, p, o]) or 1.0
        matches.append({"s": s, "p": p, "o": o, "step": hop, "w": float(weight)})
    return matches
}

/// Compatibility wrapper for existing API
ClassMethod BFSJSON(srcId As %String, preds As %DynamicArray, maxHops As %Integer = 2, dstLabel As %String = "") As %DynamicArray
{
    Return ..BFS(srcId, preds, maxHops, dstLabel)
}

/// Run BFSFast and return results as a JSON string.
/// Primary entry point for Python callers -- reads ^||BFS.Results in the same session.
/// 
/// @param srcId    Source node ID
/// @param predsJson JSON array of predicates per hop, e.g. '["rel1","rel2"]' or null/"" for all
/// @param maxHops  Maximum hops (default 2)
/// @param dstLabel Filter destination nodes by label (empty = no filter)
/// @returns JSON array: [{"s":..,"p":..,"o":..,"w":..,"step":..}, ...] in traversal order
ClassMethod BFSFastJson(srcId As %String, predsJson As %String = "", maxHops As %Integer = 2, dstLabel As %String = "", direction As %String = "out") As %String
{
    // Parse predicates JSON into a %DynamicArray (null/"" means all predicates)
    Set preds = ""
    If (predsJson '= "") && (predsJson '= "null") {
        Try {
            Set preds = ##class(%DynamicArray).%FromJSON(predsJson)
        } Catch {
            Set preds = ""
        }
    }

    // Run BFSFast -- writes to ^||BFS.Results(i) = $ListBuild(s,p,o,w,step)
    Set count = ..BFSFast(srcId, preds, maxHops, dstLabel, direction)

    If count = 0 Return "[]"

    // Build JSON array from ^||BFS.Results
    Set result = ##class(%DynamicArray).%New()
    For i = 1:1:count {
        Set lb = $Get(^||BFS.Results(i))
        If lb = "" Continue
        Set obj = ##class(%DynamicObject).%New()
        Do obj.%Set("s",    $ListGet(lb, 1))
        Do obj.%Set("p",    $ListGet(lb, 2))
        Do obj.%Set("o",    $ListGet(lb, 3))
        Do obj.%Set("w",    +$ListGet(lb, 4))
        Do obj.%Set("step", +$ListGet(lb, 5))
        Do result.%Push(obj)
    }

    Return result.%ToJSON()
}

ClassMethod ShortestPathJson(srcId As %String, dstId As %String, maxHops As %Integer = 5, predsJson As %String = "", direction As %String = "both", findAll As %Integer = 0) As %String
{
	Kill ^||SP.parents, ^||SP.frontier, ^||SP.seen

	If srcId = "" || (dstId = "") {
		Kill ^||SP.parents, ^||SP.frontier, ^||SP.seen
		Return "[]"
	}

	If srcId = dstId {
		Set zeroPath = ##class(%DynamicObject).%New()
		Set nodes = ##class(%DynamicArray).%New()
		Do nodes.%Push(srcId)
		Do zeroPath.%Set("nodes", nodes)
		Do zeroPath.%Set("rels", ##class(%DynamicArray).%New())
		Do zeroPath.%Set("length", 0)
		Set result = ##class(%DynamicArray).%New()
		Do result.%Push(zeroPath)
		Kill ^||SP.parents, ^||SP.frontier, ^||SP.seen
		Return result.%ToJSON()
	}

	Set preds = ""
	If predsJson '= "" && (predsJson '= "[]") && (predsJson '= "null") {
		Set preds = ##class(%DynamicArray).%FromJSON(predsJson)
	}

	Set ^||SP.seen(srcId) = ""
	Set ^||SP.frontier(srcId) = ""

	Set foundDepth = -1

	For hop = 1:1:maxHops {
		Kill ^||SP.nextFrontier
		Set s = ""
		For {
			Set s = $Order(^||SP.frontier(s))
			Quit:s=""

			If direction = "both" || (direction = "out") {
				Set p = ""
				For {
					Set p = $Order(^KG("out", 0, s, p))
					Quit:p=""
					If $IsObject(preds) {
						Set allowed = 0
						For pi = 0:1:(preds.%Size()-1) { If preds.%Get(pi) = p { Set allowed = 1  Quit } }
						Continue:'allowed
					}
					Set o = ""
					For {
						Set o = $Order(^KG("out", 0, s, p, o))
						Quit:o=""
						Set ^||SP.parents(o, s, p) = ""
						If o = dstId {
							Set foundDepth = hop
						} ElseIf '$Data(^||SP.seen(o)) {
							Set ^||SP.seen(o) = ""
							Set ^||SP.nextFrontier(o) = ""
						}
					}
				}
			}

			If direction = "both" {
				Set p = ""
				For {
					Set p = $Order(^KG("in", 0, s, p))
					Quit:p=""
					If $IsObject(preds) {
						Set allowed = 0
						For pi = 0:1:(preds.%Size()-1) { If preds.%Get(pi) = p { Set allowed = 1  Quit } }
						Continue:'allowed
					}
					Set o = ""
					For {
						Set o = $Order(^KG("in", 0, s, p, o))
						Quit:o=""
						Set ^||SP.parents(o, s, p) = ""
						If o = dstId {
							Set foundDepth = hop
						} ElseIf '$Data(^||SP.seen(o)) {
							Set ^||SP.seen(o) = ""
							Set ^||SP.nextFrontier(o) = ""
						}
					}
				}
			}
		}

		If foundDepth > 0 {
			Set results = ..BuildPaths(srcId, dstId, foundDepth)
			Kill ^||SP.parents, ^||SP.frontier, ^||SP.seen, ^||SP.nextFrontier
			Return results
		}

		If '$Data(^||SP.nextFrontier) Quit
		Kill ^||SP.frontier
		Merge ^||SP.frontier = ^||SP.nextFrontier
		Kill ^||SP.nextFrontier
	}

	Kill ^||SP.parents, ^||SP.frontier, ^||SP.seen
	Return "[]"
}

ClassMethod BuildPaths(srcId As %String, dstId As %String, depth As %Integer) As %String [ Private ]
{
	Set results = ##class(%DynamicArray).%New()
	Do ..DFSPaths(srcId, dstId, depth, $ListBuild(dstId), "", results)
	Return results.%ToJSON()
}

ClassMethod DFSPaths(srcId As %String, cur As %String, remaining As %Integer, nodesSoFar As %List, relsSoFar As %List, ByRef results As %DynamicArray) [ Private ]
{
	If cur = srcId && (remaining = 0) {
		Set pathObj = ##class(%DynamicObject).%New()
		Set nodes = ##class(%DynamicArray).%New()
		Set rels = ##class(%DynamicArray).%New()
		Set ptr = 0
		While $ListNext(nodesSoFar, ptr, n) { Do nodes.%Push(n) }
		Set ptr = 0
		While $ListNext(relsSoFar, ptr, r) { Do rels.%Push(r) }
		Do pathObj.%Set("nodes", nodes)
		Do pathObj.%Set("rels", rels)
		Do pathObj.%Set("length", $ListLength(nodesSoFar) - 1)
		Do results.%Push(pathObj)
		Return
	}

	If remaining <= 0 Return

	Set parentNode = ""
	For {
		Set parentNode = $Order(^||SP.parents(cur, parentNode))
		Quit:parentNode=""
		Set relType = $Order(^||SP.parents(cur, parentNode, ""))
		Set newNodes = $ListBuild(parentNode) _ nodesSoFar
		Set newRels = $ListBuild(relType) _ relsSoFar
		Do ..DFSPaths(srcId, parentNode, remaining - 1, newNodes, newRels, .results)
	}
}

ClassMethod BacktrackPath(srcId As %String, dstId As %String, depth As %Integer) As %DynamicObject [ Private ]
{
	Set nodes = ##class(%DynamicArray).%New()
	Set rels = ##class(%DynamicArray).%New()

	Set cur = dstId
	Set nodeStack = $ListBuild(dstId)
	Set relStack = ""

	While cur '= srcId {
		Set parentNode = $Order(^||SP.parents(cur, ""))
		If parentNode = "" Quit
		Set relType = $Order(^||SP.parents(cur, parentNode, ""))
		Set nodeStack = $ListBuild(parentNode) _ nodeStack
		Set relStack = $ListBuild(relType) _ relStack
		Set cur = parentNode
	}

	Set ptr = 0
	While $ListNext(nodeStack, ptr, n) { Do nodes.%Push(n) }
	Set ptr = 0
	While $ListNext(relStack, ptr, r) { Do rels.%Push(r) }

	Set path = ##class(%DynamicObject).%New()
	Do path.%Set("nodes", nodes)
	Do path.%Set("rels", rels)
	Do path.%Set("length", depth)
	Return path
}

ClassMethod DijkstraJson(srcId As %String, dstId As %String, weightProp As %String = "", maxCost As %Double = 9999, maxHops As %Integer = 10, direction As %String = "out") As %String
{
	Kill ^||Dij.pq, ^||Dij.parent, ^||Dij.seen, ^||Dij.dist

	Set srcId = $Get(srcId, "")
	Set dstId = $Get(dstId, "")
	Set weightProp = $Get(weightProp, "")

	If srcId = "" || (dstId = "") {
		Kill ^||Dij.pq, ^||Dij.parent, ^||Dij.seen, ^||Dij.dist
		Return "{}"
	}

	If srcId = dstId {
		Set result = ##class(%DynamicObject).%New()
		Set nodes = ##class(%DynamicArray).%New()
		Do nodes.%Push(srcId)
		Do result.%Set("nodes", nodes)
		Do result.%Set("rels", ##class(%DynamicArray).%New())
		Do result.%Set("costs", ##class(%DynamicArray).%New())
		Do result.%Set("length", 0)
		Do result.%Set("totalCost", 0)
		Kill ^||Dij.pq, ^||Dij.parent, ^||Dij.seen, ^||Dij.dist
		Return result.%ToJSON()
	}

	Set ^||Dij.pq(0, srcId) = ""
	Set ^||Dij.dist(srcId) = 0

	For {
		Set minCost = $Order(^||Dij.pq(""))
		Quit:minCost=""
		Set curNode = $Order(^||Dij.pq(minCost, ""))
		Quit:curNode=""
		Kill ^||Dij.pq(minCost, curNode)

		If $Data(^||Dij.seen(curNode)) Continue
		Set ^||Dij.seen(curNode) = ""

		If curNode = dstId {
			Set result = ..DijkstraBacktrack(srcId, dstId, +minCost)
			Kill ^||Dij.pq, ^||Dij.parent, ^||Dij.seen, ^||Dij.dist
			Return result
		}

		If minCost >= maxCost Continue

		If direction = "out" || (direction = "both") {
			Set p = ""
			For {
				Set p = $Order(^KG("out", 0, curNode, p))
				Quit:p=""
				Set neighbor = ""
				For {
					Set neighbor = $Order(^KG("out", 0, curNode, p, neighbor))
					Quit:neighbor=""
					If $Data(^||Dij.seen(neighbor)) Continue
					Set rawW = $Get(^KG("out", 0, curNode, p, neighbor), 1)
					Set edgeW = +rawW
					If edgeW <= 0 Set edgeW = 1.0
					Set newCost = minCost + edgeW
					If newCost > maxCost Continue
					If '$Data(^||Dij.seen(neighbor)) {
						If '$Data(^||Dij.dist(neighbor)) || (newCost < ^||Dij.dist(neighbor)) {
							Set ^||Dij.dist(neighbor) = newCost
							Set ^||Dij.pq(newCost, neighbor) = ""
							Set ^||Dij.parent(neighbor) = $ListBuild(curNode, p, edgeW)
						}
					}
				}
			}
		}

		If direction = "both" {
			Set p = ""
			For {
				Set p = $Order(^KG("in", 0, curNode, p))
				Quit:p=""
				Set neighbor = ""
				For {
					Set neighbor = $Order(^KG("in", 0, curNode, p, neighbor))
					Quit:neighbor=""
					If $Data(^||Dij.seen(neighbor)) Continue
					Set rawW = $Get(^KG("in", 0, curNode, p, neighbor), 1)
					Set edgeW = +rawW
					If edgeW <= 0 Set edgeW = 1.0
					Set newCost = minCost + edgeW
					If newCost > maxCost Continue
					If '$Data(^||Dij.dist(neighbor)) || (newCost < ^||Dij.dist(neighbor)) {
						Set ^||Dij.dist(neighbor) = newCost
						Set ^||Dij.pq(newCost, neighbor) = ""
						Set ^||Dij.parent(neighbor) = $ListBuild(curNode, p, edgeW)
					}
				}
			}
		}
	}

	Kill ^||Dij.pq, ^||Dij.parent, ^||Dij.seen, ^||Dij.dist
	Return "{}"
}

ClassMethod DijkstraBacktrack(srcId As %String, dstId As %String, totalCost As %Double) As %String [ Private ]
{
	Set nodeStack = $ListBuild(dstId)
	Set relStack = ""
	Set costStack = ""
	Set cur = dstId

	While cur '= srcId {
		Set lb = $Get(^||Dij.parent(cur))
		If lb = "" Quit
		Set parentNode = $ListGet(lb, 1)
		Set relType = $ListGet(lb, 2)
		Set edgeCost = +$ListGet(lb, 3)
		Set nodeStack = $ListBuild(parentNode) _ nodeStack
		Set relStack = $ListBuild(relType) _ relStack
		Set costStack = $ListBuild(edgeCost) _ costStack
		Set cur = parentNode
	}

	Set nodes = ##class(%DynamicArray).%New()
	Set rels = ##class(%DynamicArray).%New()
	Set costs = ##class(%DynamicArray).%New()

	Set ptr = 0
	While $ListNext(nodeStack, ptr, n) { Do nodes.%Push(n) }
	Set ptr = 0
	While $ListNext(relStack, ptr, r) { Do rels.%Push(r) }
	Set ptr = 0
	While $ListNext(costStack, ptr, c) { Do costs.%Push(+c) }

	Set result = ##class(%DynamicObject).%New()
	Do result.%Set("nodes", nodes)
	Do result.%Set("rels", rels)
	Do result.%Set("costs", costs)
	Do result.%Set("length", nodes.%Size() - 1)
	Do result.%Set("totalCost", totalCost)
	Return result.%ToJSON()
}

ClassMethod DijkstraProc(srcId As %String, dstId As %String, weightProp As %String = "", maxCost As %Double = 9999, maxHops As %Integer = 10, direction As %String = "out") As %String [ SqlProc, SqlName = DijkstraPath ]
{
	Quit ..DijkstraJson(srcId, dstId, weightProp, maxCost, maxHops, direction)
}

ClassMethod KGEdgeCount(maxCount As %Integer = 10000) As %Integer
{
    Set count = 0
    Set s = ""
    For {
        Set s = $Order(^KG("out", 0, s))
        Quit:s=""
        Set count = count + 1
        If count >= maxCount Quit
    }
    Return count
}

ClassMethod NKGPopulated() As %Integer
{
    Return ($Data(^NKG) > 0)
}

}
