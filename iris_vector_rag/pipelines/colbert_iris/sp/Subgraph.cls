/// Bounded k-hop subgraph extraction over ^KG globals.
/// Pure ObjectScript -- no iris module dependency.
Class Graph.KG.Subgraph Extends %RegisteredObject
{

ClassMethod SubgraphJson(seedJson As %String, maxHops As %Integer = 2, edgeTypesJson As %String = "", maxNodes As %Integer = 10000) As %String
{
	If seedJson = "" Return "{""nodes"":[],""edges"":[],""properties"":{},""labels"":{}}"
	
	Try {
		Set seedArr = ##class(%DynamicArray).%FromJSON(seedJson)
	} Catch {
		Return "{""nodes"":[],""edges"":[],""properties"":{},""labels"":{}}"
	}
	If seedArr.%Size() = 0 Return "{""nodes"":[],""edges"":[],""properties"":{},""labels"":{}}"
	
	// Parse edge type filter
	Set hasFilter = 0
	If edgeTypesJson '= "" {
		Try {
			Set filterArr = ##class(%DynamicArray).%FromJSON(edgeTypesJson)
			If filterArr.%Size() > 0 {
				Set hasFilter = 1
				For fi = 0:1:(filterArr.%Size()-1) {
					Set allowedPred(filterArr.%Get(fi)) = ""
				}
			}
		} Catch {
		}
	}
	
	// Initialize BFS: frontier = seeds, seen = {}
	Set nodeCount = 0
	For i = 0:1:(seedArr.%Size()-1) {
		Set s = seedArr.%Get(i)
		// Accept seed if it has outgoing OR incoming edges in ^KG
		If $Data(^KG("deg", s)) || ($Order(^KG("in", 0, s, "")) '= "") {
			Set frontier(s) = ""
			If '$Data(seen(s)) {
				Set seen(s) = ""
				Set nodeCount = nodeCount + 1
			}
		}
	}
	
	// BFS: expand frontier hop by hop
	For hop = 1:1:maxHops {
		Kill nextFrontier
		Set node = ""
		For {
			Set node = $Order(frontier(node))
			Quit:node=""
			
			// Walk outgoing edges: ^KG("out", 0, node, predicate, object)
			Set p = ""
			For {
				Set p = $Order(^KG("out", 0, node, p))
				Quit:p=""
				
				// Edge type filter
				If hasFilter && '$Data(allowedPred(p)) Continue
				
				Set o = ""
				For {
					Set o = $Order(^KG("out", 0, node, p, o))
					Quit:o=""
					
					// Record edge
					Set edgeKey = node_$Char(1)_p_$Char(1)_o
					If '$Data(edgeSeen(edgeKey)) {
						Set edgeSeen(edgeKey) = ""
						Set edges($Increment(edgeCount)) = $ListBuild(node, p, o)
					}
					
					// Add neighbor to next frontier if not seen
					If '$Data(seen(o)) {
						If nodeCount >= maxNodes Quit
						Set seen(o) = ""
						Set nodeCount = nodeCount + 1
						Set nextFrontier(o) = ""
					}
				}
				If nodeCount >= maxNodes Quit
			}
			If nodeCount >= maxNodes Quit
		}
		
		// Swap frontiers
		Kill frontier
		Merge frontier = nextFrontier
		
		// Stop if no new nodes discovered or at limit
		Set check = $Order(frontier(""))
		If check = "" Quit
		If nodeCount >= maxNodes Quit
	}
	
	// Build result JSON
	Set nodesArr = ##class(%DynamicArray).%New()
	Set edgesArr = ##class(%DynamicArray).%New()
	Set propsObj = ##class(%DynamicObject).%New()
	Set labelsObj = ##class(%DynamicObject).%New()
	
	// Collect nodes + properties + labels
	Set n = ""
	For {
		Set n = $Order(seen(n))
		Quit:n=""
		Do nodesArr.%Push(n)
		
		// Properties: ^KG("prop", node, key) = val
		Set propObj = ##class(%DynamicObject).%New()
		Set hasProp = 0
		Set k = ""
		For {
			Set k = $Order(^KG("prop", n, k), 1, val)
			Quit:k=""
			Do propObj.%Set(k, val)
			Set hasProp = 1
		}
		If hasProp Do propsObj.%Set(n, propObj)
		
		// Labels: scan all known labels for this node
		// ^KG("label", labelName, nodeId) = ""
		Set labelArr = ##class(%DynamicArray).%New()
		Set hasLabel = 0
		&sql(DECLARE lblCursor CURSOR FOR SELECT label FROM Graph_KG.rdf_labels WHERE s = :n)
		&sql(OPEN lblCursor)
		For {
			&sql(FETCH lblCursor INTO :lbl)
			Quit:SQLCODE'=0
			Do labelArr.%Push(lbl)
			Set hasLabel = 1
		}
		&sql(CLOSE lblCursor)
		If hasLabel Do labelsObj.%Set(n, labelArr)
	}
	
	// Collect edges
	Set ei = ""
	For {
		Set ei = $Order(edges(ei), 1, edgeList)
		Quit:ei=""
		Set edgeObj = ##class(%DynamicObject).%New()
		Do edgeObj.%Set("s", $ListGet(edgeList, 1))
		Do edgeObj.%Set("p", $ListGet(edgeList, 2))
		Do edgeObj.%Set("o", $ListGet(edgeList, 3))
		Do edgesArr.%Push(edgeObj)
	}
	
	Set result = ##class(%DynamicObject).%New()
	Do result.%Set("nodes", nodesArr)
	Do result.%Set("edges", edgesArr)
	Do result.%Set("properties", propsObj)
	Do result.%Set("labels", labelsObj)
	
	Return result.%ToJSON()
}

/// PPR-guided subgraph extraction. Runs PPR then BFS within pruned node set.
/// All server-side over ^KG globals -- no Python bridge overhead.
/// alpha = teleport probability (NOT damping); damping = 1 - alpha.
ClassMethod PPRGuidedJson(seedJson As %String, alpha As %Double = 0.15, eps As %Double = 0.00001, topK As %Integer = 50, maxHops As %Integer = 5, edgeTypesJson As %String = "") As %String
{
	If seedJson = "" Return "{""nodes"":[],""edges"":[],""ppr_scores"":[],""seed_ids"":[],""nodes_before_pruning"":0,""nodes_after_pruning"":0}"
	
	Try {
		Set seedArr = ##class(%DynamicArray).%FromJSON(seedJson)
	} Catch {
		Return "{""nodes"":[],""edges"":[],""ppr_scores"":[],""seed_ids"":[],""nodes_before_pruning"":0,""nodes_after_pruning"":0}"
	}
	If seedArr.%Size() = 0 Return "{""nodes"":[],""edges"":[],""ppr_scores"":[],""seed_ids"":[],""nodes_before_pruning"":0,""nodes_after_pruning"":0}"
	
	// ── Step 1: Personalized PageRank ──
	Set damping = 1.0 - alpha
	Set nSeeds = seedArr.%Size()
	Set teleport = 1.0 / nSeeds
	For i = 0:1:(nSeeds - 1) {
		Set s = seedArr.%Get(i)
		Set seeds(s) = ""
		Set ranks(s) = teleport
	}
	
	For iter = 1:1:50 {
		Kill newRanks
		Set node = ""
		For {
			Set node = $Order(ranks(node), 1, rank)
			Quit:node=""
			Set deg = +$Get(^KG("deg", node))
			If deg = 0 {
				Set newRanks(node) = $Get(newRanks(node), 0) + (rank * (1 - damping))
				Continue
			}
			Set p = ""
			For {
				Set p = $Order(^KG("out", 0, node, p))
				Quit:p=""
				Set share = (damping * rank) / deg
				Set o = ""
				For {
					Set o = $Order(^KG("out", 0, node, p, o))
					Quit:o=""
					Set newRanks(o) = $Get(newRanks(o), 0) + share
				}
			}
		}
		Set teleportShare = (1 - damping) * teleport
		Set s = ""
		For {
			Set s = $Order(seeds(s))
			Quit:s=""
			Set newRanks(s) = $Get(newRanks(s), 0) + teleportShare
		}
		If iter > 2 {
			Set maxDelta = 0
			Set ck = ""
			For {
				Set ck = $Order(newRanks(ck), 1, newVal)
				Quit:ck=""
				Set delta = newVal - (+$Get(ranks(ck)))
				If delta < 0 Set delta = -delta
				If delta > maxDelta Set maxDelta = delta
			}
			If maxDelta < 0.0001 {
				Kill ranks
				Merge ranks = newRanks
				Quit
			}
		}
		Kill ranks
		Merge ranks = newRanks
	}
	
	// ── Step 2: Sort scores descending, apply eps threshold + topK cap ──
	Kill ^||PPRG
	Set nodesBefore = 0
	Set maxScore = 0
	Set node = ""
	For {
		Set node = $Order(ranks(node), 1, score)
		Quit:node=""
		Set nodesBefore = nodesBefore + 1
		If score > maxScore Set maxScore = score
		Set ^||PPRG(-score, node) = ""
	}
	
	Set threshold = eps * maxScore
	Set nodesAfter = 0
	Kill pruned
	Set negScore = ""
	For {
		Set negScore = $Order(^||PPRG(negScore))
		Quit:negScore=""
		Set n = ""
		For {
			Set n = $Order(^||PPRG(negScore, n))
			Quit:n=""
			If (-negScore) < threshold Quit
			If nodesAfter >= topK Quit
			Set pruned(n) = -negScore
			Set nodesAfter = nodesAfter + 1
		}
		If (-negScore) < threshold Quit
		If nodesAfter >= topK Quit
	}
	Kill ^||PPRG
	
	// ── Step 3: BFS from pruned set ──
	Set hasFilter = 0
	If edgeTypesJson '= "" {
		Try {
			Set filterArr = ##class(%DynamicArray).%FromJSON(edgeTypesJson)
			If filterArr.%Size() > 0 {
				Set hasFilter = 1
				For fi = 0:1:(filterArr.%Size()-1) {
					Set allowedPred(filterArr.%Get(fi)) = ""
				}
			}
		} Catch {
		}
	}
	
	Set bfsMax = topK * 2
	Set nodeCount = 0
	Set n = ""
	For {
		Set n = $Order(pruned(n))
		Quit:n=""
		If '$Data(seen(n)) {
			Set seen(n) = ""
			Set nodeCount = nodeCount + 1
			Set frontier(n) = ""
		}
	}
	
	For hop = 1:1:maxHops {
		Kill nextFrontier
		Set node = ""
		For {
			Set node = $Order(frontier(node))
			Quit:node=""
			Set p = ""
			For {
				Set p = $Order(^KG("out", 0, node, p))
				Quit:p=""
				If hasFilter && '$Data(allowedPred(p)) Continue
				Set o = ""
				For {
					Set o = $Order(^KG("out", 0, node, p, o))
					Quit:o=""
					Set edgeKey = node_$Char(1)_p_$Char(1)_o
					If '$Data(edgeSeen(edgeKey)) {
						Set edgeSeen(edgeKey) = ""
						Set edges($Increment(edgeCount)) = $ListBuild(node, p, o)
					}
					If '$Data(seen(o)) {
						If nodeCount >= bfsMax Quit
						Set seen(o) = ""
						Set nodeCount = nodeCount + 1
						Set nextFrontier(o) = ""
					}
				}
				If nodeCount >= bfsMax Quit
			}
			If nodeCount >= bfsMax Quit
		}
		Kill frontier
		Merge frontier = nextFrontier
		Set check = $Order(frontier(""))
		If check = "" Quit
		If nodeCount >= bfsMax Quit
	}
	
	// ── Step 4: Build JSON result ──
	Set nodesArr = ##class(%DynamicArray).%New()
	Set edgesArr = ##class(%DynamicArray).%New()
	Set scoresArr = ##class(%DynamicArray).%New()
	Set seedIdsArr = ##class(%DynamicArray).%FromJSON(seedJson)
	
	Set n = ""
	For {
		Set n = $Order(seen(n))
		Quit:n=""
		Do nodesArr.%Push(n)
	}
	
	Set ei = ""
	For {
		Set ei = $Order(edges(ei), 1, edgeList)
		Quit:ei=""
		Set eObj = ##class(%DynamicObject).%New()
		Do eObj.%Set("src", $ListGet(edgeList, 1))
		Do eObj.%Set("type", $ListGet(edgeList, 2))
		Do eObj.%Set("dst", $ListGet(edgeList, 3))
		Do edgesArr.%Push(eObj)
	}
	
	// PPR scores -- re-sort descending by score
	Kill ^||PPRG
	Set n = ""
	For {
		Set n = $Order(pruned(n), 1, sc)
		Quit:n=""
		Set ^||PPRG(-sc, n) = ""
	}
	Set negSc = ""
	For {
		Set negSc = $Order(^||PPRG(negSc))
		Quit:negSc=""
		Set sn = ""
		For {
			Set sn = $Order(^||PPRG(negSc, sn))
			Quit:sn=""
			Set scoreObj = ##class(%DynamicObject).%New()
			Do scoreObj.%Set("id", sn)
			Do scoreObj.%Set("score", -negSc, "number")
			Do scoresArr.%Push(scoreObj)
		}
	}
	Kill ^||PPRG
	
	Set result = ##class(%DynamicObject).%New()
	Do result.%Set("nodes", nodesArr)
	Do result.%Set("edges", edgesArr)
	Do result.%Set("ppr_scores", scoresArr)
	Do result.%Set("seed_ids", seedIdsArr)
	Do result.%Set("nodes_before_pruning", nodesBefore)
	Do result.%Set("nodes_after_pruning", nodesAfter)
	
	Return result.%ToJSON()
}

}
