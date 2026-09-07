/// Graph.KG.EdgeScan — adjacency-global maintenance + scan procedures.
///
/// ============================================================================
/// ^KG ADJACENCY LAYOUT OF RECORD (spec 214 — named-graph aware)
/// ============================================================================
///
///     ^KG("out", graph, s, p, o) = weight
///     ^KG("in",  graph, o, p, s) = weight
///     ^KG("deg", graph, s)           = out-degree
///     ^KG("degp", graph, s, p)       = per-predicate out-degree
///
/// graph = "" (empty string) = default graph (FR-003 sentinel).
/// graph = "umls" etc. = named graph.
/// Merged-view traversal iterates the full ^KG("out") tree (all graph subscripts).
/// Graph-filtered traversal uses a fixed graph subscript.
///
/// ^NKG is per-identifier (no graph dimension); InternNode unchanged.
/// ============================================================================
Class Graph.KG.EdgeScan Extends %RegisteredObject
{

/// Scan adjacency globals. graph="" returns merged view across all graphs.
ClassMethod MatchEdges(sourceId As %String, predicate As %String, shard As %Integer = 0, graph = 0) As %String [ SqlProc, SqlName = MatchEdges ]
{
	Set results = ##class(%DynamicArray).%New()
	Set sourceId = $Get(sourceId, "")
	Set predicate = $Get(predicate, "")

	// Build the list of graph subscripts to iterate
	Kill tGraphs
	// graph=0 means merged view (all graphs); non-0 = specific named graph
	If (graph '= 0) && (graph '= "") {
		Set tGraphs(graph) = ""
	} Else {
		Set g = ""
		For {
			Set g = $Order(^KG("out", g))
			Quit:g=""
			Set tGraphs(g) = ""
		}
	}

	Set g = ""
	For {
		Set g = $Order(tGraphs(g))
		Quit:g=""
		If sourceId '= "" {
			If $Length(predicate) > 0 {
				Set o = ""
				For {
					Set o = $Order(^KG("out", g, sourceId, predicate, o))
					Quit:o=""
					Set w = $Get(^KG("out", g, sourceId, predicate, o), 1.0)
					Set entry = ##class(%DynamicObject).%New()
					Do entry.%Set("s", sourceId) Do entry.%Set("p", predicate) Do entry.%Set("o", o) Do entry.%Set("w", +w)
					Do results.%Push(entry)
				}
			} Else {
				Set p = ""
				For {
					Set p = $Order(^KG("out", g, sourceId, p))
					Quit:p=""
					Set o = ""
					For {
						Set o = $Order(^KG("out", g, sourceId, p, o))
						Quit:o=""
						Set w = $Get(^KG("out", g, sourceId, p, o), 1.0)
						Set entry = ##class(%DynamicObject).%New()
						Do entry.%Set("s", sourceId) Do entry.%Set("p", p) Do entry.%Set("o", o) Do entry.%Set("w", +w)
						Do results.%Push(entry)
					}
				}
			}
		} Else {
			Set s = ""
			For {
				Set s = $Order(^KG("out", g, s))
				Quit:s=""
				Set p = ""
				For {
					Set p = $Order(^KG("out", g, s, p))
					Quit:p=""
					If ($Length(predicate) > 0) && (p '= predicate) Continue
					Set o = ""
					For {
						Set o = $Order(^KG("out", g, s, p, o))
						Quit:o=""
						Set w = $Get(^KG("out", g, s, p, o), 1.0)
						Set entry = ##class(%DynamicObject).%New()
						Do entry.%Set("s", s) Do entry.%Set("p", p) Do entry.%Set("o", o) Do entry.%Set("w", +w)
						Do results.%Push(entry)
					}
				}
			}
		}
	}

	Return results.%ToJSON()
}

/// Write adjacency entries for one edge. graph="" = default graph (spec 214).
ClassMethod WriteAdjacency(s As %String, p As %String, o As %String, w As %String = "1.0", graph = 0)
{
	Set ^KG("out", graph, s, p, o) = +w
	Set ^KG("in", graph, o, p, s) = +w
	Set tmp = $Increment(^KG("deg", graph, s))
	Set tmp = $Increment(^KG("degp", graph, s, p))
	// ^NKG is per-identifier (no graph dimension); unchanged by spec 214
	If $Data(^NKG("$meta", "nodeCount")) {
		Set sIdx = ##class(Graph.KG.GraphIndex).InternNode(s)
		Set oIdx = ##class(Graph.KG.GraphIndex).InternNode(o)
		Set pIdx = ##class(Graph.KG.GraphIndex).InternLabel(p)
		If '$Data(^NKG(-1, sIdx, -(pIdx+1), oIdx)) {
			Set ^NKG(-1, sIdx, -(pIdx+1), oIdx) = +w
			Set ^NKG(-2, oIdx, -(pIdx+1), sIdx) = +w
			Set tmp = $Increment(^NKG(-3, sIdx))
			Set tmp = $Increment(^NKG("$meta", "edgeCount"))
		}
		Set tmp = $Increment(^NKG("$meta", "version"))
		Do ##class(Graph.KG.GraphIndex).UpdateStructuralHLL(sIdx, pIdx, oIdx)
	}
}

/// Delete adjacency entries for one edge. graph="" = default graph (spec 214).
ClassMethod DeleteAdjacency(s As %String, p As %String, o As %String, graph = 0)
{
	Kill ^KG("out", graph, s, p, o)
	Kill ^KG("in", graph, o, p, s)
	If $Data(^NKG("$meta", "nodeCount")) {
		Do ##class(Graph.KG.GraphIndex).DeleteIndex("", s, p, o, "")
	}
}

/// Bulk ingest edges via SQL INSERT + persistent ^KG global writes.
ClassMethod BulkIngestEdgesSQL(edgesJson As %String, predicate As %String = "KNOWS") As %Integer
{
	Set data = {}.%FromJSON(edgesJson)
	Set n = 0
	TSTART
	Set iter = data.%GetIterator()
	While iter.%GetNext(.idx, .edge) {
		Set s = edge.s
		Set p = $Select(edge.p'="":edge.p, 1:predicate)
		Set o = edge.o
		If (s = "") || (o = "") { Continue }
		If s = o { Continue }
		&sql(INSERT INTO Graph_KG.rdf_edges (s, p, o_id) VALUES (:s, :p, :o))
		If (SQLCODE '= 0) && (SQLCODE '= -119) { Continue }
		Do ..WriteAdjacency(s, p, o, "1.0", 0)
		Set n = n + 1
	}
	TCOMMIT
	Quit n
}

ClassMethod BulkIngestEdgesFile(filePath As %String, predicate As %String = "KNOWS") As %Integer
{
	Set stream = ##class(%Stream.FileCharacter).%New()
	Do stream.LinkToFile(filePath)
	Set json = ""
	While 'stream.AtEnd {
		Set json = json _ stream.Read(32000)
	}
	Quit ..BulkIngestEdgesSQL(json, predicate)
}

ClassMethod BulkIngestNodesSQL(nodesJson As %String) As %Integer
{
	Set data = {}.%FromJSON(nodesJson)
	Set n = 0
	TSTART
	Set iter = data.%GetIterator()
	While iter.%GetNext(.idx, .node) {
		Set id = node.id
		If id = "" { Continue }
		&sql(INSERT INTO Graph_KG.nodes (node_id, graph_id) VALUES (:id, ''))
		If (SQLCODE '= 0) && (SQLCODE '= -119) { Continue }
		If $IsObject(node.labels) {
			Set liter = node.labels.%GetIterator()
			While liter.%GetNext(.li, .label) {
				If label = "" { Continue }
				&sql(INSERT INTO Graph_KG.rdf_labels (s, label) VALUES (:id, :label))
			}
		}
		If $IsObject(node.props) {
			Set piter = node.props.%GetIterator()
			While piter.%GetNext(.key, .val) {
				If key = "" { Continue }
				Try {
					Set valStr = $Select($IsObject(val):val.%ToJSON(), $IsValidNum(val):val, 1:""""_$Replace($ZConvert(val,"O","UTF8"),"""","""""")_"""")
					&sql(INSERT INTO Graph_KG.rdf_props (s, "key", val) VALUES (:id, :key, :valStr))
				} Catch ex { }
			}
		}
		Set n = n + 1
	}
	TCOMMIT
	Quit n
}

/// Remove ^KG adjacency entries for each node in JSON array (merged view).
ClassMethod BulkDeleteAdjacency(nodeIdsJSON As %String) As %Integer
{
	Set arr = ##class(%DynamicArray).%FromJSON(nodeIdsJSON)
	Set count = 0
	Set iter = arr.%GetIterator()
	While iter.%GetNext(.idx, .nodeId) {
		If nodeId = "" { Continue }
		// Kill all graph subscripts for this node as source
		Set g = ""
		For {
			Set g = $Order(^KG("out", g))
			Quit:g=""
			Kill ^KG("out", g, nodeId)
			Kill ^KG("deg", g, nodeId)
		}
		// Kill all graph subscripts for this node as target
		Set g = ""
		For {
			Set g = $Order(^KG("in", g))
			Quit:g=""
			Kill ^KG("in", g, nodeId)
		}
		Set count = count + 1
	}
	Quit count
}

}
