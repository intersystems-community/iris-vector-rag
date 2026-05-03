/// Graph analytics kernels: WCC, CDLP.
/// Pure ObjectScript over ^KG globals.
Class Graph.KG.Algorithms Extends %RegisteredObject
{

ClassMethod WCCJson(maxIter As %Integer = 100) As %String
{
	Set nodeCount = 0
	Set node = ""
	For {
		Set node = $Order(^KG("deg", node))
		Quit:node=""
		Set labels(node) = node
		Set nodeCount = nodeCount + 1
	}
	// Also include nodes that only appear as targets (no outgoing edges = no "deg" entry)
	Set node = ""
	For {
		Set node = $Order(^KG("in", 0, node))
		Quit:node=""
		If '$Data(labels(node)) {
			Set labels(node) = node
			Set nodeCount = nodeCount + 1
		}
	}
	If nodeCount = 0 Return "{}"
	
	For iter = 1:1:maxIter {
		Set changed = 0
		Kill newLabels
		
		Set node = ""
		For {
			Set node = $Order(labels(node), 1, myLabel)
			Quit:node=""
			Set minLabel = myLabel
			
			// Check outgoing neighbors
			Set p = ""
			For {
				Set p = $Order(^KG("out", 0, node, p))
				Quit:p=""
				Set o = ""
				For {
					Set o = $Order(^KG("out", 0, node, p, o))
					Quit:o=""
					Set nbrLabel = $Get(labels(o), o)
					If minLabel ]] nbrLabel Set minLabel = nbrLabel
				}
			}
			
			// Check incoming neighbors
			Set p = ""
			For {
				Set p = $Order(^KG("in", 0, node, p))
				Quit:p=""
				Set s = ""
				For {
					Set s = $Order(^KG("in", 0, node, p, s))
					Quit:s=""
					Set nbrLabel = $Get(labels(s), s)
					If minLabel ]] nbrLabel Set minLabel = nbrLabel
				}
			}
			
			If minLabel '= myLabel Set changed = changed + 1
			Set newLabels(node) = minLabel
		}
		
		Kill labels
		Merge labels = newLabels
		If changed = 0 Quit
	}
	
	Set result = ##class(%DynamicObject).%New()
	Set node = ""
	For {
		Set node = $Order(labels(node), 1, lbl)
		Quit:node=""
		Do result.%Set(node, lbl)
	}
	Return result.%ToJSON()
}

ClassMethod CDLPJson(maxIter As %Integer = 10) As %String
{
	Set nodeCount = 0
	Set node = ""
	For {
		Set node = $Order(^KG("deg", node))
		Quit:node=""
		Set labels(node) = node
		Set nodeCount = nodeCount + 1
	}
	Set node = ""
	For {
		Set node = $Order(^KG("in", 0, node))
		Quit:node=""
		If '$Data(labels(node)) {
			Set labels(node) = node
			Set nodeCount = nodeCount + 1
		}
	}
	If nodeCount = 0 Return "{}"
	
	For iter = 1:1:maxIter {
		Set changed = 0
		Kill newLabels
		
		Set node = ""
		For {
			Set node = $Order(labels(node), 1, myLabel)
			Quit:node=""
			Kill counts
			
			// Count outgoing neighbor labels
			Set p = ""
			For {
				Set p = $Order(^KG("out", 0, node, p))
				Quit:p=""
				Set o = ""
				For {
					Set o = $Order(^KG("out", 0, node, p, o))
					Quit:o=""
					Set nbrLabel = $Get(labels(o), o)
					Set counts(nbrLabel) = $Get(counts(nbrLabel), 0) + 1
				}
			}
			
			// Count incoming neighbor labels
			Set p = ""
			For {
				Set p = $Order(^KG("in", 0, node, p))
				Quit:p=""
				Set s = ""
				For {
					Set s = $Order(^KG("in", 0, node, p, s))
					Quit:s=""
					Set nbrLabel = $Get(labels(s), s)
					Set counts(nbrLabel) = $Get(counts(nbrLabel), 0) + 1
				}
			}
			
			// Find most frequent label (ties: smallest label wins)
			Set bestLabel = myLabel
			Set bestCount = 0
			Set lbl = ""
			For {
				Set lbl = $Order(counts(lbl), 1, cnt)
				Quit:lbl=""
				If (cnt > bestCount) || ((cnt = bestCount) && (lbl < bestLabel)) {
					Set bestLabel = lbl
					Set bestCount = cnt
				}
			}
			
			If bestLabel '= myLabel Set changed = changed + 1
			Set newLabels(node) = bestLabel
		}
		
		Kill labels
		Merge labels = newLabels
		If changed = 0 Quit
	}
	
	Set result = ##class(%DynamicObject).%New()
	Set node = ""
	For {
		Set node = $Order(labels(node), 1, lbl)
		Quit:node=""
		Do result.%Set(node, lbl)
	}
	Return result.%ToJSON()
}

}
