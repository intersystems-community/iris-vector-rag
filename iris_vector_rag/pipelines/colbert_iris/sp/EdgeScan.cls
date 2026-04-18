Class Graph.KG.EdgeScan Extends %RegisteredObject
{

ClassMethod MatchEdges(sourceId As %String, predicate As %String, shard As %Integer = 0) As %String [ SqlProc, SqlName = MatchEdges ]
{
	Set results = ##class(%DynamicArray).%New()
	Set sourceId = $Get(sourceId, "")
	Set predicate = $Get(predicate, "")

	If sourceId '= "" {
		If $Length(predicate) > 0 {
			Set o = ""
			For {
				Set o = $Order(^KG("out", 0, sourceId, predicate, o))
				Quit:o=""
				Set w = $Get(^KG("out", 0, sourceId, predicate, o), 1.0)
				Set entry = ##class(%DynamicObject).%New()
				Do entry.%Set("s", sourceId)
				Do entry.%Set("p", predicate)
				Do entry.%Set("o", o)
				Do entry.%Set("w", +w)
				Do results.%Push(entry)
			}
		} Else {
			Set p = ""
			For {
				Set p = $Order(^KG("out", 0, sourceId, p))
				Quit:p=""
				Set o = ""
				For {
					Set o = $Order(^KG("out", 0, sourceId, p, o))
					Quit:o=""
					Set w = $Get(^KG("out", 0, sourceId, p, o), 1.0)
					Set entry = ##class(%DynamicObject).%New()
					Do entry.%Set("s", sourceId)
					Do entry.%Set("p", p)
					Do entry.%Set("o", o)
					Do entry.%Set("w", +w)
					Do results.%Push(entry)
				}
			}
		}
	} Else {
		Set s = ""
		For {
			Set s = $Order(^KG("out", 0, s))
			Quit:s=""
			Set p = ""
			For {
				Set p = $Order(^KG("out", 0, s, p))
				Quit:p=""
				If ($Length(predicate) > 0) && (p '= predicate) Continue
				Set o = ""
				For {
					Set o = $Order(^KG("out", 0, s, p, o))
					Quit:o=""
					Set w = $Get(^KG("out", 0, s, p, o), 1.0)
					Set entry = ##class(%DynamicObject).%New()
					Do entry.%Set("s", s)
					Do entry.%Set("p", p)
					Do entry.%Set("o", o)
					Do entry.%Set("w", +w)
					Do results.%Push(entry)
				}
			}
		}
	}

	Return results.%ToJSON()
}

ClassMethod WriteAdjacency(s As %String, p As %String, o As %String, w As %String = "1.0")
{
	Set ^KG("out", 0, s, p, o) = +w
	Set ^KG("in", 0, o, p, s) = +w
}

ClassMethod DeleteAdjacency(s As %String, p As %String, o As %String)
{
	Kill ^KG("out", 0, s, p, o)
	Kill ^KG("in", 0, o, p, s)
}

}
