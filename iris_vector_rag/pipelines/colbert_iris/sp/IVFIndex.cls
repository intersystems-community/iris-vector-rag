Class Graph.KG.IVFIndex Extends %RegisteredObject
{

ClassMethod Build(name As %String, nlistJSON As %String, metricJSON As %String, centroidsJSON As %String, assignmentsJSON As %String) As %String
{
	Kill ^IVF(name)

	Set nlist = +nlistJSON
	Set metric = $Extract(metricJSON, 2, $Length(metricJSON)-1)
	Set centroids = ##class(%DynamicArray).%FromJSON(centroidsJSON)
	Set assignments = ##class(%DynamicArray).%FromJSON(assignmentsJSON)

	Set nC = centroids.%Size()
	Set dim = 0

	For k = 0:1:(nC-1) {
		Set cArr = centroids.%Get(k)
		Set cDim = cArr.%Size()
		If dim = 0 Set dim = cDim
		Set vec = ""
		For i = 0:1:(cDim-1) {
			Set $vector(vec, i+1, "double") = +cArr.%Get(i)
		}
		Set ^IVF(name, "centroid", k) = vec
	}

	Set nAssign = assignments.%Size()
	For i = 0:1:(nAssign-1) {
		Set item = assignments.%Get(i)
		Set nodeId = item.%Get("nodeId")
		Set cellIdx = +item.%Get("cellIdx")
		Set b64 = item.%Get("vec")
		Set floatList = ##class(%DynamicArray).%FromJSON(b64)
		Set fDim = floatList.%Size()
		Set vec = ""
		For j = 0:1:(fDim-1) {
			Set $vector(vec, j+1, "double") = +floatList.%Get(j)
		}
		Set ^IVF(name, "list", cellIdx, nodeId) = vec
	}

	Set ^IVF(name, "cfg", "nlist") = nlist
	Set ^IVF(name, "cfg", "dim") = dim
	Set ^IVF(name, "cfg", "metric") = metric
	Set ^IVF(name, "cfg", "indexed") = nAssign

	Set result = "{""nlist"":"_nlist_",""dim"":"_dim_",""metric"":"""_metric_""",""indexed"":"_nAssign_"}"
	Quit result
}

ClassMethod Search(name As %String, queryVecB64 As %String, k As %Integer = 10, nprobe As %Integer = 8) As %String
{
	Set nlist = +$Get(^IVF(name, "cfg", "nlist"), 0)
	If nlist = 0 Quit "[]"

	Set metric = $Get(^IVF(name, "cfg", "metric"), "cosine")

	If nprobe > nlist Set nprobe = nlist

	Set qArr = ##class(%DynamicArray).%FromJSON(queryVecB64)
	Set qDim = qArr.%Size()
	Set qVec = ""
	For qi = 0:1:(qDim-1) {
		Set $vector(qVec, qi+1, "double") = +qArr.%Get(qi)
	}

	Kill ^||ivfCentScore
	For c = 0:1:(nlist-1) {
		Set cVec = $Get(^IVF(name, "centroid", c))
		If cVec = "" Continue
		If metric = "cosine" {
			Set sc = $vectorop("sum", $vectorop("v*", qVec, cVec))
		} ElseIf metric = "dot" {
			Set sc = $vectorop("sum", $vectorop("v*", qVec, cVec))
		} Else {
			Set diff = $vectorop("v-", qVec, cVec)
			Set sc = -$vectorop("sum", $vectorop("v*", diff, diff))
		}
		Set ^||ivfCentScore(-sc, c) = ""
	}

	Kill ^||ivfSelected
	Set count = 0, negSc = ""
	For {
		Set negSc = $Order(^||ivfCentScore(negSc))
		Quit:negSc=""
		Set cc = ""
		For {
			Set cc = $Order(^||ivfCentScore(negSc, cc))
			Quit:cc=""
			Set ^||ivfSelected(cc) = ""
			Set count = count + 1
			If count >= nprobe Goto ScanCells
		}
	}
ScanCells

	Kill ^||ivfScores
	Set sc = ""
	For {
		Set sc = $Order(^||ivfSelected(sc))
		Quit:sc=""
		Set nodeId = ""
		For {
			Set nodeId = $Order(^IVF(name, "list", sc, nodeId))
			Quit:nodeId=""
			Set dVec = ^IVF(name, "list", sc, nodeId)
			If metric = "cosine" {
				Set sim = $vectorop("sum", $vectorop("v*", qVec, dVec))
			} ElseIf metric = "dot" {
				Set sim = $vectorop("sum", $vectorop("v*", qVec, dVec))
			} Else {
				Set diff = $vectorop("v-", qVec, dVec)
				Set sim = -$vectorop("sum", $vectorop("v*", diff, diff))
			}
			If '$Data(^||ivfScores(nodeId)) {
				Set ^||ivfScores(nodeId) = sim
			} ElseIf sim > ^||ivfScores(nodeId) {
				Set ^||ivfScores(nodeId) = sim
			}
		}
	}

	Kill ^||ivfRanked
	Set nodeId = ""
	For {
		Set nodeId = $Order(^||ivfScores(nodeId))
		Quit:nodeId=""
		Set ^||ivfRanked(-^||ivfScores(nodeId), nodeId) = ""
	}

	Set results = ##class(%DynamicArray).%New()
	Set count = 0, negSim = ""
	For {
		Set negSim = $Order(^||ivfRanked(negSim))
		Quit:negSim=""
		Quit:count>=k
		Set nodeId = ""
		For {
			Set nodeId = $Order(^||ivfRanked(negSim, nodeId))
			Quit:nodeId=""
			Quit:count>=k
			Set sim = -negSim
			Set entry = ##class(%DynamicObject).%New()
			Do entry.%Set("id", nodeId)
			Do entry.%Set("score", sim)
			Do results.%Push(entry)
			Set count = count + 1
		}
	}

	Kill ^||ivfCentScore, ^||ivfSelected, ^||ivfScores, ^||ivfRanked
	Quit results.%ToJSON()
}

ClassMethod Drop(name As %String)
{
	Kill ^IVF(name)
}

ClassMethod Info(name As %String) As %String
{
	If '$Data(^IVF(name, "cfg")) Quit "{}"
	Set nlist = +$Get(^IVF(name, "cfg", "nlist"), 0)
	Set dim = +$Get(^IVF(name, "cfg", "dim"), 0)
	Set metric = $Get(^IVF(name, "cfg", "metric"), "cosine")
	Set indexed = +$Get(^IVF(name, "cfg", "indexed"), 0)
	Quit "{""nlist"":"_nlist_",""dim"":"_dim_",""metric"":"""_metric_""",""indexed"":"_indexed_"}"
}

ClassMethod SearchProc(name As %String, queryVecB64 As %String, k As %Integer = 10, nprobe As %Integer = 8) As %String [ SqlProc, SqlName = kg_IVF ]
{
	Quit ..Search(name, queryVecB64, k, nprobe)
}

}
