Class Graph.KG.PLAIDSearch Extends %RegisteredObject
{

ClassMethod JsonToVector(jsonArray As %String) As %Binary
{
	Set arr = ##class(%DynamicArray).%FromJSON(jsonArray)
	Set dim = arr.%Size()
	Set vec = ""
	For i = 0:1:(dim-1) {
		Set $vector(vec, i+1, "double") = +arr.%Get(i)
	}
	Return vec
}

ClassMethod StoreCentroids(name As %String, centroidsJSON As %String)
{
	Set arr = ##class(%DynamicArray).%FromJSON(centroidsJSON)
	Set K = arr.%Size()
	For k = 0:1:(K-1) {
		Set cArr = arr.%Get(k)
		Set dim = cArr.%Size()
		Set vec = ""
		For i = 0:1:(dim-1) {
			Set $vector(vec, i+1, "double") = +cArr.%Get(i)
		}
		Set ^PLAID(name, "centroid", k) = vec
	}
	Set ^PLAID(name, "meta", "nCentroids") = K
	If K > 0 Set ^PLAID(name, "meta", "dim") = dim
}

ClassMethod StoreDocTokensBatch(name As %String, batchJSON As %String)
{
	Set batch = ##class(%DynamicArray).%FromJSON(batchJSON)
	Set nDocs = batch.%Size()
	For d = 0:1:(nDocs-1) {
		Set docObj = batch.%Get(d)
		Set docId = docObj.%Get("id")
		Set tokArr = docObj.%Get("tokens")
		Set nTok = tokArr.%Size()
		Set packed = ""
		For t = 0:1:(nTok-1) {
			Set tArr = tokArr.%Get(t)
			Set dim = tArr.%Size()
			Set vec = ""
			For i = 0:1:(dim-1) {
				Set $vector(vec, i+1, "double") = +tArr.%Get(i)
			}
			Set packed = packed _ $ListBuild(vec)
		}
		Set ^PLAID(name, "docPacked", docId) = packed
		Set ^PLAID(name, "docNTok", docId) = nTok
		Set ^PLAID(name, "meta", "nDocs") = $Get(^PLAID(name, "meta", "nDocs"), 0) + 1
		Set ^PLAID(name, "meta", "totalTokens") = $Get(^PLAID(name, "meta", "totalTokens"), 0) + nTok
	}
}

ClassMethod StoreDocTokens(name As %String, docId As %String, tokensJSON As %String)
{
	Set arr = ##class(%DynamicArray).%FromJSON(tokensJSON)
	Set nTok = arr.%Size()
	Set packed = ""
	For t = 0:1:(nTok-1) {
		Set tArr = arr.%Get(t)
		Set dim = tArr.%Size()
		Set vec = ""
		For i = 0:1:(dim-1) {
			Set $vector(vec, i+1, "double") = +tArr.%Get(i)
		}
		Set packed = packed _ $ListBuild(vec)
	}
	Set ^PLAID(name, "docPacked", docId) = packed
	Set ^PLAID(name, "docNTok", docId) = nTok
	Set ^PLAID(name, "meta", "nDocs") = $Get(^PLAID(name, "meta", "nDocs"), 0) + 1
	Set ^PLAID(name, "meta", "totalTokens") = $Get(^PLAID(name, "meta", "totalTokens"), 0) + nTok
}

ClassMethod BuildInvertedIndex(name As %String, assignmentsJSON As %String)
{
	Set arr = ##class(%DynamicArray).%FromJSON(assignmentsJSON)
	Set n = arr.%Size()
	For i = 0:1:(n-1) {
		Set item = arr.%Get(i)
		Set docId = item.%Get("docId")
		Set centroid = +item.%Get("centroid")
		Set ^PLAID(name, "docCentroid", centroid, docId) = ""
	}
}

ClassMethod Search(name As %String, queryTokensJSON As %String, k As %Integer = 10, nprobe As %Integer = 4) As %String
{
	Set tokens = ##class(%DynamicArray).%FromJSON(queryTokensJSON)
	Set nQ = tokens.%Size()
	Set nC = +$Get(^PLAID(name, "meta", "nCentroids"), 0)
	If nC = 0 Return "[]"
	
	For q = 0:1:(nQ-1) {
		Set qArr = tokens.%Get(q)
		Set dim = qArr.%Size()
		Set qVec = ""
		For i = 0:1:(dim-1) {
			Set $vector(qVec, i+1, "double") = +qArr.%Get(i)
		}
		Set queryVecs(q) = qVec
	}
	
	// Stage 1: centroid scoring
	Kill ^||centroidScore
	For q = 0:1:(nQ-1) {
		Set qVec = queryVecs(q)
		For c = 0:1:(nC-1) {
			Set cVec = ^PLAID(name, "centroid", c)
			Set dot = $vectorop("sum", $vectorop("v*", qVec, cVec))
			Set ^||centroidScore(c) = $Get(^||centroidScore(c), 0) + dot
		}
	}
	
	// Pick top nprobe centroids
	Kill ^||topCentroids
	For c = 0:1:(nC-1) {
		Set sc = +$Get(^||centroidScore(c), 0)
		Set ^||topCentroids(-sc, c) = ""
	}
	Kill ^||selectedCentroids
	Set count = 0, negSc = ""
	For {
		Set negSc = $Order(^||topCentroids(negSc))
		Quit:negSc=""
		Set cc = ""
		For {
			Set cc = $Order(^||topCentroids(negSc, cc))
			Quit:cc=""
			Set ^||selectedCentroids(cc) = ""
			Set count = count + 1
			If count >= nprobe Goto StageTwo
		}
	}
StageTwo
	
	// Stage 1.5: candidate docs from top centroids
	Kill ^||candidates
	Set sc = ""
	For {
		Set sc = $Order(^||selectedCentroids(sc))
		Quit:sc=""
		Set docId = ""
		For {
			Set docId = $Order(^PLAID(name, "docCentroid", sc, docId))
			Quit:docId=""
			Set ^||candidates(docId) = ""
		}
	}
	
	// Stage 2: MaxSim with packed token storage (1 $Get per doc)
	Kill ^||ranked
	Set docId = ""
	For {
		Set docId = $Order(^||candidates(docId))
		Quit:docId=""
		Set packed = $Get(^PLAID(name, "docPacked", docId))
		If packed = "" Continue
		Set nTok = +$Get(^PLAID(name, "docNTok", docId))
		Set maxsim = 0
		For q = 0:1:(nQ-1) {
			Set qVec = queryVecs(q)
			Set maxDot = -999999
			Set ptr = 0
			While $ListNext(packed, ptr, tVec) {
				Set dot = $vectorop("sum", $vectorop("v*", qVec, tVec))
				If dot > maxDot Set maxDot = dot
			}
			Set maxsim = maxsim + maxDot
		}
		Set ^||ranked(-maxsim, docId) = ""
	}
	
	// TopK
	Set result = "[", first = 1, cnt = 0
	Set negSc = ""
	For {
		Set negSc = $Order(^||ranked(negSc))
		Quit:negSc=""
		Set did = ""
		For {
			Set did = $Order(^||ranked(negSc, did))
			Quit:did=""
			Set:'first result = result _ ","
			Set first = 0
			Set result = result _ "{""id"":""" _ did _ """,""score"":" _ $FNumber(-negSc,"",8) _ "}"
			Set cnt = cnt + 1
			If cnt >= k Goto SearchDone
		}
	}
SearchDone
	Set result = result _ "]"
	Return result
}

ClassMethod Insert(name As %String, docId As %String, tokensJSON As %String)
{
	Set arr = ##class(%DynamicArray).%FromJSON(tokensJSON)
	Set nTok = arr.%Size()
	Set nC = +$Get(^PLAID(name, "meta", "nCentroids"), 0)
	Set packed = ""
	
	For t = 0:1:(nTok-1) {
		Set tArr = arr.%Get(t)
		Set dim = tArr.%Size()
		Set vec = ""
		For i = 0:1:(dim-1) {
			Set $vector(vec, i+1, "double") = +tArr.%Get(i)
		}
		Set packed = packed _ $ListBuild(vec)
		
		Set bestC = 0, bestDot = -999999
		For c = 0:1:(nC-1) {
			Set cVec = ^PLAID(name, "centroid", c)
			Set dot = $vectorop("sum", $vectorop("v*", vec, cVec))
			If dot > bestDot Set bestDot = dot, bestC = c
		}
		Set ^PLAID(name, "docCentroid", bestC, docId) = ""
	}
	Set ^PLAID(name, "docPacked", docId) = packed
	Set ^PLAID(name, "docNTok", docId) = nTok
	Set ^PLAID(name, "meta", "nDocs") = $Get(^PLAID(name, "meta", "nDocs"), 0) + 1
	Set ^PLAID(name, "meta", "totalTokens") = $Get(^PLAID(name, "meta", "totalTokens"), 0) + nTok
}

ClassMethod Info(name As %String) As %String
{
	Set nC = +$Get(^PLAID(name, "meta", "nCentroids"), 0)
	Set nD = +$Get(^PLAID(name, "meta", "nDocs"), 0)
	Set dim = +$Get(^PLAID(name, "meta", "dim"), 0)
	Set nT = +$Get(^PLAID(name, "meta", "totalTokens"), 0)
	Return "{""name"":"""_name_""",""nCentroids"":"_nC_",""nDocs"":"_nD_",""dim"":"_dim_",""totalTokens"":"_nT_"}"
}

ClassMethod Drop(name As %String)
{
	Kill ^PLAID(name)
}

}
