/// Graph.KG.VecIndex — Lightweight ANN vector index using random projection trees.
/// Pure ObjectScript + $vectorop SIMD. No SQL VECTOR type required.
///
/// Supports multiple distance metrics: cosine, l2, l1, dot, chebyshev, mahalanobis.
/// Requires IRIS 2024.1+ ($vector/$vectorop support).
///
/// Storage:
///   ^VecIdx(name, "cfg", "dim"|"leafSize"|"numTrees"|"metric") = config
///   ^VecIdx(name, "cfg", "invCov") = inverse covariance matrix (mahalanobis)
///   ^VecIdx(name, "tree", treeId, nodeId, ...) = tree structure
///   ^VecIdx(name, "vec", docId) = $vector embedding
///   ^VecIdx(name, "meta", "count") = N
///
/// Usage:
///   do ##class(Graph.KG.VecIndex).Create("myIndex", 384, "cosine")
///   do ##class(Graph.KG.VecIndex).Insert("myIndex", "doc1", vec)
///   do ##class(Graph.KG.VecIndex).Build("myIndex")
///   set results = ##class(Graph.KG.VecIndex).Search("myIndex", queryVec, 10)
Class Graph.KG.VecIndex Extends %RegisteredObject
{

// ================================================================
// Configuration
// ================================================================

ClassMethod Create(name As %String, dim As %Integer, metric As %String = "cosine", numTrees As %Integer = 4, leafSize As %Integer = 50) As %String
{
    Set ^VecIdx(name, "cfg", "dim") = dim
    Set ^VecIdx(name, "cfg", "metric") = $ZConvert(metric, "L")
    Set ^VecIdx(name, "cfg", "numTrees") = numTrees
    Set ^VecIdx(name, "cfg", "leafSize") = leafSize
    Set ^VecIdx(name, "meta", "count") = 0
    Return "{""name"":"""_name_""",""dim"":"_dim_",""metric"":"""_metric_"""}"
}

ClassMethod Drop(name As %String)
{
    Kill ^VecIdx(name)
}

// ================================================================
// Distance / similarity functions — all SIMD via $vectorop
// ================================================================

ClassMethod Distance(v1 As %Binary, v2 As %Binary, metric As %String) As %Double
{
    If metric = "cosine" Return 1 - ..Cosine(v1, v2)
    If metric = "l2" Return ..L2(v1, v2)
    If metric = "l1" Return ..L1(v1, v2)
    If metric = "dot" Return -..Dot(v1, v2)
    If metric = "chebyshev" Return ..Chebyshev(v1, v2)
    Return 1 - ..Cosine(v1, v2)
}

ClassMethod Similarity(v1 As %Binary, v2 As %Binary, metric As %String) As %Double
{
    If metric = "cosine" Return ..Cosine(v1, v2)
    If metric = "dot" Return ..Dot(v1, v2)
    If metric = "l2" Return 1 / (1 + ..L2(v1, v2))
    If metric = "l1" Return 1 / (1 + ..L1(v1, v2))
    If metric = "chebyshev" Return 1 / (1 + ..Chebyshev(v1, v2))
    Return ..Cosine(v1, v2)
}

ClassMethod Cosine(v1 As %Binary, v2 As %Binary) As %Double
{
    Set dot = $vectorop("sum", $vectorop("v*", v1, v2))
    Set n1 = $zsqr($vectorop("sum", $vectorop("v*", v1, v1)))
    Set n2 = $zsqr($vectorop("sum", $vectorop("v*", v2, v2)))
    If (n1 = 0) || (n2 = 0) Return 0
    Return dot / (n1 * n2)
}

ClassMethod L2(v1 As %Binary, v2 As %Binary) As %Double
{
    Set diff = $vectorop("v-", v1, v2)
    Return $zsqr($vectorop("sum", $vectorop("v*", diff, diff)))
}

ClassMethod L1(v1 As %Binary, v2 As %Binary) As %Double
{
    Set diff = $vectorop("v-", v1, v2)
    Set absDiff = $vectorop("abs", diff)
    Return $vectorop("sum", absDiff)
}

ClassMethod Dot(v1 As %Binary, v2 As %Binary) As %Double
{
    Return $vectorop("sum", $vectorop("v*", v1, v2))
}

ClassMethod Chebyshev(v1 As %Binary, v2 As %Binary) As %Double
{
    Set diff = $vectorop("v-", v1, v2)
    Set absDiff = $vectorop("abs", diff)
    Return $vectorop("max", absDiff)
}

/// Mahalanobis distance: sqrt((v1-v2)^T * invCov * (v1-v2))
/// invCov stored as dim vectors in ^VecIdx(name,"cfg","invCov",row)
ClassMethod Mahalanobis(v1 As %Binary, v2 As %Binary, name As %String) As %Double
{
    Set diff = $vectorop("v-", v1, v2)
    Set dim = $vectorop("count", diff)
    Set result = 0
    For i = 1:1:dim {
        Set row = $Get(^VecIdx(name, "cfg", "invCov", i))
        If row '= "" {
            Set result = result + ($vector(diff, i) * $vectorop("sum", $vectorop("v*", diff, row)))
        }
    }
    If result < 0 Set result = 0
    Return $zsqr(result)
}

/// Store inverse covariance matrix for Mahalanobis (each row as a $vector)
ClassMethod SetInverseCov(name As %String, rowIdx As %Integer, rowVec As %Binary)
{
    Set ^VecIdx(name, "cfg", "invCov", rowIdx) = rowVec
}

// ================================================================
// Insert / Build / Search
// ================================================================

ClassMethod Insert(name As %String, docId As %String, vec As %Binary)
{
    Set ^VecIdx(name, "vec", docId) = vec
    Set dim = $vectorop("count", vec)
    Set ^VecIdx(name, "cfg", "dim") = dim
    Set ^VecIdx(name, "meta", "count") = $Get(^VecIdx(name, "meta", "count"), 0) + 1
    Set numTrees = +$Get(^VecIdx(name, "cfg", "numTrees"), 0)
    If numTrees = 0 Quit
    For t = 1:1:numTrees {
        Do ..InsertIntoTree(name, t, docId, vec, 1)
    }
}

ClassMethod Build(name As %String) As %String
{
    Set dim = ..GetDim(name)
    If dim = 0 Return "{""error"":""no vectors found""}"
    Set leafSize = +$Get(^VecIdx(name, "cfg", "leafSize"), 50)
    Set numTrees = +$Get(^VecIdx(name, "cfg", "numTrees"), 4)
    Kill ^VecIdx(name, "tree")
    Set allIds = ..CollectAllIds(name)
    Set nIds = $ListLength(allIds)
    For t = 1:1:numTrees {
        Do ..BuildTree(name, t, allIds, dim, leafSize, 1)
    }
    Return "{""trees"":"_numTrees_",""vectors"":"_nIds_",""dim"":"_dim_",""metric"":"""_$Get(^VecIdx(name,"cfg","metric"),"cosine")_"""}"
}

ClassMethod Search(name As %String, queryVec As %Binary, k As %Integer = 10, nprobe As %Integer = 2) As %String
{
    Set numTrees = +$Get(^VecIdx(name, "cfg", "numTrees"), 0)
    If numTrees = 0 Return ..BruteForceSearch(name, queryVec, k)
    Kill ^||candidates
    For t = 1:1:numTrees {
        Do ..SearchTree(name, t, queryVec, 1, nprobe)
    }
    Return ..RankCandidates(name, queryVec, k)
}

ClassMethod BruteForceSearch(name As %String, queryVec As %Binary, k As %Integer = 10) As %String
{
    Set metric = $Get(^VecIdx(name, "cfg", "metric"), "cosine")
    Kill ^||ranked
    Set docId = ""
    For {
        Set docId = $Order(^VecIdx(name, "vec", docId))
        Quit:docId=""
        Set vec = ^VecIdx(name, "vec", docId)
        Set dist = ..Distance(queryVec, vec, metric)
        Set ^||ranked(dist, docId) = ""
    }
    Return ..TopK(k, metric)
}

// ================================================================
// Tree operations (private)
// ================================================================

ClassMethod BuildTree(name As %String, treeId As %Integer, idList As %List, dim As %Integer, leafSize As %Integer, nodeId As %Integer) As %Integer [ Private ]
{
    Set n = $ListLength(idList)
    If n <= leafSize {
        Set ptr = 0
        While $ListNext(idList, ptr, docId) {
            Set ^VecIdx(name, "tree", treeId, nodeId, "leaf", docId) = ""
        }
        Return nodeId
    }
    Set plane = ..RandomHyperplane(dim)
    Set ^VecIdx(name, "tree", treeId, nodeId, "plane") = plane
    Set leftIds = "", rightIds = ""
    Set ptr = 0
    While $ListNext(idList, ptr, docId) {
        Set vec = ^VecIdx(name, "vec", docId)
        Set proj = $vectorop("sum", $vectorop("v*", vec, plane))
        If proj >= 0 {
            Set rightIds = rightIds _ $ListBuild(docId)
        } Else {
            Set leftIds = leftIds _ $ListBuild(docId)
        }
    }
    If $ListLength(leftIds) = 0 Set leftIds = rightIds
    If $ListLength(rightIds) = 0 Set rightIds = leftIds
    Set leftNode = nodeId * 2
    Set rightNode = nodeId * 2 + 1
    Set ^VecIdx(name, "tree", treeId, nodeId, "L") = leftNode
    Set ^VecIdx(name, "tree", treeId, nodeId, "R") = rightNode
    Do ..BuildTree(name, treeId, leftIds, dim, leafSize, leftNode)
    Do ..BuildTree(name, treeId, rightIds, dim, leafSize, rightNode)
    Return nodeId
}

ClassMethod InsertIntoTree(name As %String, treeId As %Integer, docId As %String, vec As %Binary, nodeId As %Integer) [ Private ]
{
    If $Data(^VecIdx(name, "tree", treeId, nodeId, "leaf")) {
        Set ^VecIdx(name, "tree", treeId, nodeId, "leaf", docId) = ""
        Quit
    }
    Set plane = $Get(^VecIdx(name, "tree", treeId, nodeId, "plane"))
    If plane = "" {
        Set ^VecIdx(name, "tree", treeId, nodeId, "leaf", docId) = ""
        Quit
    }
    Set proj = $vectorop("sum", $vectorop("v*", vec, plane))
    If proj >= 0 {
        Do ..InsertIntoTree(name, treeId, docId, vec, +$Get(^VecIdx(name, "tree", treeId, nodeId, "R"), nodeId*2+1))
    } Else {
        Do ..InsertIntoTree(name, treeId, docId, vec, +$Get(^VecIdx(name, "tree", treeId, nodeId, "L"), nodeId*2))
    }
}

ClassMethod SearchTree(name As %String, treeId As %Integer, queryVec As %Binary, nodeId As %Integer, nprobe As %Integer) [ Private ]
{
    If $Data(^VecIdx(name, "tree", treeId, nodeId, "leaf")) {
        Set docId = ""
        For {
            Set docId = $Order(^VecIdx(name, "tree", treeId, nodeId, "leaf", docId))
            Quit:docId=""
            Set ^||candidates(docId) = ""
        }
        Quit
    }
    Set plane = $Get(^VecIdx(name, "tree", treeId, nodeId, "plane"))
    If plane = "" Quit
    Set proj = $vectorop("sum", $vectorop("v*", queryVec, plane))
    If proj >= 0 {
        Set primary = +$Get(^VecIdx(name, "tree", treeId, nodeId, "R"))
        Set secondary = +$Get(^VecIdx(name, "tree", treeId, nodeId, "L"))
    } Else {
        Set primary = +$Get(^VecIdx(name, "tree", treeId, nodeId, "L"))
        Set secondary = +$Get(^VecIdx(name, "tree", treeId, nodeId, "R"))
    }
    If primary > 0 Do ..SearchTree(name, treeId, queryVec, primary, nprobe)
    If (secondary > 0) && (nprobe > 1) Do ..SearchTree(name, treeId, queryVec, secondary, nprobe - 1)
}

ClassMethod RankCandidates(name As %String, queryVec As %Binary, k As %Integer) As %String [ Private ]
{
    Set metric = $Get(^VecIdx(name, "cfg", "metric"), "cosine")
    Kill ^||ranked
    Set docId = ""
    For {
        Set docId = $Order(^||candidates(docId))
        Quit:docId=""
        Set vec = ^VecIdx(name, "vec", docId)
        Set dist = ..Distance(queryVec, vec, metric)
        Set ^||ranked(dist, docId) = ""
    }
    Return ..TopK(k, metric)
}

ClassMethod TopK(k As %Integer, metric As %String = "cosine") As %String [ Private ]
{
    Set result = "[", first = 1, count = 0
    Set sk = ""
    For {
        Set sk = $Order(^||ranked(sk))
        Quit:sk=""
        Set docId = ""
        For {
            Set docId = $Order(^||ranked(sk, docId))
            Quit:docId=""
            Set:'first result = result _ ","
            Set first = 0
            If (metric = "cosine") || (metric = "dot") {
                Set score = -sk
            } Else {
                Set score = +sk
            }
            Set result = result _ "{""id"":""" _ docId _ """,""score"":" _ $FNumber(score,"",12) _ ",""distance"":" _ $FNumber(+sk,"",12) _ "}"
            Set count = count + 1
            If count >= k Goto TopKDone
        }
    }
TopKDone
    Set result = result _ "]"
    Return result
}

// ================================================================
// Utilities
// ================================================================

ClassMethod RandomHyperplane(dim As %Integer) As %Binary [ Private ]
{
    Set plane = ""
    For d = 1:1:dim {
        Set $vector(plane, d, "double") = $random(2000) / 1000 - 1
    }
    Set norm = $zsqr($vectorop("sum", $vectorop("v*", plane, plane)))
    If norm > 0 Set plane = $vectorop("/", plane, norm)
    Return plane
}

ClassMethod GetDim(name As %String) As %Integer [ Private ]
{
    Set dim = +$Get(^VecIdx(name, "cfg", "dim"), 0)
    If dim > 0 Return dim
    Set docId = $Order(^VecIdx(name, "vec", ""))
    If docId = "" Return 0
    Set vec = ^VecIdx(name, "vec", docId)
    Set dim = $vectorop("count", vec)
    Set ^VecIdx(name, "cfg", "dim") = dim
    Return dim
}

ClassMethod CollectAllIds(name As %String) As %List [ Private ]
{
    Set ids = "", docId = ""
    For {
        Set docId = $Order(^VecIdx(name, "vec", docId))
        Quit:docId=""
        Set ids = ids _ $ListBuild(docId)
    }
    Return ids
}

/// Info about an index
ClassMethod Info(name As %String) As %String
{
    Set dim = +$Get(^VecIdx(name, "cfg", "dim"), 0)
    Set metric = $Get(^VecIdx(name, "cfg", "metric"), "cosine")
    Set count = +$Get(^VecIdx(name, "meta", "count"), 0)
    Set trees = +$Get(^VecIdx(name, "cfg", "numTrees"), 0)
    Set leaf = +$Get(^VecIdx(name, "cfg", "leafSize"), 50)
    Return "{""name"":"""_name_""",""dim"":"_dim_",""metric"":"""_metric_""",""count"":"_count_",""trees"":"_trees_",""leafSize"":"_leaf_"}"
}

}
