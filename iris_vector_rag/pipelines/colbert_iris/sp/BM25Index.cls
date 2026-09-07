/// Graph.KG.BM25Index — Pure ObjectScript BM25 lexical search index.
///
/// Implements Okapi BM25 (Robertson, 1994) over ^BM25Idx globals.
/// No SQL tables. No iFind Enterprise license required.
///
/// Storage:
///   ^BM25Idx(name, "cfg", "N")          integer — document count
///   ^BM25Idx(name, "cfg", "avgdl")      float   — average doc length
///   ^BM25Idx(name, "cfg", "k1")         float   — BM25 k1 parameter
///   ^BM25Idx(name, "cfg", "b")          float   — BM25 b parameter
///   ^BM25Idx(name, "cfg", "vocab_size") integer — distinct token count
///   ^BM25Idx(name, "idf",  term)        float   — Robertson IDF
///   ^BM25Idx(name, "tf",   term, docId) integer — term frequency (term-first!)
///   ^BM25Idx(name, "len",  docId)       integer — document token count
///
/// Stopwords: common English function words stored in ^BM25Idx("__stop__", word)
/// at class-load time. Filtered out during Tokenize to reduce IDF noise.
///
/// Usage:
///   set result = ##class(Graph.KG.BM25Index).Build("ncit", $ListBuild("name","definition"))
///   set hits   = ##class(Graph.KG.BM25Index).Search("ncit", "ankylosing spondylitis", 10)
Class Graph.KG.BM25Index Extends %RegisteredObject
{

// ================================================================
// InitStopwords — populate ^BM25Idx("__stop__") once per process
// ================================================================

ClassMethod InitStopwords() [ Private ]
{
    If $Data(^BM25Idx("__stop__", "the")) Return
    Set words = "a,an,the,and,or,but,in,on,at,to,for,of,with,by,from,is,are,was,were,be,been,being,have,has,had,do,does,did,will,would,could,should,may,might,shall,this,that,these,those,it,its,i,we,you,he,she,they,them,their,our,your,his,her,not,no,nor,so,yet,both,either,neither,each,all,any,such,as,if,than,then,there,here,when,where,which,who,what,how"
    Set n = $LENGTH(words, ",")
    For i = 1:1:n {
        Set w = $PIECE(words, ",", i)
        Set ^BM25Idx("__stop__", w) = 1
    }
}

// ================================================================
// Tokenize — private helper used by Build, Search, Insert, Delete
// ================================================================

ClassMethod Tokenize(text As %String) As %List [ Private ]
{
    Do ..InitStopwords()
    Set lower = $ZCONVERT(text, "L")
    // Try iFind tokenizer first (better punctuation handling, base IRIS)
    Set raw = ""
    Try {
        Set raw = ##class(%iFind.Utils).Analyze(lower, "en", 1)
    } Catch {}
    If raw = "" {
        // Fallback: iterate chars, accumulate alphanumeric runs
        Set tok = ""
        For i = 1:1:$LENGTH(lower) {
            Set c = $EXTRACT(lower, i)
            If c?1AN {
                Set tok = tok_c
            } ElseIf tok '= "" {
                Set raw = raw_$LISTBUILD(tok)
                Set tok = ""
            }
        }
        If tok '= "" Set raw = raw_$LISTBUILD(tok)
    }
    // Filter stopwords
    Set result = "", tIdx = 0
    While $LISTNEXT(raw, tIdx, tok) {
        If tok = "" Continue
        If $Data(^BM25Idx("__stop__", tok)) Continue
        Set result = result_$LISTBUILD(tok)
    }
    Return result
}

// ================================================================
// Build — index all nodes over specified text properties
// ================================================================

ClassMethod Build(name As %String, propsCSV As %String, k1 As %Double = 1.5, b As %Double = 0.75) As %String
{
    Kill ^BM25Idx(name)

    // Build an IN-list literal from propsCSV for the single batch query.
    // e.g. "name,definition" → "'name','definition'"
    Set inList = ""
    Set nProps = $LENGTH(propsCSV, ",")
    For pi = 1:1:nProps {
        Set propKey = $ZSTRIP($PIECE(propsCSV, ",", pi), "<>W")
        If propKey = "" Continue
        Set escaped = $Replace(propKey, "'", "''")
        Set inList = inList _ $Select(inList="":"", 1:",") _ "'" _ escaped _ "'"
    }
    If inList = "" Return "{""indexed"":0,""avgdl"":0,""vocab_size"":0}"

    // One query fetches all (node_id, val) pairs for the requested properties.
    // Ordered by node_id so we can accumulate per-doc by scanning in sequence.
    Set sql = "SELECT n.node_id, p.val FROM Graph_KG.nodes n "
            _ "LEFT JOIN Graph_KG.rdf_props p ON p.s = n.node_id AND p.key IN ("_inList_") "
            _ "ORDER BY n.node_id"
    Set stmt = ##class(%SQL.Statement).%New()
    Do stmt.%Prepare(sql)
    Set rs = stmt.%Execute()

    Set N = 0
    Set totalLen = 0
    Set curDoc = ""
    Set docText = ""

    While rs.%Next() {
        Set docId = rs.%Get("node_id")
        Set val = rs.%GetData(2)  // p.val (NULL when no matching prop row)

        If docId '= curDoc {
            // Flush previous doc
            If curDoc '= "" {
                Do ..IndexDoc(name, curDoc, docText, .N, .totalLen)
            }
            Set curDoc = docId
            Set docText = $Select(val="":"", 1:val)
        } Else {
            If val '= "" Set docText = docText _ " " _ val
        }
    }
    // Flush last doc
    If curDoc '= "" {
        Do ..IndexDoc(name, curDoc, docText, .N, .totalLen)
    }

    // Compute IDF for all terms from accumulated tf subscripts
    Set avgdl = $Select(N > 0: totalLen / N, 1: 0)
    Set vocabSize = 0
    Set term = ""
    For {
        Set term = $Order(^BM25Idx(name, "tf", term))
        Quit:term=""
        Set df = 0
        Set d = ""
        For {
            Set d = $Order(^BM25Idx(name, "tf", term, d))
            Quit:d=""
            Set df = df + 1
        }
        Set idf = $ZLN((N - df + 0.5) / (df + 0.5) + 1)
        Set ^BM25Idx(name, "idf", term) = idf
        Set vocabSize = vocabSize + 1
    }

    Set ^BM25Idx(name, "cfg", "N") = N
    Set ^BM25Idx(name, "cfg", "avgdl") = avgdl
    Set ^BM25Idx(name, "cfg", "k1") = k1
    Set ^BM25Idx(name, "cfg", "b") = b
    Set ^BM25Idx(name, "cfg", "vocab_size") = vocabSize

    Return "{""indexed"":"_N_",""avgdl"":"_avgdl_",""vocab_size"":"_vocabSize_"}"
}

// Accumulate tf and length for one document — shared by Build batch loop.
ClassMethod IndexDoc(name As %String, docId As %String, docText As %String, ByRef N As %Integer, ByRef totalLen As %Integer) [ Private ]
{
    Set tokens = ..Tokenize(docText)
    Set docLen = $LISTLENGTH(tokens)
    If docLen = 0 Return
    Kill tfMap
    Set tIdx = 0
    While $LISTNEXT(tokens, tIdx, tok) {
        If tok = "" Continue
        Set tfMap(tok) = $Get(tfMap(tok), 0) + 1
    }
    Set tok = ""
    For {
        Set tok = $Order(tfMap(tok))
        Quit:tok=""
        Set ^BM25Idx(name, "tf", tok, docId) = tfMap(tok)
    }
    Set ^BM25Idx(name, "len", docId) = docLen
    Set N = N + 1
    Set totalLen = totalLen + docLen
}

// ================================================================
// Search — BM25 ranked retrieval
// ================================================================

ClassMethod Search(name As %String, queryText As %String, k As %Integer = 10) As %String
{
    // Load config
    Set N = +$Get(^BM25Idx(name, "cfg", "N"), 0)
    If N = 0 || (queryText = "") Return "[]"

    Set avgdl = +$Get(^BM25Idx(name, "cfg", "avgdl"), 1)
    Set k1 = +$Get(^BM25Idx(name, "cfg", "k1"), 1.5)
    Set b = +$Get(^BM25Idx(name, "cfg", "b"), 0.75)

    // Tokenize query
    Set queryTokens = ..Tokenize(queryText)
    If $LISTLENGTH(queryTokens) = 0 Return "[]"

    // Accumulate BM25 scores per doc in process-private array
    Kill ^||bm25scores

    Set tIdx = 0
    While $LISTNEXT(queryTokens, tIdx, tok) {
        If tok = "" Continue
        Set idf = +$Get(^BM25Idx(name, "idf", tok), 0)
        If idf = 0 Continue

        // Iterate posting list for this term
        Set docId = ""
        For {
            Set docId = $Order(^BM25Idx(name, "tf", tok, docId), 1, tf)
            Quit:docId=""
            Set docLen = +$Get(^BM25Idx(name, "len", docId), 1)

            // BM25 score contribution
            Set numer = tf * (k1 + 1)
            Set denom = tf + k1 * (1 - b + b * docLen / avgdl)
            Set contrib = idf * numer / denom
            Set ^||bm25scores(docId) = $Get(^||bm25scores(docId), 0) + contrib
        }
    }

    // Sort by score descending — collect top k
    // Scale score to a 12-decimal fixed-point integer subscript so $Order
    // gives ascending-numeric order; we iterate in reverse with $Order(...,-1).
    // This avoids the "10M offset" overflow that breaks when BM25 scores exceed it.
    Kill ^||bm25sorted
    Set docId = ""
    For {
        Set docId = $Order(^||bm25scores(docId), 1, score)
        Quit:docId=""
        // Multiply by 1e12 and round to integer → subscript safe for any BM25 score
        Set sortKey = $NUMBER(score * 1000000000000, 0)
        Set ^||bm25sorted(sortKey, docId) = score
    }

    // Walk in descending order using $Order with direction -1
    Set result = "["
    Set count = 0
    Set sortKey = ""
    For {
        If count >= k Quit
        Set sortKey = $Order(^||bm25sorted(sortKey), -1)
        Quit:sortKey=""
        Set docId = ""
        For {
            If count >= k Quit
            Set docId = $Order(^||bm25sorted(sortKey, docId), 1, score)
            Quit:docId=""
            If count > 0 Set result = result_","
            Set safeId = $Replace(docId, """", "\""")
            Set result = result_"{""id"":"""_safeId_""",""score"":"_score_"}"
            Set count = count + 1
        }
    }
    Set result = result_"]"

    Kill ^||bm25scores, ^||bm25sorted
    Return result
}

// ================================================================
// Insert — incremental document add / update
// ================================================================

ClassMethod Insert(name As %String, docId As %String, text As %String) As %Integer
{
    Set N = +$Get(^BM25Idx(name, "cfg", "N"), 0)
    Set totalLen = N * +$Get(^BM25Idx(name, "cfg", "avgdl"), 0)

    // If replacing existing doc: remove old TF entries and adjust totals
    Set oldLen = +$Get(^BM25Idx(name, "len", docId), 0)
    If oldLen > 0 {
        // Remove old tf entries for this doc
        Set term = ""
        For {
            Set term = $Order(^BM25Idx(name, "tf", term))
            Quit:term=""
            Kill ^BM25Idx(name, "tf", term, docId)
        }
        Set totalLen = totalLen - oldLen
        Set N = N - 1
        Kill ^BM25Idx(name, "len", docId)
    }

    // Tokenize new text
    Set tokens = ..Tokenize(text)
    Set docLen = $LISTLENGTH(tokens)
    If docLen = 0 Return 0

    Kill tfMap
    Set tIdx = 0
    While $LISTNEXT(tokens, tIdx, tok) {
        If tok = "" Continue
        Set tfMap(tok) = $Get(tfMap(tok), 0) + 1
    }

    Set tok = ""
    For {
        Set tok = $Order(tfMap(tok))
        Quit:tok=""
        Set ^BM25Idx(name, "tf", tok, docId) = tfMap(tok)
    }
    Set ^BM25Idx(name, "len", docId) = docLen

    // Update N and avgdl
    Set N = N + 1
    Set totalLen = totalLen + docLen
    Set ^BM25Idx(name, "cfg", "N") = N
    Set ^BM25Idx(name, "cfg", "avgdl") = totalLen / N

    // Update IDF only for terms in new document (O(doc_length))
    Set tok = ""
    For {
        Set tok = $Order(tfMap(tok))
        Quit:tok=""
        // Recount df for this term
        Set df = 0
        Set d = ""
        For {
            Set d = $Order(^BM25Idx(name, "tf", tok, d))
            Quit:d=""
            Set df = df + 1
        }
        // Check for new term BEFORE overwriting idf entry
        Set isNewTerm = '$Data(^BM25Idx(name, "idf", tok))
        Set idf = $ZLN((N - df + 0.5) / (df + 0.5) + 1)
        Set ^BM25Idx(name, "idf", tok) = idf
        If isNewTerm {
            Set ^BM25Idx(name, "cfg", "vocab_size") = $Increment(^BM25Idx(name, "cfg", "vocab_size"))
        }
    }

    Return 1
}

// ================================================================
// Delete — remove a single document from the index
// ================================================================

ClassMethod Delete(name As %String, docId As %String) As %Integer
{
    Set oldLen = +$Get(^BM25Idx(name, "len", docId), 0)
    If oldLen = 0 Return 0  // doc not in index

    Set N = +$Get(^BM25Idx(name, "cfg", "N"), 0)
    Set totalLen = N * +$Get(^BM25Idx(name, "cfg", "avgdl"), 0)

    // Collect terms this doc contributed to
    Kill affectedTerms
    Set term = ""
    For {
        Set term = $Order(^BM25Idx(name, "tf", term))
        Quit:term=""
        If $Data(^BM25Idx(name, "tf", term, docId)) {
            Kill ^BM25Idx(name, "tf", term, docId)
            Set affectedTerms(term) = 1
        }
    }
    Kill ^BM25Idx(name, "len", docId)

    // Update doc count and avgdl
    Set N = N - 1
    Set totalLen = totalLen - oldLen
    Set ^BM25Idx(name, "cfg", "N") = N
    Set ^BM25Idx(name, "cfg", "avgdl") = $Select(N > 0: totalLen / N, 1: 0)

    // Recompute IDF for affected terms; remove terms that now have df=0
    Set term = ""
    For {
        Set term = $Order(affectedTerms(term))
        Quit:term=""
        Set df = 0
        Set d = ""
        For {
            Set d = $Order(^BM25Idx(name, "tf", term, d))
            Quit:d=""
            Set df = df + 1
        }
        If df = 0 {
            Kill ^BM25Idx(name, "idf", term)
            Set ^BM25Idx(name, "cfg", "vocab_size") = $Increment(^BM25Idx(name, "cfg", "vocab_size"), -1)
        } Else {
            If N > 0 {
                Set idf = $ZLN((N - df + 0.5) / (df + 0.5) + 1)
                Set ^BM25Idx(name, "idf", term) = idf
            }
        }
    }

    Return 1
}

// ================================================================
// Drop — remove entire index
// ================================================================

ClassMethod Drop(name As %String) As %Integer
{
    Kill ^BM25Idx(name)
    Return 1
}

// ================================================================
// Info — return index metadata
// ================================================================

ClassMethod Info(name As %String) As %String
{
    Set N = $Get(^BM25Idx(name, "cfg", "N"), "")
    If N = "" Return "{}"
    Set avgdl = +$Get(^BM25Idx(name, "cfg", "avgdl"), 0)
    Set vocabSize = +$Get(^BM25Idx(name, "cfg", "vocab_size"), 0)
    Return "{""N"":"_N_",""avgdl"":"_avgdl_",""vocab_size"":"_vocabSize_"}"
}

// ================================================================
// SearchProc — SQL stored procedure wrapper for Cypher translator
// ================================================================

ClassMethod SearchProc(name As %String, queryText As %String, k As %Integer) As %String [ SqlProc, SqlName = kg_BM25 ]
{
    Return ..Search(name, queryText, k)
}

ClassMethod List() As %String
{
	Set result = "", name = ""
	For { Set name = $Order(^BM25Idx(name)) Quit:name=""  Set result = result _ $Select(result="":"", 1:",") _ name }
	Return result
}

}
