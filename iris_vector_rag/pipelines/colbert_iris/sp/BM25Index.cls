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
/// Usage:
///   set result = ##class(Graph.KG.BM25Index).Build("ncit", $ListBuild("name","definition"))
///   set hits   = ##class(Graph.KG.BM25Index).Search("ncit", "ankylosing spondylitis", 10)
Class Graph.KG.BM25Index Extends %RegisteredObject
{

// ================================================================
// Tokenize — private helper used by Build, Search, Insert
// ================================================================

ClassMethod Tokenize(text As %String) As %List [ Private ]
{
    Set lower = $ZCONVERT(text, "L")
    // Try iFind tokenizer first (better punctuation handling, base IRIS)
    Try {
        Set tokens = ##class(%iFind.Utils).Analyze(lower, "en", 1)
        Return tokens
    } Catch {}
    // Fallback: iterate chars, accumulate alphanumeric runs
    Set result = "", tok = ""
    For i = 1:1:$LENGTH(lower) {
        Set c = $EXTRACT(lower, i)
        If c?1AN {
            Set tok = tok_c
        } ElseIf tok '= "" {
            Set result = result_$LISTBUILD(tok)
            Set tok = ""
        }
    }
    If tok '= "" Set result = result_$LISTBUILD(tok)
    Return result
}

// ================================================================
// Build — index all nodes over specified text properties
// ================================================================

ClassMethod Build(name As %String, propsCSV As %String, k1 As %Double = 1.5, b As %Double = 0.75) As %String
{
    Kill ^BM25Idx(name)

    Set N = 0
    Set totalLen = 0

    Set sql = "SELECT node_id FROM Graph_KG.nodes"
    Set stmt = ##class(%SQL.Statement).%New()
    Do stmt.%Prepare(sql)
    Set rs = stmt.%Execute()

    While rs.%Next() {
        Set docId = rs.%Get("node_id")

        Set docText = ""
        Set nProps = $LENGTH(propsCSV, ",")
        For pi = 1:1:nProps {
            Set propKey = $PIECE(propsCSV, ",", pi)
            Set propKey = $ZSTRIP(propKey, "<>W")
            If propKey = "" Continue
            Set pSql = "SELECT val FROM Graph_KG.rdf_props WHERE s = ? AND key = ?"
            Set pStmt = ##class(%SQL.Statement).%New()
            Do pStmt.%Prepare(pSql)
            Set pRs = pStmt.%Execute(docId, propKey)
            While pRs.%Next() {
                Set docText = docText_" "_pRs.%Get("val")
            }
        }

        Set tokens = ..Tokenize(docText)
        Set docLen = $LISTLENGTH(tokens)
        If docLen = 0 Continue

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

    // Compute and store IDF for all terms
    Set avgdl = $Select(N > 0: totalLen / N, 1: 0)
    Set vocabSize = 0
    Set term = ""
    For {
        Set term = $Order(^BM25Idx(name, "idf", term))
        Quit:term=""
        Kill ^BM25Idx(name, "idf", term)
    }

    // Compute IDF from df (document frequency) via tf subscripts
    Set term = ""
    For {
        Set term = $Order(^BM25Idx(name, "tf", term))
        Quit:term=""
        // Count df = number of docs containing this term
        Set df = 0
        Set docId = ""
        For {
            Set docId = $Order(^BM25Idx(name, "tf", term, docId))
            Quit:docId=""
            Set df = df + 1
        }
        // Robertson IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        Set idf = $ZLN((N - df + 0.5) / (df + 0.5) + 1)
        Set ^BM25Idx(name, "idf", term) = idf
        Set vocabSize = vocabSize + 1
    }

    // Store config
    Set ^BM25Idx(name, "cfg", "N") = N
    Set ^BM25Idx(name, "cfg", "avgdl") = avgdl
    Set ^BM25Idx(name, "cfg", "k1") = k1
    Set ^BM25Idx(name, "cfg", "b") = b
    Set ^BM25Idx(name, "cfg", "vocab_size") = vocabSize

    Return "{""indexed"":"_N_",""avgdl"":"_avgdl_",""vocab_size"":"_vocabSize_"}"
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
    // Use inverted score as subscript for natural $Order sorting
    Kill ^||bm25sorted
    Set docId = ""
    For {
        Set docId = $Order(^||bm25scores(docId), 1, score)
        Quit:docId=""
        // Negate score for descending sort (use large offset to avoid negative subscripts)
        Set negScore = 10000000 - score
        Set ^||bm25sorted(negScore, docId) = score
    }

    // Build JSON result array
    Set result = "["
    Set count = 0
    Set negScore = ""
    For {
        If count >= k Quit
        Set negScore = $Order(^||bm25sorted(negScore))
        Quit:negScore=""
        Set docId = ""
        For {
            If count >= k Quit
            Set docId = $Order(^||bm25sorted(negScore, docId), 1, score)
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
        Set idf = $ZLN((N - df + 0.5) / (df + 0.5) + 1)
        Set ^BM25Idx(name, "idf", tok) = idf
        // Update vocab_size if this is a new term
        If '$Data(^BM25Idx(name, "idf", tok)) {
            Set ^BM25Idx(name, "cfg", "vocab_size") = $Increment(^BM25Idx(name, "cfg", "vocab_size"))
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

}
