Class RAG.ColBERTSearch
{

ClassMethod Search(qVecsJson As %String, topK As %Integer, nProbe As %Integer) As %String [ Language = python, SqlProc ]
{
    import iris, json, time

    t0 = time.perf_counter()
    q_vecs = json.loads(qVecsJson)
    n_dim = len(q_vecs[0])

    hit_ids = set()
    for q_vec in q_vecs:
        q_str = "[" + ",".join(f"{v:.4f}" for v in q_vec) + "]"
        rs = iris.sql.exec(
            f"SELECT TOP {nProbe} centroid_id "
            f"FROM RAG.ColBERTCentroids "
            f"ORDER BY VECTOR_DOT_PRODUCT(centroid_vec, TO_VECTOR('{q_str}', FLOAT, {n_dim})) DESC"
        )
        for row in rs:
            hit_ids.add(int(row[0]))
    t1 = time.perf_counter()

    if not hit_ids:
        return json.dumps({"error": "no_centroids", "results": []})

    id_in = ",".join(str(i) for i in hit_ids)
    rs2 = iris.sql.exec(
        f"SELECT DISTINCT doc_id FROM RAG.ColBERTDocCentroids WHERE centroid_id IN ({id_in})"
    )
    candidates = [row[0] for row in rs2]
    t15 = time.perf_counter()

    doc_scores = {}
    for q_vec in q_vecs:
        q_str = "[" + ",".join(f"{v:.4f}" for v in q_vec) + "]"
        for i in range(0, len(candidates), 400):
            batch = candidates[i:i+400]
            doc_in = ",".join(f"'{d}'" for d in batch)
            rs3 = iris.sql.exec(
                f"SELECT doc_id, MAX(VECTOR_DOT_PRODUCT(tok_vec, TO_VECTOR('{q_str}', FLOAT, {n_dim}))) AS ms "
                f"FROM RAG.DocumentTokenEmbeddings "
                f"WHERE doc_id IN ({doc_in}) "
                f"GROUP BY doc_id"
            )
            for row in rs3:
                did = row[0]
                doc_scores[did] = doc_scores.get(did, 0.0) + float(row[1])
    t2 = time.perf_counter()

    top = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:topK]
    return json.dumps({
        "results": [(d, round(s, 4)) for d, s in top],
        "n_centroids": len(hit_ids),
        "n_candidates": len(candidates),
        "stage1_ms": round((t1-t0)*1000, 1),
        "stage15_ms": round((t15-t1)*1000, 1),
        "stage2_ms": round((t2-t15)*1000, 1),
        "total_ms": round((t2-t0)*1000, 1)
    })
}

}
