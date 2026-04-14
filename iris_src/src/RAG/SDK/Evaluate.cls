Class RAG.SDK.Evaluate
{

ClassMethod RunRAGAS(questionsJson As %String, groundTruthsJson As %String, topK As %Integer = 5, llmConfig As %String = "") As %String [ Language = python ]
{
    try:
        import ragas
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        import iris
    except ImportError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else str(e)
        return ('{"error":"RAGAS requires Python packages. Install with: Do $system.Python.Install(\\"ragas\\") and Do $system.Python.Install(\\"datasets\\"). Missing: ' + missing + '"}')

    import json
    questions = json.loads(questionsJson)
    ground_truths = json.loads(groundTruthsJson)

    data = []
    for q, gt in zip(questions, ground_truths):
        results_json = iris.cls("RAG.SDK.Search").VectorSearch(q if len(q) < 200 else q[:200], topK)
        results = json.loads(str(results_json))
        contexts = [r.get("text", "") for r in results if isinstance(r, dict)]
        answer = contexts[0] if contexts else ""
        data.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": gt,
        })

    try:
        dataset = Dataset.from_list(data)
        scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        per_q = []
        for i, row in enumerate(data):
            per_q.append({
                "question": row["question"],
                "faithfulness": float(scores["faithfulness"]),
                "answer_relevancy": float(scores["answer_relevancy"]),
            })
        return json.dumps({
            "faithfulness": float(scores["faithfulness"]),
            "answer_relevancy": float(scores["answer_relevancy"]),
            "context_precision": float(scores["context_precision"]),
            "context_recall": float(scores["context_recall"]),
            "per_question": per_q,
        })
    except Exception as ex:
        return '{"error":"RAGAS evaluation failed: ' + str(ex).replace('"', '\\"')[:200] + '"}'
}

}
