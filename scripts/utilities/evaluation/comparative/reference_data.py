# eval/comparative/reference_data.py
"""
Reference benchmark results from published papers for comparison.
"""

# Reference benchmark results from published papers
REFERENCE_BENCHMARKS = {
    "multihop": {
        "GraphRAG": {
            "answer_f1": 0.7964,
            "supporting_facts_f1": 0.8493,
            "joint_f1": 0.7028
        },
        "ColBERT": {
            "answer_f1": 0.6870,
            "supporting_facts_f1": 0.7280,
            "joint_f1": 0.5780
        },
        "Basic Dense Retrieval": {
            "answer_f1": 0.6310,
            "supporting_facts_f1": 0.6670,
            "joint_f1": 0.4920
        }
    },
    "bioasq": {
        "SOTA (2022)": {
            "yesno_accuracy": 0.872,
            "factoid_mrr": 0.564,
            "list_f1": 0.479,
            "summary_rouge2": 0.497
        },
        "ColBERT + T5": {
            "yesno_accuracy": 0.841,
            "factoid_mrr": 0.481,
            "list_f1": 0.436,
            "summary_rouge2": 0.449
        },
        "BM25 + T5": {
            "yesno_accuracy": 0.814,
            "factoid_mrr": 0.423,
            "list_f1": 0.385,
            "summary_rouge2": 0.412
        }
    }
}
