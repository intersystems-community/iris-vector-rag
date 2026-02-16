#!/usr/bin/env python3
"""
Test script to verify all 4 pipelines can be initialized correctly
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from evaluation_framework.real_production_evaluation import RealProductionEvaluator


def test_pipeline_initialization():
    """Test that all 4 pipelines can be initialized"""
    print("🔧 Testing pipeline initialization...")

    try:
        # Create evaluator - this will call _initialize_real_pipelines()
        evaluator = RealProductionEvaluator()

        # Check which pipelines were successfully initialized
        pipeline_names = list(evaluator.pipelines.keys())
        print(f"\n✅ Successfully initialized {len(pipeline_names)}/4 pipelines:")
        for name in pipeline_names:
            print(f"  ✓ {name}")

        # List any missing pipelines
        expected_pipelines = [
            "BasicRAGPipeline",
            "CRAGPipeline",
            "GraphRAGPipeline",
            "BasicRAGRerankingPipeline",
        ]
        missing_pipelines = [p for p in expected_pipelines if p not in pipeline_names]
        if missing_pipelines:
            print("\n❌ Missing pipelines:")
            for name in missing_pipelines:
                print(f"  ✗ {name}")

        # Success check
        if len(pipeline_names) == 4:
            print("\n🎉 SUCCESS: All 4 pipelines initialized correctly!")
            print(
                f"🚀 Evaluation will now run {len(pipeline_names)} × 500 questions = {len(pipeline_names) * 500} total evaluations"
            )
            return True
        else:
            print(
                f"\n⚠️  PARTIAL SUCCESS: {len(pipeline_names)}/4 pipelines initialized"
            )
            return False

    except Exception as e:
        print(f"\n❌ FAILED: Error during pipeline initialization: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pipeline_initialization()
    sys.exit(0 if success else 1)
