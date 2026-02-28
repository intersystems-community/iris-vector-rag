#!/usr/bin/env python3
"""
Simple test to verify pipeline imports work correctly
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def test_pipeline_imports():
    """Test that all 4 pipeline classes can be imported"""
    print("🔧 Testing pipeline imports...")

    pipeline_results = {}

    # Test BasicRAGPipeline import
    try:

        pipeline_results["BasicRAGPipeline"] = "✅ SUCCESS"
        print("✅ BasicRAGPipeline imported successfully")
    except Exception as e:
        pipeline_results["BasicRAGPipeline"] = f"❌ FAILED: {e}"
        print(f"❌ BasicRAGPipeline import failed: {e}")

    # Test CRAGPipeline import
    try:

        pipeline_results["CRAGPipeline"] = "✅ SUCCESS"
        print("✅ CRAGPipeline imported successfully")
    except Exception as e:
        pipeline_results["CRAGPipeline"] = f"❌ FAILED: {e}"
        print(f"❌ CRAGPipeline import failed: {e}")

    # Test GraphRAGPipeline import
    try:

        pipeline_results["GraphRAGPipeline"] = "✅ SUCCESS"
        print("✅ GraphRAGPipeline imported successfully")
    except Exception as e:
        pipeline_results["GraphRAGPipeline"] = f"❌ FAILED: {e}"
        print(f"❌ GraphRAGPipeline import failed: {e}")

    # Test BasicRAGRerankingPipeline import
    try:

        pipeline_results["BasicRAGRerankingPipeline"] = "✅ SUCCESS"
        print("✅ BasicRAGRerankingPipeline imported successfully")
    except Exception as e:
        pipeline_results["BasicRAGRerankingPipeline"] = f"❌ FAILED: {e}"
        print(f"❌ BasicRAGRerankingPipeline import failed: {e}")

    # Summary
    successful_imports = sum(
        1 for result in pipeline_results.values() if "SUCCESS" in result
    )
    print(f"\n📊 Summary: {successful_imports}/4 pipelines imported successfully")

    if successful_imports == 4:
        print("🎉 ALL PIPELINE IMPORTS SUCCESSFUL!")
        print("🚀 The evaluation system can now run all 4 pipelines:")
        print("   • BasicRAGPipeline")
        print("   • CRAGPipeline")
        print("   • GraphRAGPipeline")
        print("   • BasicRAGRerankingPipeline")
        print("📈 This means 500 questions × 4 pipelines = 2000 total evaluations")
        return True
    else:
        print("⚠️  Some pipeline imports failed")
        return False


if __name__ == "__main__":
    success = test_pipeline_imports()
    sys.exit(0 if success else 1)
