"""
Test Evaluation System
Quick test script for the evaluation system
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import Config
from src.evaluation_data_loader import EvaluationDataLoader
from src.evaluation_orchestrator import EvaluationOrchestrator
from src.evaluation_report_generator import ReportGenerator


def test_data_loader():
    """Test DataLoader"""
    print("🧪 Testing DataLoader...")

    document_id = "449a92bc-0905-43e8-818d-9382ce195df4"
    output_dir = "output"

    try:
        loader = EvaluationDataLoader(output_dir, document_id)
        data = loader.load_all_data()

        print(f"  ✅ Loaded structured document: {data['structured'].document_id}")
        print(f"  ✅ Loaded {len(data['steps'].steps)} steps")
        print(f"  ✅ Loaded {len(data['problems'].problem_mapping)} problem mappings")
        print(f"  ✅ Loaded raw document with {len(data['raw'].pages)} pages")

        # Test section extraction
        purpose = loader.extract_purpose_text()
        print(f"  ✅ Purpose text length: {len(purpose) if purpose else 0}")

        return True
    except Exception as e:
        print(f"  ❌ DataLoader test failed: {e}")
        return False


def test_evaluation():
    """Test full evaluation"""
    print("\n🧪 Testing Full Evaluation...")

    document_id = "449a92bc-0905-43e8-818d-9382ce195df4"
    output_dir = "output"

    try:
        config = Config("config/config.yaml")
        evaluator = EvaluationOrchestrator(config)

        print("  🔄 Running evaluation...")
        eval_report = evaluator.evaluate(
            document_id=document_id,
            output_dir=output_dir
        )

        print(f"  ✅ Total score: {eval_report.total_score:.1f}/100")
        print(f"  ✅ Pass rate: {eval_report.overall_pass_rate * 100:.1f}%")
        print(f"  ✅ Total items: {eval_report.summary['total_items']}")

        # Show category results
        print("\n  📊 Category Results:")
        for category in eval_report.categories:
            status = "✅" if category.pass_rate >= 0.5 else "❌"
            print(f"    {status} {category.category}: {category.pass_count}/{category.total_count} ({category.pass_rate*100:.0f}%)")

        # Test report generation
        print("\n  🔄 Generating reports...")
        report_gen = ReportGenerator()
        report_gen.save_all_formats(eval_report, output_dir, "test")

        print(f"  ✅ Reports saved to {output_dir}/")

        return True
    except Exception as e:
        print(f"  ❌ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 Evaluation System Test Suite")
    print("=" * 60)

    results = []

    # Test 1: DataLoader
    results.append(("DataLoader", test_data_loader()))

    # Test 2: Full Evaluation
    results.append(("Full Evaluation", test_evaluation()))

    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
