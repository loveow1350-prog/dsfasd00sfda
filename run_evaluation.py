"""
Run Evaluation Only
Runs evaluation on existing pipeline outputs
"""
import sys
from pathlib import Path

from src.utils import Config
from src.evaluation_orchestrator import EvaluationOrchestrator
from src.evaluation_report_generator import ReportGenerator


def main():
    """Run evaluation on existing outputs"""
    if len(sys.argv) < 2:
        print("Usage: python run_evaluation.py <document_id> [output_dir]")
        print("\nExample:")
        print("  python run_evaluation.py 449a92bc-0905-43e8-818d-9382ce195df4")
        sys.exit(1)

    document_id = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    print(f"🔍 Starting evaluation for document: {document_id}")
    print(f"📂 Output directory: {output_dir}\n")

    try:
        # Initialize
        config = Config("config/config.yaml")
        evaluator = EvaluationOrchestrator(config)

        # Run evaluation
        eval_report = evaluator.evaluate(
            document_id=document_id,
            output_dir=output_dir
        )

        # Save reports
        report_gen = ReportGenerator()
        report_gen.save_all_formats(eval_report, output_dir)

        # Print results
        print(f"\n✅ Evaluation completed successfully!")
        print(f"\n📊 Results:")
        print(f"  - Total score: {eval_report.total_score:.1f}/100")
        print(f"  - Pass rate: {eval_report.overall_pass_rate * 100:.1f}%")
        print(f"  - Total items: {eval_report.summary['total_items']}")
        print(f"  - Passed items: {eval_report.summary['total_passed']}")

        print(f"\n📄 Generated files:")
        print(f"  - {output_dir}/{document_id}_evaluation_report.json")
        print(f"  - {output_dir}/{document_id}_evaluation_checklist.md")
        print(f"  - {output_dir}/{document_id}_evaluation_feedback.md")

        # Show category scores
        print(f"\n📋 Category scores:")
        for cat_name, cat_score in eval_report.summary['category_scores'].items():
            print(f"  - {cat_name}: {cat_score:.1f}")

        return eval_report

    except FileNotFoundError as e:
        print(f"\n❌ Error: Required file not found")
        print(f"   {e}")
        print(f"\n💡 Make sure you have run the pipeline first:")
        print(f"   python main_pipeline.py <pdf_path>")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
