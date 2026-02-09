"""
Main Pipeline Orchestrator
Coordinates all agents and manages state
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import logging

from src.models import SequentialSteps, ProblemMapping
from src.utils import Config, LLMRouter, CacheManager
from src.pdf_extractor import PDFExtractor
from src.structure_parser import StructureParser
from src.step_decomposer import StepDecomposer
from src.search_client import SearchClient
from src.problem_analyzer import ProblemAnalyzer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrate multi-agent pipeline"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = Config(config_path)
        self.llm = LLMRouter(self.config)
        self.cache = CacheManager(self.config)
        self.search = SearchClient(self.config)

        # Initialize agents
        self.pdf_extractor = PDFExtractor()
        self.structure_parser = StructureParser(self.config, self.llm, self.cache)
        self.step_decomposer = StepDecomposer(self.config, self.llm)
        self.problem_analyzer = ProblemAnalyzer(
            self.config, self.llm, self.cache, self.search
        )

    def process_document(
        self, pdf_path: str, output_dir: str = "output"
    ) -> Tuple[Optional[ProblemMapping], Optional[Path]]:
        """
        Process PDF document through entire pipeline

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save outputs

        Returns:
            ProblemMapping or None if pipeline fails
        """
        logger.info(f"Starting pipeline for: {pdf_path}")
        pdf_name = Path(pdf_path).stem
        start_time = datetime.now()

        try:
            # Stage 1: PDF Extraction
            logger.info("Stage 1/4: PDF Extraction")
            raw_doc = self._run_with_status(
                agent_name="pdf_extractor",
                func=lambda: self.pdf_extractor.extract(pdf_path),
                doc_id=None,
                progress=25,
            )

            if not raw_doc:
                logger.error("PDF extraction failed")
                return None, None

            doc_id = raw_doc.document_id
            output_dir = Path(output_dir) / doc_id
            self._save_artifact(output_dir, pdf_name, doc_id, "raw_document", raw_doc)

            # Stage 2: Structure Parsing
            logger.info("Stage 2/4: Structure Parsing")
            structured_doc = self._run_with_status(
                agent_name="structure_parser",
                func=lambda: self.structure_parser.parse(raw_doc),
                doc_id=doc_id,
                progress=50,
            )

            if not structured_doc:
                logger.error("Structure parsing failed")
                return None, None

            self._save_artifact(
                output_dir, pdf_name, doc_id, "structured_document", structured_doc
            )

            # Stage 3: Step Decomposition
            logger.info("Stage 3/4: Step Decomposition")
            sequential_steps = self._run_with_status(
                agent_name="step_decomposer",
                func=lambda: self.step_decomposer.decompose(structured_doc),
                doc_id=doc_id,
                progress=75,
            )

            if not sequential_steps:
                logger.error("Step decomposition failed")
                return None, None

            self._save_artifact(
                output_dir, pdf_name, doc_id, "sequential_steps", sequential_steps
            )

            # Stage 4: Problem Analysis
            logger.info("Stage 4/4: Problem Analysis")
            problem_mapping = self._run_with_status(
                agent_name="problem_analyzer",
                func=lambda: self.problem_analyzer.analyze(sequential_steps),
                doc_id=doc_id,
                progress=100,
            )

            if not problem_mapping:
                logger.error("Problem analysis failed")
                return None, None

            self._save_artifact(
                output_dir, pdf_name, doc_id, "problem_mapping", problem_mapping
            )

            # Generate final report
            self._generate_report(
                output_dir, pdf_name, doc_id, problem_mapping, sequential_steps
            )

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Pipeline completed successfully in {elapsed:.2f}s")

            return problem_mapping, output_dir, None

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return None, None, str(e)

    def _run_with_status(
        self, agent_name: str, func, doc_id: Optional[str], progress: int
    ):
        """Run agent with status tracking"""
        try:
            if doc_id:
                self.cache.set_document_status(
                    doc_id,
                    {
                        "current_agent": agent_name,
                        "progress": progress,
                        "timestamp": datetime.now().isoformat(),
                        "errors": [],
                    },
                )

            result = func()

            if doc_id:
                self.cache.set_artifact(doc_id, agent_name, result)

            return result

        except Exception as e:
            logger.error(f"Agent {agent_name} failed: {e}")

            if doc_id:
                status = self.cache.get_document_status(doc_id) or {}
                errors = status.get("errors", [])
                errors.append(f"{agent_name}: {str(e)}")
                status["errors"] = errors
                self.cache.set_document_status(doc_id, status)

            raise

    def _save_artifact(
        self, output_dir: str, pdf_name: str, doc_id: str, name: str, artifact
    ):
        """Save artifact to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / f"{pdf_name}_{doc_id}_{name}.json"

        # Convert Pydantic model to dict (exclude image_data from serialization)
        if hasattr(artifact, "model_dump"):
            data = artifact.model_dump(exclude={"pages": {"__all__": {"image_data"}}})
        else:
            data = artifact

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved artifact: {file_path}")

    def _generate_report(
        self,
        output_dir: str,
        pdf_name: str,
        doc_id: str,
        problem_mapping: ProblemMapping,
        sequential_steps: SequentialSteps,
    ):
        """Generate human-readable report"""
        output_path = Path(output_dir)
        report_path = output_path / f"{pdf_name}_{doc_id}_report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# NLP Pipeline Analysis Report\n\n")
            f.write(f"**Document ID:** {doc_id}\n")
            f.write(
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Total Steps:** {sequential_steps.metadata['total_steps']}\n")
            f.write(
                f"- **Total Problems:** {problem_mapping.summary['total_problems']}\n"
            )
            f.write(
                f"- **Critical Steps:** {len(problem_mapping.summary['critical_steps'])}\n\n"
            )

            # Problems by category
            f.write("### Problems by Category\n\n")
            for category, count in problem_mapping.summary["by_category"].items():
                if count > 0:
                    f.write(f"- **{category}:** {count}\n")
            f.write("\n")

            # Detailed steps and problems
            f.write("## Detailed Analysis\n\n")

            for step in sequential_steps.steps:
                f.write(f"### {step.step_id}: {step.action}\n\n")
                f.write(f"**Category:** {step.category}\n\n")
                f.write(f"**Input:** {step.input}\n\n")
                f.write(f"**Output:** {step.output}\n\n")

                if step.techniques:
                    f.write(f"**Techniques:** {', '.join(step.techniques)}\n\n")

                if step.dependencies:
                    f.write(f"**Dependencies:** {', '.join(step.dependencies)}\n\n")

                # Problems for this step
                if step.step_id in problem_mapping.problem_mapping:
                    f.write("**Problems Addressed:**\n\n")

                    for problem in problem_mapping.problem_mapping[
                        step.step_id
                    ].problems:
                        # severity는 여전히 Enum이므로 .value 사용
                        severity_emoji = {
                            "low": "🟢",
                            "medium": "🟡",
                            "high": "🔴",
                        }.get(problem.severity.value, "⚪")

                        f.write(
                            f"{severity_emoji} **{problem.problem_id}** [{problem.category}]\n\n"
                        )
                        f.write(f"  {problem.description}\n\n")
                        f.write(f"  *Addressed by: {problem.addressed_by}*\n\n")

                f.write("---\n\n")

        logger.info(f"Generated report: {report_path}")


def main():
    """Main entry point"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main_pipeline.py <pdf_path> [--with-evaluation]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    run_evaluation = "--with-evaluation" in sys.argv

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    orchestrator = PipelineOrchestrator()
    result, output_dir = orchestrator.process_document(pdf_path)

    if result:
        print("\n✅ Pipeline completed successfully!")
        print(f"Total problems identified: {result.summary['total_problems']}")
        print(f"Check 'output' directory for detailed results")

        # Run evaluation if requested
        if run_evaluation:
            print("\n🔍 Starting evaluation...")
            try:
                from src.evaluation_orchestrator import EvaluationOrchestrator
                from src.evaluation_report_generator import ReportGenerator

                evaluator = EvaluationOrchestrator(orchestrator.config)
                eval_report = evaluator.evaluate(
                    document_id=result.document_id,
                    output_dir=str(output_dir),  # Convert Path to str
                )

                # Save reports in a separate reports subdirectory
                report_gen = ReportGenerator()
                pdf_name = Path(pdf_path).stem
                reports_dir = output_dir / "reports"
                reports_dir.mkdir(exist_ok=True)
                report_gen.save_all_formats(eval_report, reports_dir, pdf_name)

                print(f"\n✅ Evaluation completed!")
                print(f"📊 Total score: {eval_report.total_score:.1f}/100")
                print(f"📋 Pass rate: {eval_report.overall_pass_rate * 100:.1f}%")
                print(f"📄 Reports saved in: {reports_dir}")

            except Exception as e:
                print(f"\n❌ Evaluation failed: {e}")
                import traceback

                traceback.print_exc()
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
