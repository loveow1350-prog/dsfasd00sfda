"""
Evaluation Orchestrator
Coordinates all evaluators and generates final report
"""
from datetime import datetime
from typing import Optional
import logging

from src.models import EvaluationReport, CategoryEvaluation, ChecklistItem
from src.utils import Config, LLMRouter, setup_logger, load_prompts
from src.search_client import SearchClient
from src.evaluation_data_loader import EvaluationDataLoader
from src.evaluators import (
    TopicEvaluator,
    DataEvaluator,
    AlgorithmEvaluator,
    ProblemSolvingEvaluator,
    PlanEvaluator
)

logger = setup_logger(__name__)


class EvaluationOrchestrator:
    """Orchestrate evaluation process"""

    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMRouter(config)
        self.search = SearchClient(config)
        self.prompts = load_prompts()

        # Initialize evaluators
        self.topic_evaluator = TopicEvaluator(self.llm, self.search, self.prompts)
        self.data_evaluator = DataEvaluator(self.llm, self.prompts)
        self.algorithm_evaluator = AlgorithmEvaluator(self.llm, self.prompts)
        self.problem_evaluator = ProblemSolvingEvaluator(self.llm, self.search, self.prompts)
        self.plan_evaluator = PlanEvaluator(self.llm, self.prompts)

    def evaluate(
        self,
        document_id: str,
        output_dir: str = "output"
    ) -> EvaluationReport:
        """
        Execute full evaluation

        Args:
            document_id: Document ID to evaluate
            output_dir: Directory containing pipeline outputs

        Returns:
            EvaluationReport with all results
        """
        logger.info(f"Starting evaluation for document {document_id}")

        # 1. Load data
        logger.info("Loading evaluation data...")
        data_loader = EvaluationDataLoader(output_dir, document_id)
        data = data_loader.load_all_data()

        # 2. Run evaluations
        categories = []

        # 2.1 주제 선정 및 창의성 (20%)
        logger.info("Evaluating topic...")
        topic_items = self.topic_evaluator.evaluate(
            topic=data['structured'].sections.get('topic'),
            purpose=data['structured'].sections.get('purpose'),
            background=data['structured'].sections.get('background'),
            generated_topic=data['structured'].metadata.get('generated_topic', False)
        )
        categories.append(self._create_category_eval(
            "주제 선정 및 창의성", 0.2, topic_items
        ))

        # 2.2 데이터 수집 및 전처리 (10%)
        logger.info("Evaluating data...")
        data_items = self.data_evaluator.evaluate(
            data_text=data['structured'].sections.get('data'),
            pipeline_text=data['structured'].sections.get('pipeline')
        )
        categories.append(self._create_category_eval(
            "데이터 수집 및 전처리", 0.1, data_items
        ))

        # 2.3 알고리즘 설계 (20%)
        logger.info("Evaluating algorithm...")
        algo_items = self.algorithm_evaluator.evaluate(
            steps=data['steps'].steps,
            topic=data['structured'].sections.get('topic'),
            purpose=data['structured'].sections.get('purpose')
        )
        categories.append(self._create_category_eval(
            "알고리즘 설계", 0.2, algo_items
        ))

        # 2.4 문제 해결력 (20%)
        logger.info("Evaluating problem-solving...")
        problem_items = self.problem_evaluator.evaluate(
            problem_mapping=data['problems'].problem_mapping,
            steps=data['steps'].steps,
            data_loader=data_loader
        )
        categories.append(self._create_category_eval(
            "문제 해결력", 0.2, problem_items
        ))

        # 2.5 향후 계획 (가중치 없음)
        logger.info("Evaluating plan...")
        plan_items = self.plan_evaluator.evaluate(
            plan=data['structured'].sections.get('plan')
        )
        categories.append(self._create_category_eval(
            "향후 계획", 0.0, plan_items
        ))

        # 3. Calculate total score
        total_score = sum(cat.score for cat in categories)

        # 4. Generate report
        report = EvaluationReport(
            document_id=document_id,
            timestamp=datetime.now().isoformat(),
            categories=categories,
            total_score=total_score,
            overall_pass_rate=self._calculate_overall_pass_rate(categories),
            summary=self._generate_summary(categories),
            recommendations=self._generate_recommendations(categories)
        )

        logger.info(f"Evaluation complete! Total score: {total_score:.1f}/100")
        return report

    def _create_category_eval(
        self,
        category: str,
        weight: float,
        items: list[ChecklistItem]
    ) -> CategoryEvaluation:
        """Create category evaluation object"""
        pass_count = sum(1 for item in items if item.result)
        total = len(items)
        pass_rate = pass_count / total if total > 0 else 0.0
        score = pass_rate * weight * 100  # 0-100 scale

        return CategoryEvaluation(
            category=category,
            weight=weight,
            checklist_items=items,
            pass_count=pass_count,
            total_count=total,
            pass_rate=pass_rate,
            score=score
        )

    def _calculate_overall_pass_rate(self, categories: list[CategoryEvaluation]) -> float:
        """Calculate overall pass rate across all categories"""
        total_passed = sum(cat.pass_count for cat in categories)
        total_items = sum(cat.total_count for cat in categories)
        return total_passed / total_items if total_items > 0 else 0.0

    def _generate_summary(self, categories: list[CategoryEvaluation]) -> dict:
        """Generate summary statistics"""
        return {
            "total_categories": len(categories),
            "total_items": sum(cat.total_count for cat in categories),
            "total_passed": sum(cat.pass_count for cat in categories),
            "category_scores": {
                cat.category: cat.score for cat in categories
            },
            "weighted_categories": [
                cat.category for cat in categories if cat.weight > 0
            ]
        }

    def _generate_recommendations(self, categories: list[CategoryEvaluation]) -> list[str]:
        """Generate improvement recommendations"""
        recommendations = []

        for category in categories:
            failed_items = [item for item in category.checklist_items if not item.result]
            if failed_items:
                for item in failed_items:
                    rec = f"[{category.category}] {item.subcategory}: {item.reasoning}"
                    recommendations.append(rec)

        return recommendations
