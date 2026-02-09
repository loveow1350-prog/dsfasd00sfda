"""
Problem Solving Evaluator
Evaluates problem-solving purposefulness and validity
"""
import json
from typing import List, Dict

from src.models import ChecklistItem, Step, ProblemMapping
from src.utils import LLMRouter, setup_logger, extract_json_object
from src.search_client import SearchClient
from src.evaluation_data_loader import EvaluationDataLoader

logger = setup_logger(__name__)


class ProblemSolvingEvaluator:
    """Evaluate problem-solving approach"""

    def __init__(self, llm: LLMRouter, search: SearchClient, prompts: Dict):
        self.llm = llm
        self.search = search
        self.prompts = prompts

    def evaluate(
        self,
        problem_mapping: Dict[str, any],
        steps: List[Step],
        data_loader: EvaluationDataLoader
    ) -> List[ChecklistItem]:
        """
        Execute problem-solving evaluation

        Args:
            problem_mapping: Problem mapping dictionary
            steps: List of steps
            data_loader: Data loader for accessing raw chunks

        Returns:
            List of ChecklistItem for problem-solving evaluation
        """
        logger.info("Starting problem-solving evaluation")
        items = []

        # 1. 목적성 (단계 사용 이유)
        items.append(self._evaluate_purposefulness(problem_mapping, steps))

        # 2. 타당성 (기술의 적합성) - 검색 기반
        items.append(self._evaluate_validity(problem_mapping, steps))

        logger.info(f"Problem-solving evaluation complete: {len(items)} items")
        return items

    def _evaluate_purposefulness(
        self,
        problem_mapping: Dict[str, any],
        steps: List[Step]
    ) -> ChecklistItem:
        """
        Evaluate purposefulness

        Question: 각 단계의 프레임워크를 사용한 이유가 구체적으로 명시되었는가?
        """
        logger.info("Evaluating problem-solving purposefulness")

        if not problem_mapping:
            return ChecklistItem(
                item_id="PROBLEM_PURPOSE_001",
                category="문제 해결력",
                subcategory="목적성",
                question="각 단계의 프레임워크를 사용한 이유가 구체적으로 명시되었는가?",
                result=False,
                confidence=1.0,
                evidence=[],
                reasoning="문제 매핑(problem_mapping)이 정의되지 않았습니다"
            )

        # Create summary of problem mapping
        problem_summary = []
        for step_id, step_problems in problem_mapping.items():
            if hasattr(step_problems, 'problems'):
                problems = step_problems.problems
            else:
                problems = step_problems.get('problems', [])

            for problem in problems:
                if hasattr(problem, 'description'):
                    desc = problem.description
                    addressed = problem.addressed_by
                else:
                    desc = problem.get('description', '')
                    addressed = problem.get('addressed_by', '')

                problem_summary.append(f"Step {step_id}: {desc} → {addressed}")

        summary_text = "\n".join(problem_summary[:10])  # Limit to first 10

        prompt_template = self.prompts.get('evaluation', {}).get('problem_purposefulness', {})
        system_msg = prompt_template.get('system', '')
        user_msg = prompt_template.get('prompt', '').format(
            problem_mapping_summary=summary_text
        )

        try:
            response = self.llm.chat_eval(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                response_format="json"
            )

            result_data = extract_json_object(response)

            if not result_data:
                return ChecklistItem(
                    item_id="PROBLEM_PURPOSE_001",
                    category="문제 해결력",
                    subcategory="목적성",
                    question="각 단계의 프레임워크를 사용한 이유가 구체적으로 명시되었는가?",
                    result=False,
                    confidence=0.3,
                    evidence=[],
                    reasoning="LLM 응답을 파싱할 수 없습니다"
                )

            # Add problem count to evidence
            evidence = result_data.get('evidence', [])
            evidence.insert(0, f"총 {len(problem_summary)}개의 문제가 정의됨")

            return ChecklistItem(
                item_id="PROBLEM_PURPOSE_001",
                category="문제 해결력",
                subcategory="목적성",
                question="각 단계의 프레임워크를 사용한 이유가 구체적으로 명시되었는가?",
                result=result_data.get('result', False),
                confidence=result_data.get('confidence', 0.5),
                evidence=evidence,
                reasoning=result_data.get('reasoning', '')
            )

        except Exception as e:
            logger.error(f"Error in purposefulness evaluation: {e}")
            return ChecklistItem(
                item_id="PROBLEM_PURPOSE_001",
                category="문제 해결력",
                subcategory="목적성",
                question="각 단계의 프레임워크를 사용한 이유가 구체적으로 명시되었는가?",
                result=False,
                confidence=0.0,
                evidence=[],
                reasoning=f"평가 중 오류 발생: {str(e)}"
            )

    def _evaluate_validity(
        self,
        problem_mapping: Dict[str, any],
        steps: List[Step]
    ) -> ChecklistItem:
        """
        Evaluate validity using search

        Question: 단계마다 사용된 기술이 문제해결에 있어 타당한가?
        """
        logger.info("Evaluating problem-solving validity (search-based)")

        if not problem_mapping:
            return ChecklistItem(
                item_id="PROBLEM_VALID_001",
                category="문제 해결력",
                subcategory="타당성",
                question="단계마다 사용된 기술이 문제해결에 있어 타당한가?",
                result=False,
                confidence=1.0,
                evidence=[],
                reasoning="문제 매핑(problem_mapping)이 정의되지 않았습니다"
            )

        try:
            # Sample 3 techniques to validate
            techniques_to_validate = []
            for step_id, step_problems in list(problem_mapping.items())[:3]:
                if hasattr(step_problems, 'problems'):
                    problems = step_problems.problems
                else:
                    problems = step_problems.get('problems', [])

                for problem in problems:
                    if hasattr(problem, 'addressed_by'):
                        technique = problem.addressed_by
                        problem_desc = problem.description
                    else:
                        technique = problem.get('addressed_by', '')
                        problem_desc = problem.get('description', '')

                    techniques_to_validate.append({
                        'technique': technique,
                        'problem': problem_desc
                    })

            if not techniques_to_validate:
                return ChecklistItem(
                    item_id="PROBLEM_VALID_001",
                    category="문제 해결력",
                    subcategory="타당성",
                    question="단계마다 사용된 기술이 문제해결에 있어 타당한가?",
                    result=False,
                    confidence=0.5,
                    evidence=[],
                    reasoning="검증할 기술을 발견하지 못했습니다"
                )

            # Search for first technique
            first = techniques_to_validate[0]
            search_query = f"{first['technique']} for {first['problem']}"
            search_results = self.search.search(search_query)

            # If we found results, technique is likely valid
            is_valid = len(search_results) > 0

            return ChecklistItem(
                item_id="PROBLEM_VALID_001",
                category="문제 해결력",
                subcategory="타당성",
                question="단계마다 사용된 기술이 문제해결에 있어 타당한가?",
                result=is_valid,
                confidence=0.7,
                evidence=[
                    f"검증한 기술: {first['technique']}",
                    f"검색 결과: {len(search_results)}건"
                ],
                reasoning=f"'{first['technique']}' 기술에 대한 검색 결과 {len(search_results)}개가 발견되어 " +
                         ("타당한 기술로 판단됩니다" if is_valid else "검증이 필요합니다"),
                search_results=search_results[:2]
            )

        except Exception as e:
            logger.error(f"Error in validity evaluation: {e}")
            return ChecklistItem(
                item_id="PROBLEM_VALID_001",
                category="문제 해결력",
                subcategory="타당성",
                question="단계마다 사용된 기술이 문제해결에 있어 타당한가?",
                result=False,
                confidence=0.0,
                evidence=[],
                reasoning=f"평가 중 오류 발생 (검색 실패): {str(e)}"
            )
