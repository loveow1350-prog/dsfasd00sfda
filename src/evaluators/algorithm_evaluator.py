"""
Algorithm Evaluator
Evaluates algorithm design specificity and relevance
"""
import json
from typing import List, Optional, Dict

from src.models import ChecklistItem, Step
from src.utils import LLMRouter, setup_logger, extract_json_object

logger = setup_logger(__name__)


class AlgorithmEvaluator:
    """Evaluate algorithm design"""

    def __init__(self, llm: LLMRouter, prompts: Dict):
        self.llm = llm
        self.prompts = prompts

    def evaluate(self, steps: List[Step], topic: Optional[str], purpose: Optional[str]) -> List[ChecklistItem]:
        """
        Execute algorithm evaluation

        Args:
            steps: List of steps from sequential_steps.json
            topic: Topic section text
            purpose: Purpose section text

        Returns:
            List of ChecklistItem for algorithm evaluation
        """
        logger.info("Starting algorithm evaluation")
        items = []

        # 1. 구체성 (단계별 프레임워크)
        items.append(self._evaluate_specificity(steps))

        # 2. 적합성 (목적 부합)
        items.append(self._evaluate_relevance(steps, topic, purpose))

        logger.info(f"Algorithm evaluation complete: {len(items)} items")
        return items

    def _evaluate_specificity(self, steps: List[Step]) -> ChecklistItem:
        """
        Evaluate framework specificity

        Question: 단계별 프레임워크가 구체적으로 제시되었는가?
        """
        logger.info("Evaluating algorithm specificity")

        if not steps or len(steps) == 0:
            return ChecklistItem(
                item_id="ALGO_SPEC_001",
                category="알고리즘 설계",
                subcategory="구체성",
                question="단계별 프레임워크가 구체적으로 제시되었는가?",
                result=False,
                confidence=1.0,
                evidence=[],
                reasoning="단계(steps)가 정의되지 않았습니다"
            )

        # Convert steps to JSON for prompt
        steps_json = json.dumps([
            {
                "step_id": s.step_id,
                "action": s.action,
                "input": s.input,
                "output": s.output,
                "techniques": s.techniques
            }
            for s in steps
        ], ensure_ascii=False, indent=2)

        prompt_template = self.prompts.get('evaluation', {}).get('algorithm_specificity', {})
        system_msg = prompt_template.get('system', '')
        user_msg = prompt_template.get('prompt', '').format(steps_json=steps_json)

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
                logger.warning("Failed to parse JSON response")
                return ChecklistItem(
                    item_id="ALGO_SPEC_001",
                    category="알고리즘 설계",
                    subcategory="구체성",
                    question="단계별 프레임워크가 구체적으로 제시되었는가?",
                    result=False,
                    confidence=0.3,
                    evidence=[],
                    reasoning="LLM 응답을 파싱할 수 없습니다"
                )

            # Add step count to evidence
            evidence = result_data.get('evidence', [])
            evidence.insert(0, f"총 {len(steps)}개의 단계가 정의됨")

            return ChecklistItem(
                item_id="ALGO_SPEC_001",
                category="알고리즘 설계",
                subcategory="구체성",
                question="단계별 프레임워크가 구체적으로 제시되었는가?",
                result=result_data.get('result', False),
                confidence=result_data.get('confidence', 0.5),
                evidence=evidence,
                reasoning=result_data.get('reasoning', '')
            )

        except Exception as e:
            logger.error(f"Error in algorithm specificity evaluation: {e}")
            return ChecklistItem(
                item_id="ALGO_SPEC_001",
                category="알고리즘 설계",
                subcategory="구체성",
                question="단계별 프레임워크가 구체적으로 제시되었는가?",
                result=False,
                confidence=0.0,
                evidence=[],
                reasoning=f"평가 중 오류 발생: {str(e)}"
            )

    def _evaluate_relevance(self, steps: List[Step], topic: Optional[str], purpose: Optional[str]) -> ChecklistItem:
        """
        Evaluate framework relevance to purpose

        Question: 프레임워크의 출력이 달성하고자 하는 목적에 부합하는가?
        """
        logger.info("Evaluating algorithm relevance")

        if not steps or len(steps) == 0:
            return ChecklistItem(
                item_id="ALGO_RELEV_001",
                category="알고리즘 설계",
                subcategory="적합성",
                question="프레임워크의 출력이 달성하고자 하는 목적에 부합하는가?",
                result=False,
                confidence=1.0,
                evidence=[],
                reasoning="단계(steps)가 정의되지 않았습니다"
            )

        purpose_text = purpose if purpose else "목적 정보 없음"

        # Convert steps to JSON
        steps_json = json.dumps([
            {
                "step_id": s.step_id,
                "action": s.action,
                "output": s.output
            }
            for s in steps
        ], ensure_ascii=False, indent=2)

        prompt_template = self.prompts.get('evaluation', {}).get('algorithm_relevance', {})
        system_msg = prompt_template.get('system', '')
        user_msg = prompt_template.get('prompt', '').format(
            topic_text=topic,
            purpose_text=purpose_text,
            steps_json=steps_json
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
                logger.warning("Failed to parse JSON response")
                return ChecklistItem(
                    item_id="ALGO_RELEV_001",
                    category="알고리즘 설계",
                    subcategory="적합성",
                    question="프레임워크의 출력이 달성하고자 하는 목적에 부합하는가?",
                    result=False,
                    confidence=0.3,
                    evidence=[],
                    reasoning="LLM 응답을 파싱할 수 없습니다"
                )

            # Add final output to evidence
            evidence = result_data.get('evidence', [])
            final_step = steps[-1]
            evidence.insert(0, f"최종 출력: {final_step.output}")

            return ChecklistItem(
                item_id="ALGO_RELEV_001",
                category="알고리즘 설계",
                subcategory="적합성",
                question="프레임워크의 출력이 달성하고자 하는 목적에 부합하는가?",
                result=result_data.get('result', False),
                confidence=result_data.get('confidence', 0.5),
                evidence=evidence,
                reasoning=result_data.get('reasoning', '')
            )

        except Exception as e:
            logger.error(f"Error in algorithm relevance evaluation: {e}")
            return ChecklistItem(
                item_id="ALGO_RELEV_001",
                category="알고리즘 설계",
                subcategory="적합성",
                question="프레임워크의 출력이 달성하고자 하는 목적에 부합하는가?",
                result=False,
                confidence=0.0,
                evidence=[],
                reasoning=f"평가 중 오류 발생: {str(e)}"
            )
