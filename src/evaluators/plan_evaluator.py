"""
Plan Evaluator
Evaluates future plan specificity
"""
import json
import re
from typing import List, Optional, Dict

from src.models import ChecklistItem
from src.utils import LLMRouter, setup_logger, extract_json_object

logger = setup_logger(__name__)


def safe_llm_call(llm: LLMRouter, messages: List[Dict], default_result: ChecklistItem) -> Dict:
    """
    Safely call LLM and parse JSON response

    Args:
        llm: LLM router
        messages: Chat messages
        default_result: Default ChecklistItem to return on error

    Returns:
        Parsed JSON dict or default values
    """
    try:
        response = llm.chat_eval(
            messages=messages,
            response_format="json"
        )

        # Use extract_json_object for robust parsing
        result_data = extract_json_object(response)

        if not result_data:
            logger.warning("Empty or invalid JSON response from LLM")
            return {
                "result": default_result.result,
                "confidence": 0.3,
                "evidence": ["LLM 응답 파싱 실패"],
                "reasoning": "LLM이 유효한 JSON을 생성하지 못했습니다"
            }

        return result_data

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        logger.debug(f"Response was: {response[:200] if response else 'None'}")
        return {
            "result": default_result.result,
            "confidence": 0.3,
            "evidence": ["JSON 파싱 실패"],
            "reasoning": f"LLM 응답을 파싱할 수 없습니다: {str(e)}"
        }
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {
            "result": default_result.result,
            "confidence": 0.0,
            "evidence": [],
            "reasoning": f"평가 중 오류 발생: {str(e)}"
        }


class PlanEvaluator:
    """Evaluate future plan"""

    def __init__(self, llm: LLMRouter, prompts: Dict):
        self.llm = llm
        self.prompts = prompts

    def evaluate(self, plan: Optional[str]) -> List[ChecklistItem]:
        """
        Execute plan evaluation

        Args:
            plan: Plan section text

        Returns:
            List of ChecklistItem for plan evaluation
        """
        logger.info("Starting plan evaluation")
        items = []

        # 구체성 (날짜 포함 계획)
        items.append(self._evaluate_specificity(plan))

        logger.info(f"Plan evaluation complete: {len(items)} items")
        return items

    def _evaluate_specificity(self, plan: Optional[str]) -> ChecklistItem:
        """
        Evaluate plan specificity

        Question: 날짜에 따라 구체적인 계획이 작성되었는가?
        """
        logger.info("Evaluating plan specificity")

        if not plan or plan.strip() == "":
            return ChecklistItem(
                item_id="PLAN_SPEC_001",
                category="향후 계획",
                subcategory="구체성",
                question="날짜에 따라 구체적인 계획이 작성되었는가?",
                result=False,
                confidence=1.0,
                evidence=[],
                reasoning="계획(plan) 섹션이 문서에서 발견되지 않았습니다"
            )

        # Check for date patterns
        date_patterns = [
            r'\d{4}[-./년]\d{1,2}[-./월]\d{1,2}일?',  # 2024-01-15, 2024년 1월 15일
            r'\d{1,2}[-./]?\d{1,2}',                  # 01-15, 1/15
            r'\d{1,2}월\s*\d{1,2}일',                 # 1월 15일
            r'\d+주차',                                # 3주차
            r'\d+월',                                  # 1월
            r'\d+일',                                  # 15일
            r'Week\s*\d+',                            # Week 3
            r'Phase\s*\d+',                           # Phase 2
        ]

        date_found = False
        for pattern in date_patterns:
            if re.search(pattern, plan):
                date_found = True
                break

        # Use LLM for final evaluation
        prompt_template = self.prompts.get('evaluation', {}).get('plan_specificity', {})
        system_msg = prompt_template.get('system', '')
        user_msg = prompt_template.get('prompt', '').format(plan_text=plan)

        default_item = ChecklistItem(
            item_id="PLAN_SPEC_001",
            category="향후 계획",
            subcategory="구체성",
            question="날짜에 따라 구체적인 계획이 작성되었는가?",
            result=False,
            confidence=0.0,
            evidence=[],
            reasoning=""
        )

        result_data = safe_llm_call(
            self.llm,
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            default_item
        )

        # Add date pattern check to evidence
        evidence = result_data.get('evidence', [])
        if date_found:
            evidence.insert(0, "날짜/기간 표현이 발견됨")
        else:
            evidence.insert(0, "날짜/기간 표현이 발견되지 않음")

        return ChecklistItem(
            item_id="PLAN_SPEC_001",
            category="향후 계획",
            subcategory="구체성",
            question="날짜에 따라 구체적인 계획이 작성되었는가?",
            result=result_data.get('result', False),
            confidence=result_data.get('confidence', 0.5),
            evidence=evidence,
            reasoning=result_data.get('reasoning', '')
        )

