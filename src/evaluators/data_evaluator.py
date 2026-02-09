"""
Data Evaluator
Evaluates data collection and preprocessing
"""
import json
from typing import List, Optional, Dict

from src.models import ChecklistItem
from src.utils import LLMRouter, setup_logger, extract_json_object

logger = setup_logger(__name__)


class DataEvaluator:
    """Evaluate data collection and preprocessing"""

    def __init__(self, llm: LLMRouter, prompts: Dict):
        self.llm = llm
        self.prompts = prompts

    def evaluate(self, data_text: Optional[str], pipeline_text: Optional[str]) -> List[ChecklistItem]:
        """
        Execute data evaluation

        Args:
            data_text: Data section text
            pipeline_text: Pipeline section text

        Returns:
            List of ChecklistItem for data evaluation
        """
        logger.info("Starting data evaluation")
        items = []

        # 1. 데이터 확보 여부
        items.append(self._evaluate_availability(data_text))

        # 2. 데이터 활용 계획
        items.append(self._evaluate_usage(data_text, pipeline_text))

        # 3. 데이터 전처리
        items.append(self._evaluate_preprocessing(data_text, pipeline_text))

        logger.info(f"Data evaluation complete: {len(items)} items")
        return items

    def _evaluate_availability(self, data_text: Optional[str]) -> ChecklistItem:
        """
        Evaluate data availability

        Question: 분석에 필요한 데이터가 확보되었거나 확보 계획이 구체적인가?
        """
        logger.info("Evaluating data availability")

        if not data_text or data_text.strip() == "":
            return ChecklistItem(
                item_id="DATA_AVAIL_001",
                category="데이터 수집 및 전처리",
                subcategory="데이터 확보 여부",
                question="분석에 필요한 데이터가 확보되었거나 확보 계획이 구체적인가?",
                result=False,
                confidence=1.0,
                evidence=[],
                reasoning="데이터(data) 섹션이 문서에서 발견되지 않았습니다"
            )

        prompt_template = self.prompts.get('evaluation', {}).get('data_availability', {})
        system_msg = prompt_template.get('system', '')
        user_msg = prompt_template.get('prompt', '').format(data_text=data_text)

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
                    item_id="DATA_AVAIL_001",
                    category="데이터 수집 및 전처리",
                    subcategory="데이터 확보 여부",
                    question="분석에 필요한 데이터가 확보되었거나 확보 계획이 구체적인가?",
                    result=False,
                    confidence=0.3,
                    evidence=[],
                    reasoning="LLM 응답을 파싱할 수 없습니다"
                )

            return ChecklistItem(
                item_id="DATA_AVAIL_001",
                category="데이터 수집 및 전처리",
                subcategory="데이터 확보 여부",
                question="분석에 필요한 데이터가 확보되었거나 확보 계획이 구체적인가?",
                result=result_data.get('result', False),
                confidence=result_data.get('confidence', 0.5),
                evidence=result_data.get('evidence', []),
                reasoning=result_data.get('reasoning', '')
            )

        except Exception as e:
            logger.error(f"Error in availability evaluation: {e}")
            return ChecklistItem(
                item_id="DATA_AVAIL_001",
                category="데이터 수집 및 전처리",
                subcategory="데이터 확보 여부",
                question="분석에 필요한 데이터가 확보되었거나 확보 계획이 구체적인가?",
                result=False,
                confidence=0.0,
                evidence=[],
                reasoning=f"평가 중 오류 발생: {str(e)}"
            )

    def _evaluate_usage(self, data_text: Optional[str], pipeline_text: Optional[str]) -> ChecklistItem:
        """
        Evaluate data usage plan

        Question: 데이터를 어떻게 활용할지 구체적으로 정해졌는가?
        """
        logger.info("Evaluating data usage")

        if not data_text or data_text.strip() == "":
            return ChecklistItem(
                item_id="DATA_USAGE_001",
                category="데이터 수집 및 전처리",
                subcategory="데이터 활용",
                question="데이터를 어떻게 활용할지 구체적으로 정해졌는가?",
                result=False,
                confidence=1.0,
                evidence=[],
                reasoning="데이터(data) 섹션이 문서에서 발견되지 않았습니다"
            )

        pipeline_text = pipeline_text if pipeline_text else "파이프라인 정보 없음"

        prompt_template = self.prompts.get('evaluation', {}).get('data_usage', {})
        system_msg = prompt_template.get('system', '')
        user_msg = prompt_template.get('prompt', '').format(
            data_text=data_text,
            pipeline_text=pipeline_text
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
                    item_id="DATA_USAGE_001",
                    category="데이터 수집 및 전처리",
                    subcategory="데이터 활용",
                    question="데이터를 어떻게 활용할지 구체적으로 정해졌는가?",
                    result=False,
                    confidence=0.3,
                    evidence=[],
                    reasoning="LLM 응답을 파싱할 수 없습니다"
                )

            return ChecklistItem(
                item_id="DATA_USAGE_001",
                category="데이터 수집 및 전처리",
                subcategory="데이터 활용",
                question="데이터를 어떻게 활용할지 구체적으로 정해졌는가?",
                result=result_data.get('result', False),
                confidence=result_data.get('confidence', 0.5),
                evidence=result_data.get('evidence', []),
                reasoning=result_data.get('reasoning', '')
            )

        except Exception as e:
            logger.error(f"Error in usage evaluation: {e}")
            return ChecklistItem(
                item_id="DATA_USAGE_001",
                category="데이터 수집 및 전처리",
                subcategory="데이터 활용",
                question="데이터를 어떻게 활용할지 구체적으로 정해졌는가?",
                result=False,
                confidence=0.0,
                evidence=[],
                reasoning=f"평가 중 오류 발생: {str(e)}"
            )

    def _evaluate_preprocessing(self, data_text: Optional[str], pipeline_text: Optional[str]) -> ChecklistItem:
        """
        Evaluate data preprocessing

        Question: 알고리즘에 적합한 전처리 방법이 제시되었는가?
        """
        logger.info("Evaluating data preprocessing")

        if not data_text or data_text.strip() == "":
            return ChecklistItem(
                item_id="DATA_PREPROC_001",
                category="데이터 수집 및 전처리",
                subcategory="데이터 전처리",
                question="알고리즘에 적합한 전처리 방법이 제시되었는가?",
                result=False,
                confidence=1.0,
                evidence=[],
                reasoning="데이터(data) 섹션이 문서에서 발견되지 않았습니다"
            )

        pipeline_text = pipeline_text if pipeline_text else "파이프라인 정보 없음"

        prompt_template = self.prompts.get('evaluation', {}).get('data_preprocessing', {})
        system_msg = prompt_template.get('system', '')
        user_msg = prompt_template.get('prompt', '').format(
            data_text=data_text,
            pipeline_text=pipeline_text
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
                    item_id="DATA_PREPROC_001",
                    category="데이터 수집 및 전처리",
                    subcategory="데이터 전처리",
                    question="알고리즘에 적합한 전처리 방법이 제시되었는가?",
                    result=False,
                    confidence=0.3,
                    evidence=[],
                    reasoning="LLM 응답을 파싱할 수 없습니다"
                )

            return ChecklistItem(
                item_id="DATA_PREPROC_001",
                category="데이터 수집 및 전처리",
                subcategory="데이터 전처리",
                question="알고리즘에 적합한 전처리 방법이 제시되었는가?",
                result=result_data.get('result', False),
                confidence=result_data.get('confidence', 0.5),
                evidence=result_data.get('evidence', []),
                reasoning=result_data.get('reasoning', '')
            )

        except Exception as e:
            logger.error(f"Error in preprocessing evaluation: {e}")
            return ChecklistItem(
                item_id="DATA_PREPROC_001",
                category="데이터 수집 및 전처리",
                subcategory="데이터 전처리",
                question="알고리즘에 적합한 전처리 방법이 제시되었는가?",
                result=False,
                confidence=0.0,
                evidence=[],
                reasoning=f"평가 중 오류 발생: {str(e)}"
            )
