"""
Topic Evaluator
Evaluates topic selection, novelty, and relevance
"""
import json
from typing import List, Optional, Dict

from src.models import ChecklistItem
from src.search_client import SearchClient
from src.utils import LLMRouter, setup_logger, extract_json_object, extract_json_array

logger = setup_logger(__name__)


class TopicEvaluator:
    """Evaluates topic selection and creativity"""

    def __init__(self, llm: LLMRouter, search_client: SearchClient, prompts: Dict):
        self.llm = llm
        self.search = search_client
        self.prompts = prompts

    def evaluate(
        self,
        topic: Optional[str],
        purpose: Optional[str],
        background: Optional[str],
        generated_topic: bool
    ) -> List[ChecklistItem]:
        """
        Evaluate topic selection:
        1. Specificity (구체성)
        2. Novelty (창의성) - search-based
        3. Relevance (적합성)
        """
        logger.info("Starting topic evaluation")

        results = []
        results.append(self._evaluate_specificity(topic, purpose, generated_topic))
        results.append(self._evaluate_novelty(topic, purpose, background))
        results.append(self._evaluate_relevance(topic, purpose, background))

        logger.info(f"Topic evaluation complete: {len(results)} items")
        return results

    def _evaluate_specificity(self, topic: Optional[str], purpose: Optional[str], generated_topic: bool) -> ChecklistItem:
        """
        Evaluate topic specificity using LLM

        Question: 프로젝트 주제가 명확하고 구체적으로 정의되어 있는가?
        """
        logger.info("Evaluating topic specificity")

        if (not purpose or purpose.strip() == "") and generated_topic:
            return ChecklistItem(
                item_id="TOPIC_SPEC_001",
                category="주제 선정 및 창의성",
                subcategory="구체성",
                question="프로젝트 주제가 명확하고 구체적으로 정의되어 있는가?",
                result=False,
                confidence=1.0,
                evidence=[],
                reasoning="목적(purpose) 섹션이 없거나 주제가 자동 생성되어 평가할 수 없습니다"
            )

        purpose_text = purpose if purpose else "목적 정보 없음"

        prompt_template = self.prompts.get('evaluation', {}).get('topic_specificity', {})
        system_msg = prompt_template.get('system', '')
        user_msg = prompt_template.get('prompt', '').format(topic_text=topic, purpose_text=purpose_text)

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
                    item_id="TOPIC_SPEC_001",
                    category="주제 선정 및 창의성",
                    subcategory="구체성",
                    question="프로젝트 주제가 명확하고 구체적으로 정의되어 있는가?",
                    result=False,
                    confidence=0.3,
                    evidence=[],
                    reasoning="LLM 응답을 파싱할 수 없습니다"
                )

            return ChecklistItem(
                item_id="TOPIC_SPEC_001",
                category="주제 선정 및 창의성",
                subcategory="구체성",
                question="프로젝트 주제가 명확하고 구체적으로 정의되어 있는가?",
                result=result_data.get('result', False),
                confidence=result_data.get('confidence', 0.5),
                evidence=result_data.get('evidence', []),
                reasoning=result_data.get('reasoning', '')
            )

        except Exception as e:
            logger.error(f"Error in specificity evaluation: {e}")
            return ChecklistItem(
                item_id="TOPIC_SPEC_001",
                category="주제 선정 및 창의성",
                subcategory="구체성",
                question="프로젝트 주제가 명확하고 구체적으로 정의되어 있는가?",
                result=False,
                confidence=0.0,
                evidence=[],
                reasoning=f"평가 중 오류 발생: {str(e)}"
            )

    def _evaluate_novelty(self, topic: Optional[str], purpose: Optional[str], background: Optional[str]) -> ChecklistItem:
        """
        Evaluate novelty/creativity using 3-stage pipeline:
        1. SLM generates search queries
        2. Search API executes queries
        3. Eval LLM analyzes similarity of results

        Question: 기존에 연구된 사례가 적고 새로운 접근 방식인가?
        """
        logger.info("Evaluating topic novelty (3-stage pipeline)")

        try:
            # ===== Stage 1: Query Generation (SLM) =====
            logger.info("Stage 1: Generating search queries with SLM")
            prompt_template = self.prompts.get('evaluation', {}).get('topic_novelty_query_generation', {})
            system_msg = prompt_template.get('system', '')
            user_msg = prompt_template.get('prompt', '').format(
                topic_text=topic or "주제 정보 없음",
                purpose_text=purpose or "목적 정보 없음",
                background_text=background or "배경 정보 없음"
            )

            query_response = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                complexity="low"  # Use small model for query generation
            )

            queries = extract_json_array(query_response)
            if not queries:
                logger.warning("Failed to extract queries from SLM, using fallback")
                queries = [f"{topic} research"] if topic else ["AI research"]

            logger.info(f"Generated {len(queries)} search queries: {queries}")

            # ===== Stage 2: Search Execution =====
            logger.info("Stage 2: Executing search queries")
            all_results = []
            for query in queries[:2]:  # Use top 2 queries to avoid too many results
                results = self.search.search(query, max_results=self.search.eval_max_results)
                all_results.extend(results)

            # Remove duplicates by URL
            seen_urls = set()
            unique_results = []
            for r in all_results:
                if r['url'] not in seen_urls:
                    seen_urls.add(r['url'])
                    unique_results.append(r)

            logger.info(f"Found {len(unique_results)} unique search results")

            # If no results found, it's potentially novel
            if not unique_results:
                return ChecklistItem(
                    item_id="TOPIC_NOVEL_001",
                    category="주제 선정 및 창의성",
                    subcategory="창의성",
                    question="기존에 연구된 사례가 적고 새로운 접근 방식인가?",
                    result=True,
                    confidence=0.6,
                    evidence=["검색 결과 없음 - 새로운 주제일 가능성"],
                    reasoning="유사한 연구를 찾을 수 없어 새로운 접근으로 판단됩니다",
                    search_results=[]
                )

            # ===== Stage 3: Similarity Analysis (Eval LLM) =====
            logger.info("Stage 3: Analyzing similarity with Eval LLM")

            # Prepare search results as JSON for LLM
            import json
            search_results_json = json.dumps([
                {
                    "index": i,
                    "title": r['title'],
                    "content": r['content'][:300]  # First 300 chars only
                }
                for i, r in enumerate(unique_results)
            ], ensure_ascii=False, indent=2)

            similarity_template = self.prompts.get('evaluation', {}).get('topic_novelty_similarity_analysis', {})
            similarity_system = similarity_template.get('system', '')
            similarity_user = similarity_template.get('prompt', '').format(
                topic_text=topic or "주제 정보 없음",
                purpose_text=purpose or "목적 정보 없음",
                search_results_json=search_results_json
            )

            similarity_response = self.llm.chat_eval(
                messages=[
                    {"role": "system", "content": similarity_system},
                    {"role": "user", "content": similarity_user}
                ],
                response_format="json"
            )

            similarity_data = extract_json_object(similarity_response)

            if not similarity_data:
                logger.warning("Failed to parse similarity analysis, using count fallback")
                return self._novelty_count_fallback(unique_results)

            # ===== Process Results =====
            overall = similarity_data.get('overall_novelty', {})
            analysis = similarity_data.get('analysis', [])

            # Build evidence list
            evidence = [
                f"검색된 연구: {len(unique_results)}건",
                f"평균 유사도: {overall.get('avg_similarity', 0):.2f}",
                f"높은 유사도 (≥0.7): {overall.get('highly_similar_count', 0)}건"
            ]

            # Store detailed results in search_results field
            detailed_results = []
            for a in analysis:
                idx = a.get('result_index')
                if idx is not None and idx < len(unique_results):
                    detailed_results.append({
                        "title": unique_results[idx]['title'],
                        "url": unique_results[idx]['url'],
                        "similarity_score": a.get('similarity_score', 0.0),
                        "reasoning": a.get('reasoning', '')
                    })

            result = overall.get('result', False)
            confidence = overall.get('confidence', 0.5)

            return ChecklistItem(
                item_id="TOPIC_NOVEL_001",
                category="주제 선정 및 창의성",
                subcategory="창의성",
                question="기존에 연구된 사례가 적고 새로운 접근 방식인가?",
                result=result,
                confidence=confidence,
                evidence=evidence,
                reasoning=overall.get('reasoning', ''),
                search_results=detailed_results
            )

        except Exception as e:
            logger.error(f"Error in novelty evaluation: {e}", exc_info=True)
            return ChecklistItem(
                item_id="TOPIC_NOVEL_001",
                category="주제 선정 및 창의성",
                subcategory="창의성",
                question="기존에 연구된 사례가 적고 새로운 접근 방식인가?",
                result=False,
                confidence=0.0,
                evidence=[],
                reasoning=f"평가 중 오류 발생: {str(e)}"
            )

    def _novelty_count_fallback(self, results: List[Dict]) -> ChecklistItem:
        """
        Fallback method: Simple count-based novelty evaluation
        Used when similarity analysis fails
        """
        num_results = len(results)

        if num_results <= 5:
            result = True
            confidence = 0.7
            reasoning = f"유사한 연구가 {num_results}건으로 적어 창의적인 주제로 판단됩니다"
        elif num_results <= 15:
            result = False
            confidence = 0.6
            reasoning = f"유사한 연구가 {num_results}건 발견되어 일부 선행 연구가 존재합니다"
        else:
            result = False
            confidence = 0.8
            reasoning = f"유사한 연구가 {num_results}건 이상 발견되어 기존 연구가 많이 존재합니다"

        evidence = [f"검색 결과 {num_results}건 (유사도 분석 실패, 개수 기반 평가)"]

        return ChecklistItem(
            item_id="TOPIC_NOVEL_001",
            category="주제 선정 및 창의성",
            subcategory="창의성",
            question="기존에 연구된 사례가 적고 새로운 접근 방식인가?",
            result=result,
            confidence=confidence,
            evidence=evidence,
            reasoning=reasoning,
            search_results=[
                {"title": r['title'], "url": r['url'], "similarity_score": None, "reasoning": "분석 실패"}
                for r in results[:5]  # Store first 5 only
            ]
        )

    def _evaluate_relevance(self, topic: Optional[str], purpose: Optional[str], background: Optional[str]) -> ChecklistItem:
        """
        Evaluate topic relevance using LLM
        TODO: 실제 프로젝트 내용과의 연관성 평가로 확장

        Question: 선택한 분야가 프로젝트의 핵심 주제인가?
        """
        logger.info("Evaluating topic relevance")

        purpose_text = purpose if purpose else "목적 정보 없음"
        background_text = background if background else "배경 정보 없음"

        prompt_template = self.prompts.get('evaluation', {}).get('topic_relevance', {})
        system_msg = prompt_template.get('system', '')
        user_msg = prompt_template.get('prompt', '').format(
            topic_text=topic,
            purpose_text=purpose_text,
            background_text=background_text
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
                    item_id="TOPIC_RELEV_001",
                    category="주제 선정 및 창의성",
                    subcategory="적합성",
                    question="선택한 분야가 프로젝트의 핵심 주제인가?",
                    result=False,
                    confidence=0.3,
                    evidence=[],
                    reasoning="LLM 응답을 파싱할 수 없습니다"
                )

            return ChecklistItem(
                item_id="TOPIC_RELEV_001",
                category="주제 선정 및 창의성",
                subcategory="적합성",
                question="선택한 분야가 프로젝트의 핵심 주제인가?",
                result=result_data.get('result', False),
                confidence=result_data.get('confidence', 0.5),
                evidence=result_data.get('evidence', []),
                reasoning=result_data.get('reasoning', '')
            )

        except Exception as e:
            logger.error(f"Error in relevance evaluation: {e}")
            return ChecklistItem(
                item_id="TOPIC_RELEV_001",
                category="주제 선정 및 창의성",
                subcategory="적합성",
                question="선택한 분야가 프로젝트의 핵심 주제인가?",
                result=False,
                confidence=0.0,
                evidence=[],
                reasoning=f"평가 중 오류 발생: {str(e)}"
            )
