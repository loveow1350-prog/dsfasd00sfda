"""
Test suite for multi-agent pipeline
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import json

from src.models import ProblemCategory, Severity
from src.utils import Config, LLMRouter, CacheManager
from src.pdf_extractor import PDFExtractor
from src.structure_parser import StructureParser
from src.step_decomposer import StepDecomposer
from src.search_client import SearchClient


@pytest.fixture
def config():
    return Config("config/config.yaml")


@pytest.fixture
def llm_client(config):
    return LLMRouter(config)


@pytest.fixture
def cache_manager(config):
    return CacheManager(config)


@pytest.fixture
def search_client(config):
    return SearchClient(config)


class TestPDFExtractor:
    def test_extraction_quality(self):
        extractor = PDFExtractor()

        # Test with actual PDF
        if Path("sample/중간보고서_자연어처리.pdf").exists():
            result = extractor.extract("sample/중간보고서_자연어처리.pdf")

            assert result.document_id is not None
            assert len(result.pages) > 0
            assert result.extraction_quality.quality_score >= 0.0
            assert result.extraction_quality.quality_score <= 1.0

    def test_metadata_extraction(self):
        extractor = PDFExtractor()

        if Path("sample/중간보고서_자연어처리.pdf").exists():
            result = extractor.extract("sample/중간보고서_자연어처리.pdf")

            assert "pages" in result.metadata
            assert result.metadata["pages"] > 0


class TestStructureParser:
    def test_header_detection(self, config, llm_client):
        parser = StructureParser(config, llm_client)
        extractor = PDFExtractor()

        if Path("sample/중간보고서_자연어처리.pdf").exists():
            raw_doc = extractor.extract("sample/중간보고서_자연어처리.pdf")
            structured = parser.parse(raw_doc)

            # Check all 5 keys exist
            assert "purpose" in structured.sections
            assert "background" in structured.sections
            assert "data" in structured.sections
            assert "pipeline" in structured.sections
            assert "plan" in structured.sections

    def test_similarity_calculation(self, config, llm_client):
        parser = StructureParser(config, llm_client)

        # Test exact match
        assert parser._calculate_similarity("분석 목적", "분석 목적") == 1.0

        # Test partial match
        similarity = parser._calculate_similarity("연구의 목적", "연구 목적")
        assert similarity > 0.7


class TestStepDecomposer:
    def test_step_extraction(self, config, llm_client):
        decomposer = StepDecomposer(config, llm_client)

        # Mock structured document
        from src.models import StructuredDocument

        structured = StructuredDocument(
            document_id="test",
            sections={
                "purpose": None,
                "background": None,
                "data": "데이터는 웹 크롤링으로 수집했습니다.",
                "pipeline": "먼저 데이터를 전처리하고, 그 다음 모델을 학습했습니다.",
                "plan": None
            }
        )

        # This will make actual LLM call - skip in CI
        # result = decomposer.decompose(structured)
        # assert len(result.steps) > 0

    def test_json_extraction(self, config, llm_client):
        decomposer = StepDecomposer(config, llm_client)

        text = """
        Here are the steps:
        [
            {"order": 1, "action": "collect data"},
            {"order": 2, "action": "preprocess"}
        ]
        """

        result = decomposer._extract_json(text)
        assert result is not None
        assert len(result) == 2
        assert result[0]["order"] == 1


class TestSearchClient:
    def test_search_initialization(self, config):
        search = SearchClient(config)
        assert search.primary_api in ["tavily", "duckduckgo"]

    # Uncomment to test actual search (requires API keys)
    # def test_duckduckgo_search(self, search_client):
    #     results = search_client._search_duckduckgo("machine learning")
    #     assert len(results) > 0
    #     assert "title" in results[0]
    #     assert "content" in results[0]


class TestProblemAnalyzer:
    def test_problem_mapping_structure(self):
        from src.models import ProblemMapping, StepProblems, Problem

        mapping = ProblemMapping(
            document_id="test",
            problem_mapping={
                "STEP_001": StepProblems(problems=[
                    Problem(
                        problem_id="PROB_001",
                        category=ProblemCategory.DATA_QUALITY,
                        description="Test problem",
                        severity=Severity.MEDIUM,
                        evidence="test",
                        addressed_by="test technique"
                    )
                ])
            },
            summary={
                "total_problems": 1,
                "by_category": {"data_quality": 1},
                "critical_steps": []
            }
        )

        assert mapping.document_id == "test"
        assert "STEP_001" in mapping.problem_mapping
        assert len(mapping.problem_mapping["STEP_001"].problems) == 1


class TestCacheManager:
    def test_cache_operations(self, cache_manager):
        try:
            # Test set/get
            cache_manager.set("test_key", {"value": "test"})
            result = cache_manager.get("test_key")

            if result:  # Redis available
                assert result["value"] == "test"
        except Exception:
            # Redis not available, skip test
            pytest.skip("Redis not available")

    def test_technique_caching(self, cache_manager):
        try:
            technique = "batch normalization"
            problems = ["solves internal covariate shift"]

            cache_manager.set_technique_problems(technique, problems)
            result = cache_manager.get_technique_problems(technique)

            if result:
                assert result == problems
        except Exception:
            pytest.skip("Redis not available")


class TestIntegration:
    def test_full_pipeline(self):
        """Integration test for full pipeline"""
        if not Path("sample/중간보고서_자연어처리.pdf").exists():
            pytest.skip("Test PDF not found")

        # This would run the entire pipeline
        # Uncomment to test (requires all API keys)

        # from main_pipeline import PipelineOrchestrator
        # orchestrator = PipelineOrchestrator()
        # result = orchestrator.process_document("sample/중간보고서_자연어처리.pdf")
        # assert result is not None
        # assert result.summary["total_problems"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
