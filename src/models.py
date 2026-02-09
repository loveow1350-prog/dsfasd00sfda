"""
Data models for multi-agent pipeline
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum



class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# PyMuPDF Component Output
class TextBlock(BaseModel):
    bbox: List[float] = Field(description="Bounding box [x0, y0, x1, y0]")
    text: str
    font: str
    size: float
    flags: int = Field(description="Font flags (bold, italic bits)")


class PageData(BaseModel):
    page_num: int
    blocks: List[TextBlock]
    image_data: Optional[bytes] = Field(
        default=None,
        description="Page rendered as PNG image bytes for VLM",
        exclude=True  # Exclude from JSON serialization
    )


class ExtractionQuality(BaseModel):
    missing_pages: List[int] = Field(default_factory=list)
    encoding_errors: int = 0
    quality_score: float = Field(ge=0.0, le=1.0)


class RawDocument(BaseModel):
    document_id: str
    metadata: Dict[str, Any]
    pages: List[PageData]
    extraction_quality: ExtractionQuality


# Structure Parser Output
class HeaderMetadata(BaseModel):
    page_range: List[int]
    confidence: float = Field(ge=0.0, le=1.0)


class StructuredDocument(BaseModel):
    document_id: str
    sections: Dict[str, Optional[str]] = Field(
        description="6 fixed keys: topic, purpose, background, data, pipeline, plan"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "missing_sections": [],
            "total_chunks": None,
            "chunk_classifications": None,
            "generated_topic": False
        }
    )


# Step Abstraction Output
class Step(BaseModel):
    step_id: str
    order: int
    category: str = Field(description="Category/type of this step (free-form string)")
    action: str = Field(description="One-sentence description of the action")
    input: str = Field(description="Input data/state")
    output: str = Field(description="Output data/state")
    techniques: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list, description="List of step_ids")
    parallel_group: Optional[str] = None


class SequentialSteps(BaseModel):
    document_id: str
    steps: List[Step]
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "total_steps": 0,
            "category_counts": {},
            "unique_categories": []
        }
    )


# Problem Definition Output
class Problem(BaseModel):
    problem_id: str
    category: str = Field(description="Problem category (free-form string)")
    description: str
    severity: Severity
    evidence: str = Field(description="Evidence from document")
    addressed_by: str = Field(description="Technique addressing this problem")
    confidence: str = Field(default="inferred", description="derived or inferred")


class StepProblems(BaseModel):
    problems: List[Problem]


class ProblemMapping(BaseModel):
    document_id: str
    problem_mapping: Dict[str, StepProblems] = Field(
        description="Map from step_id to problems"
    )
    summary: Dict[str, Any] = Field(
        default_factory=lambda: {
            "total_problems": 0,
            "by_category": {},
            "critical_steps": []
        }
    )
    problem_index: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Reverse index: problem_id -> [step_ids]"
    )


# Pipeline State
class PipelineStatus(BaseModel):
    document_id: str
    current_agent: str
    progress: int = Field(ge=0, le=100)
    timestamp: str
    errors: List[str] = Field(default_factory=list)


# Evaluation Models
class ChecklistItem(BaseModel):
    """개별 체크리스트 항목"""
    item_id: str = Field(description="예: TOPIC_SPEC_001")
    category: str = Field(description="예: 주제 선정 및 창의성")
    subcategory: str = Field(description="예: 구체성")
    question: str = Field(description="평가 질문")
    result: bool = Field(description="Yes/No 결과")
    confidence: float = Field(ge=0.0, le=1.0, description="0.0 ~ 1.0")
    evidence: List[str] = Field(default_factory=list, description="근거 텍스트")
    reasoning: str = Field(description="판단 근거 설명")
    search_results: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="검색 결과 (해당시)"
    )


class CategoryEvaluation(BaseModel):
    """카테고리별 평가 결과"""
    category: str = Field(description="예: 프로젝트 설계")
    weight: float = Field(description="가중치 (예: 0.2)")
    checklist_items: List[ChecklistItem]
    pass_count: int = Field(description="Yes 개수")
    total_count: int = Field(description="전체 항목 수")
    pass_rate: float = Field(description="통과율")
    score: float = Field(description="점수 (가중치 적용)")


class EvaluationReport(BaseModel):
    """최종 평가 리포트"""
    document_id: str
    timestamp: str
    categories: List[CategoryEvaluation]
    total_score: float = Field(description="0 ~ 100")
    overall_pass_rate: float = Field(description="전체 통과율")
    summary: Dict[str, Any] = Field(default_factory=dict, description="요약 정보")
    recommendations: List[str] = Field(default_factory=list, description="개선 권장사항")

