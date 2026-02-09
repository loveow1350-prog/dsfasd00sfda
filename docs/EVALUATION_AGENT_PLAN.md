# 평가 에이전트 구현 계획서 (Evaluation Agent Implementation Plan)

## 📋 개요

본 문서는 중간보고서 평가를 자동화하기 위한 **평가 에이전트(Evaluation Agent)** 시스템의 구체적인 구현 계획을 제시합니다.

**목표**: 루브릭 기반 체크리스트 형식의 Yes/No 평가 시스템 구축

---

## 🎯 평가 루브릭 구조

### 1. 프로젝트 설계 (20%)
#### 1.1 주제 선정 및 창의성 (20%)
- **구체성**: 주제가 모호하지 않고 구체적인가? (작성한 주제 기반)
- **창의성**: 기존에 연구된 사례가 다수 존재하는가? (검색 사용)
  - 정확한 주제 명시 X -> LLM 생성 대체 주제 사용
- **적합성**: 선택한 분야가 메인이 되는 주제인가?
  - 정확한 주제 명시 X -> LLM 생성 대체 주제 사용

### 2. 프로젝트 구현 (50%)
#### 2.1 데이터 수집 및 전처리 (10%)
- **데이터 확보 여부**: 분석에 활용할 데이터를 확보하였는가?
- **데이터 활용**: 데이터를 어떻게 활용할지에 대한 방법이 구체적으로 정해졌는가?
- **데이터 전처리**: 알고리즘에 맞는 전처리 방법을 사용하였는가?

#### 2.2 알고리즘 설계 (20%)
- **구체성**: 단계별 프레임워크를 제시하였는가?
- **적합성**: 프레임워크의 출력이 달성하고자 하는 목적에 부합하는가?
  - 정확한 주제 명시 X -> LLM 생성 대체 주제 사용

#### 2.3 문제 해결력 (20%)
- **목적성**: 각 단계의 프레임워크를 사용한 이유 혹은 의도가 구체적으로 명시되었는가?
- **타당성**: 단계마다 사용된 기술이 문제해결에 있어 타당한가? (검색 사용)

### 3. 향후 계획
#### 3.1 계획의 구체성
- **구체성**: 날짜에 따라 구체적인 계획이 작성되었는가?

---

## 🗂️ 데이터 매핑 전략

기존 파이프라인에서 추출된 데이터와 평가 항목을 매핑합니다.

### 데이터 소스별 활용 계획

| 평가 영역            | 평가 항목 | 데이터 소스 | 데이터 필드                                                                                |
|------------------|---------|-----------|---------------------------------------------------------------------------------------|
| **주제 선정 및 창의성**  | 구체성 | `structured_document.json` | `sections.topic`+`sections.purpose`                                                   |
|                  | 창의성 | `structured_document.json` + **검색 API** | `sections.purpose` + `sections.background` + `sections.topic(LLM generation allowed)` |
|                  | 적합성 | `structured_document.json` | `sections.purpose` + `sections.background` + `sections.topic(LLM generation allowed)` |
| **데이터 수집 및 전처리** | 데이터 확보 여부 | `structured_document.json` | `sections.data`                                                                       |
|                  | 데이터 활용 | `structured_document.json` | `sections.data` + `sections.pipeline`                                                 |
|                  | 데이터 전처리 | `structured_document.json` | `sections.data` + `sections.pipeline`                                                 |
| **알고리즘 설계**      | 구체성 | `sequential_steps.json` | `steps[]` (전체 단계 목록)                                                                  |
|                  | 적합성 | `sequential_steps.json` + `structured_document.json` | `steps[]` + `sections.purpose` + `sections.topic(LLM generation allowed)`                                                     |
| **문제 해결력**       | 목적성 | `problem_mapping.json` + `raw_document.json` | `problem_mapping[step_id]` + 해당 step의 청크                                              |
|                  | 타당성 | `problem_mapping.json` + **검색 API** | `problem_mapping[step_id].problems[].addressed_by`                                    |
| **향후 계획**        | 계획의 구체성 | `structured_document.json` | `sections.plan`                                                                       |

---

## 🏗️ 시스템 아키텍처

### 전체 구조

```
EvaluationOrchestrator
├── DataLoader (데이터 로더)
│   ├── load_structured_document()
│   ├── load_sequential_steps()
│   ├── load_problem_mapping()
│   └── load_raw_document()
│
├── EvaluationAgent (평가 에이전트)
│   ├── TopicEvaluator (주제 선정 평가자)
│   ├── DataEvaluator (데이터 평가자)
│   ├── AlgorithmEvaluator (알고리즘 평가자)
│   ├── ProblemSolvingEvaluator (문제해결 평가자)
│   └── PlanEvaluator (계획 평가자)
│
├── SearchEnhancer (검색 강화 모듈)
│   ├── check_novelty() - 창의성 검색
│   └── validate_technique() - 타당성 검색
│
└── ReportGenerator (리포트 생성기)
    ├── generate_checklist()
    └── generate_detailed_feedback()
```

---

## 📦 데이터 모델 설계

### 1. ChecklistItem (체크리스트 항목)

```python
class ChecklistItem(BaseModel):
    """개별 체크리스트 항목"""
    item_id: str                    # 예: "TOPIC_SPEC_001"
    category: str                   # 예: "주제 선정 및 창의성"
    subcategory: str                # 예: "구체성"
    question: str                   # 평가 질문
    result: bool                    # Yes/No 결과
    confidence: float               # 0.0 ~ 1.0
    evidence: List[str]             # 근거 텍스트 (문서에서 추출)
    reasoning: str                  # 판단 근거 설명
    search_results: Optional[List[Dict]] = None  # 검색 결과 (해당시)
```

### 2. CategoryEvaluation (카테고리별 평가)

```python
class CategoryEvaluation(BaseModel):
    """카테고리별 평가 결과"""
    category: str                   # 예: "프로젝트 설계"
    weight: float                   # 가중치 (예: 0.2)
    checklist_items: List[ChecklistItem]
    pass_count: int                 # Yes 개수
    total_count: int                # 전체 항목 수
    pass_rate: float                # 통과율
    score: float                    # 점수 (가중치 적용)
```

### 3. EvaluationReport (최종 평가 리포트)

```python
class EvaluationReport(BaseModel):
    """최종 평가 리포트"""
    document_id: str
    timestamp: str
    categories: List[CategoryEvaluation]
    total_score: float              # 0 ~ 100
    overall_pass_rate: float        # 전체 통과율
    summary: Dict[str, Any]         # 요약 정보
    recommendations: List[str]      # 개선 권장사항
```

---

## 🔧 구현 단계별 상세 계획

### Phase 1: 데이터 로더 구현 (DataLoader)

**파일**: `src/evaluation_data_loader.py`

#### 기능
1. **기존 JSON 파일 로드**
   - `structured_document.json` → `StructuredDocument`
   - `sequential_steps.json` → `SequentialSteps`
   - `problem_mapping.json` → `ProblemMapping`
   - `raw_document.json` → `RawDocument`

2. **데이터 추출 헬퍼 함수**
   - `extract_purpose_text()`: purpose 섹션 추출
   - `extract_background_text()`: background 섹션 추출
   - `extract_data_text()`: data 섹션 추출
   - `extract_pipeline_text()`: pipeline 섹션 추출
   - `extract_plan_text()`: plan 섹션 추출
   - `get_step_chunks()`: 특정 step_id에 해당하는 원본 청크 추출

3. **청크-Step 매핑 로직**
   - `raw_document.json`의 페이지 블록과 `sequential_steps.json`의 단계를 연결
   - Step에서 언급된 technique/action과 매칭되는 텍스트 블록 찾기
   - **방법**: 
     - `sequential_steps.json`의 각 step의 `action`, `techniques` 키워드 추출
     - `raw_document.json`의 각 페이지 블록에서 해당 키워드 검색
     - 유사도 기반 매칭 (LLM 또는 임베딩 사용)

#### 구현 예시 구조
```python
class EvaluationDataLoader:
    def __init__(self, output_dir: str, document_id: str):
        self.output_dir = Path(output_dir)
        self.document_id = document_id
        
    def load_all_data(self) -> Dict[str, Any]:
        """모든 필요한 데이터 로드"""
        return {
            'structured': self._load_structured_document(),
            'steps': self._load_sequential_steps(),
            'problems': self._load_problem_mapping(),
            'raw': self._load_raw_document()
        }
    
    def get_step_related_chunks(self, step_id: str) -> List[str]:
        """특정 step과 관련된 원본 텍스트 청크 반환"""
        # 구현 로직
        pass
```

---

### Phase 2: 개별 평가자 구현 (Evaluators)

각 평가 영역별로 전문화된 평가자를 구현합니다.

#### 2.1 TopicEvaluator (주제 선정 평가자)

**파일**: `src/evaluators/topic_evaluator.py`

**입력 데이터**:
- `sections.purpose`
- `sections.background`

**평가 항목**:
1. **구체성 평가**
   - **질문**: "프로젝트 주제가 명확하고 구체적으로 정의되어 있는가?"
   - **평가 기준**:
     - Yes: 구체적인 목표, 대상, 방법이 명시됨
     - No: 추상적이거나 모호한 표현만 존재
   - **LLM 프롬프트**:
     ```
     다음 프로젝트 목적을 분석하여 주제가 구체적인지 평가하세요.
     
     [목적 텍스트]
     {purpose_text}
     
     평가 기준:
     - 구체적인 목표가 명시되어 있는가?
     - 대상 문제/도메인이 명확한가?
     - 해결 방법의 개요가 제시되어 있는가?
     
     JSON 형식으로 응답:
     {
       "result": true/false,
       "confidence": 0.0-1.0,
       "evidence": ["근거1", "근거2"],
       "reasoning": "판단 이유"
     }
     ```

2. **창의성 평가 (검색 기반)**
   - **질문**: "기존 연구가 적고 새로운 접근 방식인가?"
   - **평가 기준**:
     - Yes: 기존 사례가 5개 미만 또는 새로운 조합
     - No: 기존 사례가 많이 존재
   - **프로세스**:
     1. purpose에서 핵심 키워드 추출 (LLM)
     2. 검색 API로 유사 연구 검색 (Tavily/Perplexity)
     3. 검색 결과 개수 및 유사도 분석
     4. LLM으로 창의성 판단

3. **적합성 평가**
   - **질문**: "선택한 분야가 프로젝트의 핵심 주제인가?"
   - **평가 기준**:
     - Yes: 주제와 방법론이 일치
     - No: 부차적인 요소가 주제로 제시됨

**구현 구조**:
```python
class TopicEvaluator:
    def __init__(self, llm: LLMRouter, search: SearchClient, prompts: Dict):
        self.llm = llm
        self.search = search
        self.prompts = prompts
    
    def evaluate(self, purpose: str, background: Optional[str]) -> List[ChecklistItem]:
        """주제 선정 평가 실행"""
        items = []
        
        # 1. 구체성
        items.append(self._evaluate_specificity(purpose))
        
        # 2. 창의성 (검색)
        items.append(self._evaluate_novelty(purpose, background))
        
        # 3. 적합성
        items.append(self._evaluate_relevance(purpose, background))
        
        return items
    
    def _evaluate_specificity(self, purpose: str) -> ChecklistItem:
        """구체성 평가"""
        # LLM 호출
        pass
    
    def _evaluate_novelty(self, purpose: str, background: Optional[str]) -> ChecklistItem:
        """창의성 평가 (검색 기반)"""
        # 1. 키워드 추출
        # 2. 검색
        # 3. 결과 분석
        pass
```

---

#### 2.2 DataEvaluator (데이터 평가자)

**파일**: `src/evaluators/data_evaluator.py`

**입력 데이터**:
- `sections.data`
- `sections.pipeline`

**평가 항목**:
1. **데이터 확보 여부**
   - **질문**: "분석에 필요한 데이터가 확보되었거나 확보 계획이 구체적인가?"
   - **평가 기준**:
     - Yes: 데이터셋 이름/출처 명시 또는 수집 방법 구체적
     - No: 데이터 언급 없음 또는 매우 추상적

2. **데이터 활용 계획**
   - **질문**: "데이터를 어떻게 활용할지 구체적으로 정해졌는가?"
   - **평가 기준**:
     - Yes: 입력/출력 형태, 사용 단계 명시
     - No: 활용 방법 미언급

3. **데이터 전처리**
   - **질문**: "알고리즘에 적합한 전처리 방법이 제시되었는가?"
   - **평가 기준**:
     - Yes: 구체적인 전처리 기법 명시
     - No: 전처리 언급 없음

**특수 케이스**: `sections.data`가 null인 경우
- 모든 항목에 대해 `result: false`
- `reasoning`에 "데이터 섹션이 문서에서 발견되지 않았습니다" 명시

---

#### 2.3 AlgorithmEvaluator (알고리즘 평가자)

**파일**: `src/evaluators/algorithm_evaluator.py`

**입력 데이터**:
- `sequential_steps.json` → `steps[]` 리스트
- `sections.purpose`

**평가 항목**:
1. **구체성 (단계별 프레임워크)**
   - **질문**: "단계별 프레임워크가 구체적으로 제시되었는가?"
   - **평가 기준**:
     - Yes: 3개 이상의 단계, 각 단계의 입출력 명시
     - No: 단계가 모호하거나 2개 이하

2. **적합성 (목적 부합)**
   - **질문**: "프레임워크의 출력이 프로젝트 목적에 부합하는가?"
   - **평가 기준**:
     - Yes: 마지막 단계의 출력이 목적과 일치
     - No: 출력과 목적의 불일치

**구현 로직**:
```python
class AlgorithmEvaluator:
    def evaluate(self, steps: List[Step], purpose: str) -> List[ChecklistItem]:
        items = []
        
        # 1. 구체성: 단계 수, 입출력 명확성 체크
        specificity = self._check_framework_specificity(steps)
        items.append(specificity)
        
        # 2. 적합성: 최종 출력과 목적 비교 (LLM)
        relevance = self._check_framework_relevance(steps, purpose)
        items.append(relevance)
        
        return items
    
    def _check_framework_specificity(self, steps: List[Step]) -> ChecklistItem:
        """구체성 체크 - 룰 기반 + LLM"""
        # 단계 수 체크
        has_enough_steps = len(steps) >= 3
        
        # 각 단계에 입출력 있는지 체크
        all_have_io = all(step.input and step.output for step in steps)
        
        # LLM으로 최종 판단
        pass
```

---

#### 2.4 ProblemSolvingEvaluator (문제해결 평가자)

**파일**: `src/evaluators/problem_solving_evaluator.py`

**입력 데이터**:
- `problem_mapping.json`
- `raw_document.json` (청크 매핑용)

**평가 항목**:
1. **목적성**
   - **질문**: "각 단계의 프레임워크를 사용한 이유가 구체적으로 명시되었는가?"
   - **평가 기준**:
     - Yes: 각 step에 대응하는 problem이 명확히 정의됨
     - No: 이유 없이 기술만 나열

2. **타당성 (검색 기반)**
   - **질문**: "사용된 기술이 문제 해결에 타당한가?"
   - **평가 기준**:
     - Yes: 검색 결과 해당 기술이 문제 해결에 적합
     - No: 부적절하거나 검증되지 않은 기술

**구현 로직**:
```python
class ProblemSolvingEvaluator:
    def evaluate(
        self, 
        problem_mapping: Dict[str, StepProblems],
        steps: List[Step],
        data_loader: EvaluationDataLoader
    ) -> List[ChecklistItem]:
        items = []
        
        # 1. 목적성: problem과 step의 연결성
        purposefulness = self._evaluate_purposefulness(problem_mapping, steps, data_loader)
        items.append(purposefulness)
        
        # 2. 타당성: 기술-문제 매칭 (검색)
        validity = self._evaluate_validity(problem_mapping, steps)
        items.append(validity)
        
        return items
    
    def _evaluate_purposefulness(self, ...) -> ChecklistItem:
        """각 단계의 사용 이유가 명시되었는지"""
        # 각 step_id에 대응하는 problem 존재 여부 확인
        # 원본 청크에서 "이유", "목적", "위해" 등의 표현 확인
        pass
    
    def _evaluate_validity(self, problem_mapping, steps) -> ChecklistItem:
        """기술의 타당성 검증 (검색)"""
        # 각 problem의 addressed_by 추출
        # "기술명 + 문제 설명"으로 검색
        # 검색 결과에서 적합성 확인
        pass
```

---

#### 2.5 PlanEvaluator (계획 평가자)

**파일**: `src/evaluators/plan_evaluator.py`

**입력 데이터**:
- `sections.plan`

**평가 항목**:
1. **계획의 구체성**
   - **질문**: "날짜와 함께 구체적인 계획이 작성되었는가?"
   - **평가 기준**:
     - Yes: 날짜/기간 + 구체적인 작업 항목
     - No: 날짜 없음 또는 추상적인 계획만 존재

**구현 로직**:
```python
class PlanEvaluator:
    def evaluate(self, plan: Optional[str]) -> List[ChecklistItem]:
        items = []
        
        # 구체성: 날짜 표현 + 작업 내용 확인
        specificity = self._evaluate_plan_specificity(plan)
        items.append(specificity)
        
        return items
    
    def _evaluate_plan_specificity(self, plan: Optional[str]) -> ChecklistItem:
        """날짜 패턴 + LLM으로 구체성 평가"""
        if not plan:
            return ChecklistItem(
                item_id="PLAN_SPEC_001",
                category="향후 계획",
                subcategory="구체성",
                question="날짜에 따라 구체적인 계획이 작성되었는가?",
                result=False,
                confidence=1.0,
                evidence=[],
                reasoning="계획 섹션이 문서에서 발견되지 않았습니다"
            )
        
        # 날짜 패턴 찾기
        import re
        date_patterns = [
            r'\d{4}[-./]\d{1,2}[-./]\d{1,2}',  # 2024-01-15
            r'\d{1,2}월\s*\d{1,2}일',           # 1월 15일
            r'\d+주차',                         # 3주차
            # ... 더 많은 패턴
        ]
        
        # LLM으로 최종 판단
        pass
```

---

### Phase 3: 검색 강화 모듈 (SearchEnhancer)

**파일**: `src/evaluators/search_enhancer.py`

검색이 필요한 평가 항목을 지원합니다.

#### 기능
1. **창의성 검색** (`check_novelty`)
   - 입력: 프로젝트 주제/목적 텍스트
   - 출력: 유사 연구 개수, 검색 결과 요약
   
2. **타당성 검색** (`validate_technique`)
   - 입력: 기술명, 문제 설명
   - 출력: 적합성 점수, 근거 문서

```python
class SearchEnhancer:
    def __init__(self, search_client: SearchClient, llm: LLMRouter):
        self.search = search_client
        self.llm = llm
    
    def check_novelty(self, purpose: str) -> Dict[str, Any]:
        """창의성 검색"""
        # 1. 핵심 키워드 추출
        keywords = self._extract_keywords(purpose)
        
        # 2. 검색 수행
        results = self.search.search(
            query=f"{keywords} research papers projects",
            max_results=20
        )
        
        # 3. 유사도 분석
        similarity_scores = self._analyze_similarity(purpose, results)
        
        return {
            "total_results": len(results),
            "high_similarity_count": sum(1 for s in similarity_scores if s > 0.7),
            "is_novel": len([s for s in similarity_scores if s > 0.7]) < 5,
            "search_results": results[:5]
        }
    
    def validate_technique(self, technique: str, problem_desc: str) -> Dict[str, Any]:
        """기술의 타당성 검증"""
        # 검색 쿼리 생성
        query = f"{technique} for {problem_desc}"
        
        # 검색
        results = self.search.search(query, max_results=10)
        
        # LLM으로 적합성 판단
        validation = self._llm_validate_technique(technique, problem_desc, results)
        
        return validation
```

---

### Phase 4: 평가 오케스트레이터 (EvaluationOrchestrator)

**파일**: `src/evaluation_orchestrator.py`

모든 평가자를 조율하고 최종 리포트를 생성합니다.

```python
class EvaluationOrchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMRouter(config)
        self.search = SearchClient(config)
        self.prompts = load_prompts()
        
        # 평가자 초기화
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
        """전체 평가 실행"""
        logger.info(f"Starting evaluation for {document_id}")
        
        # 1. 데이터 로드
        data_loader = EvaluationDataLoader(output_dir, document_id)
        data = data_loader.load_all_data()
        
        # 2. 각 카테고리 평가
        categories = []
        
        # 2.1 주제 선정 및 창의성 (20%)
        topic_items = self.topic_evaluator.evaluate(
            purpose=data['structured'].sections.get('purpose'),
            background=data['structured'].sections.get('background')
        )
        categories.append(self._create_category_eval(
            "주제 선정 및 창의성", 0.2, topic_items
        ))
        
        # 2.2 데이터 수집 및 전처리 (10%)
        data_items = self.data_evaluator.evaluate(
            data_text=data['structured'].sections.get('data'),
            pipeline_text=data['structured'].sections.get('pipeline')
        )
        categories.append(self._create_category_eval(
            "데이터 수집 및 전처리", 0.1, data_items
        ))
        
        # 2.3 알고리즘 설계 (20%)
        algo_items = self.algorithm_evaluator.evaluate(
            steps=data['steps'].steps,
            purpose=data['structured'].sections.get('purpose')
        )
        categories.append(self._create_category_eval(
            "알고리즘 설계", 0.2, algo_items
        ))
        
        # 2.4 문제 해결력 (20%)
        problem_items = self.problem_evaluator.evaluate(
            problem_mapping=data['problems'].problem_mapping,
            steps=data['steps'].steps,
            data_loader=data_loader
        )
        categories.append(self._create_category_eval(
            "문제 해결력", 0.2, problem_items
        ))
        
        # 2.5 향후 계획
        plan_items = self.plan_evaluator.evaluate(
            plan=data['structured'].sections.get('plan')
        )
        categories.append(self._create_category_eval(
            "향후 계획", 0.0, plan_items  # 가중치 없음
        ))
        
        # 3. 최종 점수 계산
        total_score = sum(cat.score for cat in categories)
        
        # 4. 리포트 생성
        report = EvaluationReport(
            document_id=document_id,
            timestamp=datetime.now().isoformat(),
            categories=categories,
            total_score=total_score,
            overall_pass_rate=self._calculate_overall_pass_rate(categories),
            summary=self._generate_summary(categories),
            recommendations=self._generate_recommendations(categories)
        )
        
        return report
    
    def _create_category_eval(
        self, 
        category: str, 
        weight: float, 
        items: List[ChecklistItem]
    ) -> CategoryEvaluation:
        """카테고리 평가 객체 생성"""
        pass_count = sum(1 for item in items if item.result)
        total = len(items)
        pass_rate = pass_count / total if total > 0 else 0.0
        score = pass_rate * weight * 100  # 0-100 스케일
        
        return CategoryEvaluation(
            category=category,
            weight=weight,
            checklist_items=items,
            pass_count=pass_count,
            total_count=total,
            pass_rate=pass_rate,
            score=score
        )
```

---

### Phase 5: 리포트 생성기 (ReportGenerator)

**파일**: `src/evaluation_report_generator.py`

평가 결과를 다양한 형식으로 출력합니다.

#### 출력 형식
1. **JSON 형식** (`_evaluation_report.json`)
2. **Markdown 체크리스트** (`_evaluation_checklist.md`)
3. **상세 피드백 리포트** (`_evaluation_feedback.md`)

```python
class ReportGenerator:
    def generate_checklist_markdown(self, report: EvaluationReport) -> str:
        """체크리스트 마크다운 생성"""
        md = f"""# 평가 체크리스트 (Evaluation Checklist)

**문서 ID**: {report.document_id}
**평가 일시**: {report.timestamp}
**총점**: {report.total_score:.1f} / 100

---

"""
        for category in report.categories:
            md += f"\n## {category.category} ({category.pass_count}/{category.total_count} 통과)\n\n"
            
            for item in category.checklist_items:
                status = "✅" if item.result else "❌"
                md += f"{status} **{item.subcategory}**: {item.question}\n"
                md += f"   - **판단**: {item.reasoning}\n"
                if item.evidence:
                    md += f"   - **근거**: {item.evidence[0][:100]}...\n"
                md += "\n"
        
        return md
    
    def generate_detailed_feedback(self, report: EvaluationReport) -> str:
        """상세 피드백 리포트 생성"""
        # 개선 필요 항목 상세 설명
        # 강점 분석
        # 구체적 권장사항
        pass
```

---

## 🔄 실행 흐름

### 전체 실행 순서

```
1. main_pipeline.py 실행
   ├── PDF 추출
   ├── 구조 파싱
   ├── 단계 분해
   └── 문제 매핑
   
2. evaluation_orchestrator.py 실행 (NEW)
   ├── DataLoader: 기존 결과 로드
   ├── TopicEvaluator: 주제 평가
   ├── DataEvaluator: 데이터 평가
   ├── AlgorithmEvaluator: 알고리즘 평가
   ├── ProblemSolvingEvaluator: 문제해결 평가
   ├── PlanEvaluator: 계획 평가
   └── ReportGenerator: 리포트 생성
   
3. 출력
   ├── {document_id}_evaluation_report.json
   ├── {document_id}_evaluation_checklist.md
   └── {document_id}_evaluation_feedback.md
```

---

## 📝 프롬프트 설계 전략

### 프롬프트 템플릿 구조 (`config/prompts.yaml` 추가)

```yaml
evaluation:
  # 주제 평가
  topic_specificity:
    system: "당신은 프로젝트 평가 전문가입니다. 프로젝트 주제의 구체성을 평가합니다."
    prompt: |
      다음 프로젝트 목적 텍스트를 분석하고, 주제가 구체적으로 정의되었는지 평가하세요.
      
      [목적 텍스트]
      {purpose_text}
      
      평가 기준:
      1. 구체적인 목표가 명시되어 있는가?
      2. 대상 문제/도메인이 명확한가?
      3. 해결 방법의 개요가 제시되어 있는가?
      
      다음 JSON 형식으로 응답하세요:
      {{
        "result": true or false,
        "confidence": 0.0-1.0,
        "evidence": ["근거1", "근거2"],
        "reasoning": "평가 이유 설명"
      }}
    
  topic_novelty_keywords:
    system: "당신은 연구 키워드 추출 전문가입니다."
    prompt: |
      다음 프로젝트 목적에서 핵심 키워드를 3-5개 추출하세요.
      검색에 사용할 것이므로 영어로 변환하세요.
      
      [목적]
      {purpose_text}
      
      JSON 배열로 응답: ["keyword1", "keyword2", ...]
  
  # 데이터 평가
  data_availability:
    system: "당신은 데이터 과학 프로젝트 평가 전문가입니다."
    prompt: |
      다음 데이터 섹션을 분석하고, 데이터가 확보되었거나 확보 계획이 구체적인지 평가하세요.
      
      [데이터 섹션]
      {data_text}
      
      평가 기준:
      - 데이터셋 이름/출처가 명시되어 있는가?
      - 데이터 수집 방법이 구체적인가?
      
      JSON 형식으로 응답하세요.
  
  # 알고리즘 평가
  algorithm_specificity:
    prompt: |
      다음 단계별 프레임워크를 분석하세요.
      
      [단계 목록]
      {steps_json}
      
      평가 기준:
      1. 단계가 3개 이상인가?
      2. 각 단계의 입력/출력이 명시되어 있는가?
      3. 단계 간 연결이 논리적인가?
      
      JSON 형식으로 응답하세요.
  
  # 문제 해결력 평가
  problem_purposefulness:
    prompt: |
      각 단계에서 해당 기술을 사용한 이유가 명시되었는지 평가하세요.
      
      [문제 매핑]
      {problem_mapping_json}
      
      [원본 텍스트 청크]
      {chunks}
      
      "이유", "목적", "위해", "문제" 등의 표현이 있는지 확인하세요.
```

---

## 🎨 사용자 인터페이스 (CLI)

### main_pipeline.py 통합

```python
# main_pipeline.py에 추가
def run_full_pipeline_with_evaluation(pdf_path: str, output_dir: str = "output"):
    """전체 파이프라인 + 평가 실행"""
    orchestrator = PipelineOrchestrator()
    
    # 1단계: 기존 파이프라인
    problem_mapping = orchestrator.process_document(pdf_path, output_dir)
    
    if problem_mapping:
        # 2단계: 평가 실행
        evaluator = EvaluationOrchestrator(orchestrator.config)
        eval_report = evaluator.evaluate(
            document_id=problem_mapping.document_id,
            output_dir=output_dir
        )
        
        # 3단계: 리포트 저장
        report_gen = ReportGenerator()
        report_gen.save_all_formats(eval_report, output_dir)
        
        logger.info(f"평가 완료! 총점: {eval_report.total_score:.1f}/100")
        logger.info(f"체크리스트: {output_dir}/{eval_report.document_id}_evaluation_checklist.md")
```

---

## 📊 예상 출력 예시

### 체크리스트 마크다운 (`_evaluation_checklist.md`)

```markdown
# 평가 체크리스트 (Evaluation Checklist)

**문서 ID**: 449a92bc-0905-43e8-818d-9382ce195df4
**평가 일시**: 2026-02-01T10:30:00
**총점**: 72.5 / 100

---

## 주제 선정 및 창의성 (2/3 통과)

✅ **구체성**: 프로젝트 주제가 명확하고 구체적으로 정의되어 있는가?
   - **판단**: "Checkpoint AI"라는 명확한 이름과 "프로젝트 피드백 자동화"라는 구체적 목표가 제시됨
   - **근거**: Checkpoint AI: 프로젝트 피드백 자동화를 위한 AI 에이전트 개발...

❌ **창의성**: 기존에 연구된 사례가 적고 새로운 접근인가?
   - **판단**: 검색 결과 유사한 자동 피드백 시스템이 다수 존재 (18건)
   - **근거**: [검색 결과 링크]

✅ **적합성**: 선택한 분야가 프로젝트의 핵심 주제인가?
   - **판단**: NLP와 멀티에이전트가 핵심 기술로 적합함

## 데이터 수집 및 전처리 (0/3 통과)

❌ **데이터 확보**: 분석에 활용할 데이터를 확보하였는가?
   - **판단**: 데이터 섹션이 문서에서 발견되지 않았습니다

...
```

---

## 🚀 구현 우선순위

### Priority 1 (핵심 기능)
1. ✅ DataLoader 구현
2. ✅ 기본 데이터 모델 (ChecklistItem, CategoryEvaluation, EvaluationReport)
3. ✅ TopicEvaluator (구체성, 적합성만 - 검색 제외)
4. ✅ AlgorithmEvaluator
5. ✅ ReportGenerator (JSON, Markdown 체크리스트)

### Priority 2 (검색 통합)
6. ✅ SearchEnhancer 구현
7. ✅ TopicEvaluator에 창의성 검색 추가
8. ✅ ProblemSolvingEvaluator에 타당성 검색 추가

### Priority 3 (완성도)
9. ✅ DataEvaluator, PlanEvaluator
10. ✅ 상세 피드백 리포트
11. ✅ 프롬프트 최적화
12. ✅ 에러 핸들링 및 로깅

---

## 🔍 테스트 전략

### 단위 테스트
- 각 Evaluator별 독립 테스트
- Mock 데이터 사용

### 통합 테스트
- 전체 파이프라인 실행
- 샘플 PDF로 end-to-end 테스트

### 테스트 파일 구조
```
tests/
├── test_evaluation_data_loader.py
├── test_topic_evaluator.py
├── test_data_evaluator.py
├── test_algorithm_evaluator.py
├── test_problem_solving_evaluator.py
├── test_plan_evaluator.py
├── test_search_enhancer.py
└── test_evaluation_orchestrator.py
```

---

## 📌 주의사항 및 고려사항

### 1. 데이터 누락 처리
- `sections.background`, `sections.data`가 null인 경우 처리
- 해당 항목은 자동으로 `result: false`, 적절한 reasoning 제공

### 2. LLM 비용 최적화
- 캐싱 활용 (동일 질문 재사용 방지)
- 배치 처리 가능한 항목은 한 번에 처리

### 3. 검색 API 제한
- Rate limiting 고려
- 실패 시 fallback 로직 (검색 없이 LLM만 사용)

### 4. 확장성
- 새로운 평가 항목 추가 용이하도록 설계
- 평가 기준 변경 시 프롬프트만 수정하면 되도록

### 5. 신뢰도 표시
- 모든 평가 결과에 confidence 점수 포함
- 낮은 confidence의 경우 사용자에게 수동 검토 권장

---

## 📚 참고 자료

### 유사 프로젝트
- Automated Essay Scoring (AES) 시스템
- Code Review Automation Tools
- Research Paper Evaluation Systems

### 사용 기술 스택
- **LLM**: GPT-4, Claude, Gemini
- **검색**: Tavily API, Perplexity API
- **데이터 검증**: Pydantic
- **병렬 처리**: ThreadPoolExecutor
- **로깅**: Python logging + colorlog

---

## 🎯 완료 기준 (Definition of Done)

평가 에이전트 시스템이 완료되었다고 판단하는 기준:

1. ✅ 모든 루브릭 항목(11개)에 대한 Yes/No 평가 가능
2. ✅ 체크리스트 마크다운 파일 생성
3. ✅ 평가 근거(evidence) 명확히 제시
4. ✅ 검색 기반 항목(창의성, 타당성) 작동
5. ✅ 샘플 PDF 8개 모두 평가 성공
6. ✅ 총점 계산 및 리포트 생성
7. ✅ 에러 발생 시 적절한 fallback 처리

---

## 📅 예상 개발 일정

- **Phase 1**: 1-2일 (DataLoader, 기본 모델)
- **Phase 2**: 3-4일 (5개 Evaluator 구현)
- **Phase 3**: 1일 (SearchEnhancer)
- **Phase 4**: 1일 (Orchestrator)
- **Phase 5**: 1일 (ReportGenerator)
- **테스트 및 디버깅**: 2일

**총 예상 기간**: 8-10일

---

이 계획서를 바탕으로 체계적인 구현이 가능합니다. 각 Phase별로 단계적으로 진행하며, 우선순위에 따라 핵심 기능부터 구현할 수 있습니다.
