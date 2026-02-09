# 프로젝트 구조

```
자연어처리 프로젝트 2/
│
├── config/                      # 설정 파일 디렉토리
│   ├── config.yaml              # 시스템 설정
│   └── prompts.yaml             # LLM 프롬프트 템플릿
│
├── src/                         # 소스 코드 디렉토리
│   ├── models.py                # 데이터 모델 (Pydantic)
│   ├── utils.py                 # 유틸리티 (Config, LLM, Cache)
│   ├── pdf_extractor.py         # Agent 1: PDF 추출
│   ├── structure_parser.py      # Agent 2: 구조 파싱
│   ├── step_decomposer.py       # Agent 3: 스텝 분해
│   ├── search_client.py         # 검색 API 클라이언트
│   └── problem_analyzer.py      # Agent 4: 문제 분석
│
├── tests/                       # 테스트 및 검증 디렉토리
│   ├── quick_start.py           # 빠른 테스트 스크립트
│   ├── test_pipeline.py         # 테스트 스위트
│   └── validate_system.py       # 시스템 유효성 검사
│
├── docs/                        # 문서 디렉토리
│   ├── PROJECT_STRUCTURE.md     # 이 파일
│   ├── INSTALLATION.md          # 설치 가이드
│   ├── LLM_PROVIDERS.md         # LLM 설정 가이드
│   └── IMPLEMENTATION_SUMMARY.md # 구현 요약
│
├── output/                      # 출력 디렉토리 (자동 생성)
│   ├── {doc_id}_raw_document.json
│   ├── {doc_id}_structured_document.json
│   ├── {doc_id}_sequential_steps.json
│   ├── {doc_id}_problem_mapping.json
│   └── {doc_id}_report.md
│
├── sample/                      # 샘플 데이터
│   └── 중간보고서_자연어처리.pdf
│
├── main_pipeline.py             # 메인 오케스트레이터
├── requirements.txt             # Python 의존성
├── .gitignore                   # Git 제외 파일
├── .env.example                 # 환경 변수 예시
└── README.md                    # 사용 설명서
```

## 파일 설명

### 설정 파일 (config/)

- **config/config.yaml**: LLM 모델, Redis, 검색 API, 헤더 키워드 등 모든 설정
- **config/prompts.yaml**: 각 에이전트가 사용하는 LLM 프롬프트 템플릿
- **.env**: API 키 및 민감한 환경 변수 (루트 디렉토리에 생성 필요)

### 코어 모듈 (src/)

- **src/models.py**: 모든 데이터 구조 정의 (Pydantic 모델)
  - `RawDocument`: PDF 추출 결과
  - `StructuredDocument`: 5개 섹션 구조
  - `SequentialSteps`: 순차적 스텝 목록
  - `ProblemMapping`: 문제 매핑

- **src/utils.py**: 공통 유틸리티
  - `Config`: YAML 설정 로더
  - `LLMClient`: OpenAI/Anthropic 통합 클라이언트
  - `CacheManager`: Redis 캐시 관리

### 에이전트 모듈 (src/)

1. **src/pdf_extractor.py**: PyMuPDF Component
   - PDF → 텍스트 블록 + 메타데이터
   - 품질 점수 계산
   - 페이지 누락/인코딩 오류 탐지

2. **src/structure_parser.py**: Structure Parser Agent
   - 텍스트 → 5개 섹션 딕셔너리
   - 헤더 키워드 매칭
   - LLM Few-shot 분류

3. **src/step_decomposer.py**: Step Decomposer Agent
   - data + pipeline → 순차적 스텝
   - LLM Chain-of-Thought 분해
   - 의존성 그래프 구축

4. **src/search_client.py**: Search API Client
   - Tavily/DuckDuckGo 통합
   - 폴백 전략
   - Wikipedia 보조 검색

5. **src/problem_analyzer.py**: Problem Analyzer Agent
   - 스텝 → 문제 매핑
   - 검색 API로 기법-문제 관계 추론
   - LLM 기반 카테고리 분류

### 실행 및 테스트 파일

- **main_pipeline.py**: 전체 파이프라인 오케스트레이터 (루트)
  - 4개 에이전트 순차 실행
  - 상태 관리 (Redis)
  - 에러 핸들링
  - 최종 보고서 생성

- **tests/quick_start.py**: 빠른 테스트 스크립트
  - 의존성 없이 PDF 추출만 테스트
  - 설치 가이드 제공

- **tests/test_pipeline.py**: 단위/통합 테스트
  - pytest 기반
  - 각 컴포넌트 독립 테스트

- **tests/validate_system.py**: 시스템 구성 및 API 연결 확인

## 데이터 흐름

```
PDF 파일 (sample/)
  ↓
[src/pdf_extractor.py]
  → RawDocument (텍스트 블록 + 메타데이터)
  ↓
[src/structure_parser.py]
  → StructuredDocument (5개 섹션 딕셔너리)
  ↓
[src/step_decomposer.py]
  → SequentialSteps (순차적 스텝 리스트)
  ↓
[src/problem_analyzer.py]
  ↓ (각 스텝의 기법마다)
  → [src/search_client.py] → 검색 결과
  → [src/utils.py (LLMClient)] → 문제 요약/분류
  ↓
  → ProblemMapping (step_id → problems)
  ↓
[main_pipeline.py]
  → JSON 파일 + Markdown 보고서 (output/)
```

## 주요 의존성 관계

```
main_pipeline.py
  ├── src/pdf_extractor.py
  ├── src/structure_parser.py
  │   ├── src/utils.py (LLMClient)
  │   └── src/models.py
  ├── src/step_decomposer.py
  │   ├── src/utils.py (LLMClient)
  │   └── src/models.py
  └── src/problem_analyzer.py
      ├── src/search_client.py
      ├── src/utils.py (LLMClient, CacheManager)
      └── src/models.py
```