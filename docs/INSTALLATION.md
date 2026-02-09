# 설치 및 실행 가이드

## 빠른 시작 (Quick Start)

### 1단계: 기본 테스트

의존성 설치 전에 PDF가 정상적으로 열리는지 테스트:

```bash
# PyMuPDF만 설치
pip install PyMuPDF

# 빠른 테스트 실행
python quick_start.py
```

예상 출력:
```
📄 PDF 파일 열기: 중간보고서_자연어처리.pdf
✅ 총 페이지 수: X 페이지
📊 메타데이터: ...
📝 첫 페이지 텍스트 미리보기 (200자):
...
✅ PDF 추출 테스트 성공!
```

---

## 전체 설치 (Full Installation)

### 2단계: 모든 의존성 설치

```bash
pip install -r requirements.txt
```

설치되는 주요 패키지:
- PyMuPDF (PDF 처리)
- openai / anthropic (LLM)
- redis (캐싱)
- duckduckgo-search / tavily-python (검색)
- pydantic (데이터 검증)

**주의**: Redis 설치는 선택사항입니다. 없어도 작동하지만 캐싱 기능이 비활성화됩니다.

### 3단계: 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env
```

`.env` 파일 편집 (필수):

```env
# LLM API 키 (둘 중 하나 필수)
OPENAI_API_KEY=sk-...
# 또는
ANTHROPIC_API_KEY=sk-ant-...

# 검색 API 키 (선택사항, 없으면 DuckDuckGo 사용)
TAVILY_API_KEY=tvly-...

# Redis (선택사항, 없으면 캐싱 비활성화)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 4단계: Redis 설치 (선택사항)

#### Windows (WSL 사용):
```bash
wsl
sudo apt update
sudo apt install redis-server
redis-server
```

#### Docker 사용:
```bash
docker run -d -p 6379:6379 redis
```

#### Redis 없이 사용:
Redis가 없어도 프로그램은 정상 작동합니다. 다만 동일한 기법을 다시 검색할 때 캐시를 사용하지 못해 속도가 느려집니다.

---

## 실행 방법

### 기본 실행

```bash
python main_pipeline.py 중간보고서_자연어처리.pdf
```

### 실행 과정

프로그램이 다음 단계를 순차적으로 실행합니다:

```
Stage 1/4: PDF Extraction
  → 텍스트 블록 추출, 메타데이터 분석, 품질 점수 계산

Stage 2/4: Structure Parsing
  → 5개 헤더 탐지 및 분류 (purpose, background, data, pipeline, plan)
  → LLM을 사용해 애매한 헤더 분류

Stage 3/4: Step Decomposition
  → data + pipeline 섹션을 순차적 스텝으로 분해
  → LLM Chain-of-Thought로 입력/출력/기법/의존성 추출

Stage 4/4: Problem Analysis
  → 각 스텝의 기법마다:
    - 검색 API로 "기법명 + solves what problem" 검색
    - 검색 결과를 LLM으로 요약
    - 문제 카테고리 및 심각도 분류
```

### 예상 실행 시간

- **캐시 없음 (첫 실행)**: 5-7분 (100페이지 기준)
- **캐시 50% 적중**: 3-4분
- **대부분 캐시 적중**: 2-3분

### 출력 파일

`output/` 디렉토리에 생성됩니다:

```
output/
├── abc123_raw_document.json           # PDF 추출 결과
├── abc123_structured_document.json    # 5개 섹션
├── abc123_sequential_steps.json       # 순차적 스텝
├── abc123_problem_mapping.json        # 문제 매핑
└── abc123_report.md                   # 최종 보고서 (읽기 쉬운 형식)
```

**추천**: `abc123_report.md` 파일을 먼저 확인하세요!

---

## 문제 해결 (Troubleshooting)

### 문제 1: ModuleNotFoundError

```
ModuleNotFoundError: No module named 'fitz'
```

**해결**:
```bash
pip install PyMuPDF
```

### 문제 2: OpenAI API 오류

```
openai.AuthenticationError: Incorrect API key
```

**해결**:
1. `.env` 파일에 올바른 API 키 입력
2. API 키 앞뒤 공백 제거
3. 따옴표 없이 입력: `OPENAI_API_KEY=sk-...`

### 문제 3: Redis 연결 실패

```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**해결**:
- Redis가 없어도 괜찮습니다. 경고만 출력되고 계속 진행됩니다.
- Redis를 사용하려면:
  ```bash
  # WSL에서
  redis-server
  
  # 또는 Docker에서
  docker run -d -p 6379:6379 redis
  ```

### 문제 4: 검색 API 실패

```
Search API failed: ...
```

**해결**:
- Tavily API 키가 없으면 자동으로 DuckDuckGo로 폴백됩니다.
- DuckDuckGo도 실패하면 LLM만으로 문제를 추론합니다.
- 정확도는 조금 떨어지지만 작동합니다.

### 문제 5: PDF 품질 경고

```
Quality score 0.45 below threshold 0.6
```

**해결**:
- PDF가 스캔 이미지이거나 깨져있을 수 있습니다.
- OCR 처리된 PDF를 사용하세요.
- 또는 `config.yaml`에서 `pdf_quality_threshold`를 낮추세요:
  ```yaml
  processing:
    pdf_quality_threshold: 0.4
  ```

---

## 고급 설정

### LLM 모델 변경

`config.yaml` 편집:

```yaml
llm:
  provider: "openai"  # 또는 "anthropic"
  model: "gpt-4-turbo-preview"  # 또는 "gpt-3.5-turbo" (저렴)
  temperature: 0.3  # 낮을수록 일관성 있음
```

### 검색 API 우선순위 변경

```yaml
search:
  primary_api: "duckduckgo"  # Tavily 대신 DuckDuckGo 먼저 시도
  fallback_api: "tavily"
  max_results: 5  # 더 많은 검색 결과
```

### 헤더 키워드 추가

영문 보고서를 처리하려면:

```yaml
headers:
  purpose:
    keywords: ["분석 목적", "Objective", "Purpose", "Goal"]
  data:
    keywords: ["사용 데이터", "Dataset", "Data Collection"]
```

---

## 개별 컴포넌트 테스트

각 에이전트를 독립적으로 테스트할 수 있습니다:

### PDF 추출만 테스트
```bash
python pdf_extractor.py
```

### 구조 파싱 테스트
```bash
python structure_parser.py
```

### 스텝 분해 테스트
```bash
python step_decomposer.py
```

### 검색 API 테스트
```bash
python search_client.py
```

### 문제 분석 테스트
```bash
python problem_analyzer.py
```

---

## 테스트 스위트 실행

```bash
# 모든 테스트 실행
pytest test_pipeline.py -v

# 특정 테스트만 실행
pytest test_pipeline.py::TestPDFExtractor -v
```

**주의**: 일부 테스트는 API 키가 필요하며, 실제 API 호출을 합니다.

---