<div align="center">

# 🤖 CheckPoint-AI: Project Report Analyzer

**프로젝트 중간 보고서를 분석하여 파이프라인과 문제 해결 과정을 자동으로 추출하는 AI 시스템**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini-4285F4.svg)](https://ai.google.dev/)
[![OpenAI](https://img.shields.io/badge/LLM-OpenAI-412991.svg)](https://openai.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-000000.svg)](https://ollama.ai/)
[![Redis](https://img.shields.io/badge/Cache-Redis-DC382D.svg)](https://redis.io/)
[![Tavily](https://img.shields.io/badge/Search-Tavily-00A67E.svg)](https://tavily.com/)
[![DuckDuckGo](https://img.shields.io/badge/Search-DuckDuckGo-DE5833.svg)](https://duckduckgo.com/)
[![PyMuPDF](https://img.shields.io/badge/PDF-PyMuPDF-FF6B6B.svg)](https://pymupdf.readthedocs.io/)

</div>

---

## 🚀 주요 기능

- **📄 PDF 자동 파싱**: PyMuPDF를 이용한 고품질 텍스트 및 구조 정보 추출
- **🧠 AI 기반 섹션 분류**: 목적, 배경, 데이터, 파이프라인, 계획 등 5개 핵심 섹션 자동 분류
- **💡 파이프라인 분해**: 복잡한 파이프라인을 순차적 단계로 분해 및 의존성 분석
- **🔍 웹 검색 기반 분석**: Tavily/DuckDuckGo 검색을 통한 기술적 문제 해결책 매핑
- **📝 자동 평가 시스템**: 루브릭 기반 체크리스트로 프로젝트 자동 평가 (NEW!)
- **💾 Redis 캐싱**: 중복 분석 방지 및 성능 최적화를 위한 지능형 캐싱
- **📊 결과물 생성**: 정형화된 JSON 데이터 및 가독성 높은 Markdown 보고서 자동 생성

## 🛠️ 빠른 시작

### 1. 환경 설정

```bash
# 저장소 복제
git clone https://github.com/your-repo/nlp_project_2.git
cd nlp_project_2

# 의존성 설치
pip install -r requirements.txt
```

### 2. LLM Provider 설정

CheckPoint-AI는 다양한 LLM을 지원합니다. `config/config.yaml`에서 설정을 변경할 수 있습니다.

| Provider | Model (추천) | 설정 파일 (`config.yaml`) | API 키 (`.env`) |
| :--- | :--- | :--- | :--- |
| **Google** | `gemini-pro` | `provider: "google"` | `GOOGLE_API_KEY` |
| **OpenAI** | `gpt-4-turbo` | `provider: "openai"` | `OPENAI_API_KEY` |
| **Ollama** | `llava` | `provider: "ollama"` | (필요 없음) |

> 💡 상세 설정 가이드는 [docs/LLM_PROVIDERS.md](docs/LLM_PROVIDERS.md)를 참고하세요.

### 3. Redis 서버 설정 (선택사항)

성능 최적화를 위해 Redis를 사용할 수 있습니다.

- **Windows**: [Microsoft Archive Redis](https://github.com/microsoftarchive/redis/releases) 설치
- **Linux/WSL**: `sudo apt install redis-server` 실행

### 4. 실행

```bash
# 기본 실행 (샘플 파일)
python main_pipeline.py

# 특정 PDF 파일 분석
python main_pipeline.py "path/to/your/report.pdf"

# 분석 + 자동 평가 (루브릭 기반 체크리스트)
python main_pipeline.py "path/to/your/report.pdf" --with-evaluation

# 기존 분석 결과에 대해 평가만 실행
python run_evaluation.py <document_id>
```

### 5. 평가 결과 확인

평가를 실행하면 다음 파일들이 생성됩니다:

- `*_evaluation_report.json`: 전체 평가 결과 (JSON)
- `*_evaluation_checklist.md`: ✅❌ 체크리스트 형식
- `*_evaluation_feedback.md`: 상세 피드백 및 개선 권장사항

## 📂 프로젝트 구조

```text
nlp_project_2/
├── config/           # 시스템 설정 및 프롬프트
├── src/              # 코어 분석 엔진 (에이전트)
├── tests/            # 검증 및 테스트 스크립트
├── docs/             # 상세 가이드 문서
├── output/           # 분석 결과 저장소
└── main_pipeline.py  # 메인 오케스트레이터
```
> 자세한 구조는 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)에서 확인하세요.

## 📊 워크플로우

1. **Extraction**: PDF에서 텍스트 블록 및 메타데이터 추출
2. **Parsing**: LLM을 통한 섹션별 구조화
3. **Decomposition**: 파이프라인 단계 분해 및 의존성 정의
4. **Analysis**: 기술적 문제 정의 및 웹 검색 연동
5. **Evaluation**: 루브릭 기반 자동 평가 (선택사항)
6. **Reporting**: 분석 결과 통합 및 최종 보고서 생성

## 📜 문서 링크

- [설치 가이드 (Installation)](docs/INSTALLATION.md)
- [평가 구현 계획서 (Evaluation Plan)](docs/EVALUATION_AGENT_PLAN.md)
- [LLM 설정 상세 (LLM Providers)](docs/LLM_PROVIDERS.md)
- [Redis 설정 (Redis Setup)](docs/REDIS_SETUP.md)
- [프로젝트 구조 상세 (Project Structure)](docs/PROJECT_STRUCTURE.md)
- [구현 세부사항 (Implementation)](docs/IMPLEMENTATION_SUMMARY.md)
