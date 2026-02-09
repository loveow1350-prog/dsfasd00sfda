# Evaluation 전용 모델 설정 완료

## ✅ 구현 완료

### 추가된 기능

**LLMRouter에 evaluation 전용 모델 지원**
- 평가 작업에 별도의 모델 사용 가능
- 큰 모델/작은 모델과 독립적으로 설정 가능

---

## 🔧 변경 사항

### 1. config.yaml에 eval_model 설정 추가

```yaml
llm:
  # Small model (Ollama)
  small_model_provider: "ollama"
  small_model: "gemma3:4b"
  
  # Big model (Google Gemini)
  big_model_provider: "google"
  big_model: "gemma-3-27b-it"
  
  # Evaluation model (NEW!)
  eval_model_provider: "google"
  eval_model: "gemini-2.5-flash"
```

**특징**:
- `eval_model`은 big_model과 다른 모델 사용 가능
- provider도 독립적으로 설정 가능
- 미설정 시 big_model 사용 (fallback)

---

### 2. LLMRouter에 eval_llm 추가

```python
class LLMRouter:
    def __init__(self, config):
        self.small_llm = LLMClient(...)  # 작은 모델
        self.big_llm = LLMClient(...)    # 큰 모델
        self.eval_llm = LLMClient(...)   # 평가 모델 (NEW!)
```

**로직**:
- eval_model이 big_model과 다르면 → 별도 LLMClient 생성
- eval_model == big_model이면 → big_llm 재사용
- dual_model 비활성화 시 → 단일 모델 사용

---

### 3. chat_eval() 메서드 추가

```python
def chat_eval(
    self,
    messages: List[Dict[str, str]],
    response_format: str = "text"
) -> str:
    """Evaluation 전용 chat (항상 eval_llm 사용)"""
    logger.debug("Using eval model for evaluation task")
    return self.eval_llm.generate(user_msg, system_msg)
```

**특징**:
- `complexity` 파라미터 없음 (항상 eval_llm 사용)
- JSON 모드 자동 처리
- 평가 전용 로깅

---

### 4. 모든 Evaluator에서 chat_eval 사용

**수정된 파일**:
- ✅ `algorithm_evaluator.py`
- ✅ `topic_evaluator.py`
- ✅ `data_evaluator.py`
- ✅ `problem_solving_evaluator.py`
- ✅ `plan_evaluator.py`

**Before**:
```python
response = self.llm.chat(
    messages=[...],
    response_format="json",
    complexity="high"  # big_llm 사용
)
```

**After**:
```python
response = self.llm.chat_eval(
    messages=[...],
    response_format="json"  # eval_llm 사용
)
```

---

## 📊 모델 사용 분배

### 현재 설정 예시

```yaml
# config/config.yaml
small_model: "gemma3:4b"           # Ollama (로컬)
big_model: "gemma-3-27b-it"        # Gemini 27B
eval_model: "gemini-2.5-flash"     # Gemini 2.5 Flash
```

### 작업별 모델 사용

| 작업 유형 | 모델 | 사용처 |
|---------|------|--------|
| **키워드 추출** | gemma3:4b | search_client, problem_analyzer |
| **간단한 분류** | gemma3:4b | structure_parser |
| **문서 구조 파싱** | gemma-3-27b-it | structure_parser |
| **문제 분석** | gemma-3-27b-it | problem_analyzer |
| **VLM 작업** | gemma-3-27b-it | pdf_extractor |
| **프로젝트 평가** ⭐ | gemini-2.5-flash | evaluators (NEW!) |

---

## 💡 사용 시나리오

### 시나리오 1: 평가만 다른 모델 사용

```yaml
big_model: "gemma-3-27b-it"      # 메인 작업: 강력하지만 느림
eval_model: "gemini-2.5-flash"   # 평가: 빠르고 효율적
```

**장점**:
- 메인 파이프라인: 정확도 우선
- 평가: 속도 우선

### 시나리오 2: 모두 동일 모델

```yaml
big_model: "gemini-2.5-flash"
eval_model: "gemini-2.5-flash"   # 또는 미설정
```

**장점**:
- 일관된 품질
- 설정 단순

### 시나리오 3: 평가에 더 강력한 모델

```yaml
big_model: "gemini-2.5-flash"
eval_model: "gemini-1.5-pro"     # 평가에 더 강력한 모델
```

**장점**:
- 평가 정확도 극대화

---

## 🚀 실행 로그

평가 실행 시 다음 로그가 표시됩니다:

```
INFO - Initializing dual-model strategy
INFO - Small model: ollama/gemma3:4b
INFO - Big model: google/gemma-3-27b-it
INFO - Eval model: google/gemini-2.5-flash  ← NEW!

DEBUG - Using eval model for evaluation task  ← 평가 시
```

---

## 🎯 핵심 이점

### 1. 유연성
- 평가에 최적화된 모델 선택 가능
- provider도 독립적으로 설정

### 2. 비용 최적화
- 메인: 강력한 모델 (정확도)
- 평가: 빠른 모델 (효율성)

### 3. 성능 최적화
- 평가는 빠른 모델로 신속 처리
- 메인 작업은 느려도 정확하게

### 4. 하위 호환성
- eval_model 미설정 시 big_model 사용
- 기존 설정 그대로 작동

---

## 📝 설정 예시

### 예시 1: Gemini Flash (평가 전용)

```yaml
eval_model_provider: "google"
eval_model: "gemini-2.5-flash"
```

### 예시 2: GPT-3.5 (비용 절감)

```yaml
big_model_provider: "google"
big_model: "gemini-1.5-pro"

eval_model_provider: "openai"
eval_model: "gpt-3.5-turbo"
```

### 예시 3: Ollama (완전 무료)

```yaml
eval_model_provider: "ollama"
eval_model: "gemma2:9b"
eval_model_base_url: "http://localhost:11434"
```

---

## ✨ 요약

**3개의 독립적인 모델 설정**:
1. `small_model` - 간단한 작업 (키워드, 분류)
2. `big_model` - 복잡한 작업 (파싱, 분석, VLM)
3. `eval_model` - 평가 전용 (NEW!) ⭐

**모든 evaluator가 `chat_eval()` 사용**:
- ✅ 일관된 평가 모델 사용
- ✅ 독립적인 설정 가능
- ✅ 하위 호환성 유지

---

**구현 완료 일시**: 2026-02-01 15:15  
**영향 범위**: evaluation 시스템 전체  
**하위 호환성**: 유지 (eval_model 미설정 시 big_model 사용)
