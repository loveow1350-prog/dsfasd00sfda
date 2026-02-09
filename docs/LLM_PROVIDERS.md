# LLM Provider 설정 가이드

이 프로젝트는 다양한 LLM 제공업체를 지원합니다. 유료 및 무료 옵션이 모두 있습니다.

## 지원되는 Provider

### 1. OpenAI (유료)
- **모델**: `gpt-4-turbo-preview`, `gpt-3.5-turbo`
- **API 키 발급**: https://platform.openai.com/api-keys

**설정**:
```yaml
# config/config.yaml
llm:
  provider: "openai"
  model: "gpt-4-turbo-preview"
```

```bash
# .env
OPENAI_API_KEY=sk-...
```

### 2. Anthropic Claude (유료)
- **모델**: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`
- **API 키 발급**: https://console.anthropic.com/

**설정**:
```yaml
# config/config.yaml
llm:
  provider: "anthropic"
  model: "claude-3-sonnet-20240229"
```

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Google Gemini
- **모델**: `gemini-pro`, `gemini-1.5-pro`
- **API 키 발급**: https://makersuite.google.com/app/apikey

**설정**:
```yaml
# config/config.yaml
llm:
  provider: "google"
  model: "gemini-pro"
```

```bash
# .env
GOOGLE_API_KEY=AIza...
```

### 4. Ollama (무료, 로컬)
- **모델**: `llama2`, `mistral`, `mixtral`, `gemma`
- **설치**: https://ollama.ai/download

**설치 및 설정**:
```bash
# 1. Ollama 설치 (Windows)
# https://ollama.ai/download 에서 다운로드

# 2. 모델 다운로드
ollama pull llama2
# 또는
ollama pull mistral

# 3. Ollama 서버 실행 (자동으로 백그라운드 실행됨)
ollama serve
```

```yaml
# config/config.yaml
llm:
  provider: "ollama"
  model: "llama2"  # 또는 mistral, mixtral
  base_url: "http://localhost:11434"
```

### 5. Hugging Face (무료)
- **모델**: `meta-llama/Llama-2-7b-chat-hf`, `mistralai/Mistral-7B-Instruct-v0.2`
- **API 키 발급**: https://huggingface.co/settings/tokens

**설정**:
```yaml
# config/config.yaml
llm:
  provider: "huggingface"
  model: "mistralai/Mistral-7B-Instruct-v0.2"
```

```bash
# .env
HUGGINGFACE_API_KEY=hf_...
```

## 설정 예시

### Ollama 로컬 설정
```yaml
# config/config.yaml
llm:
  provider: "ollama"
  model: "mistral"
  temperature: 0.3
  max_tokens: 4096
  base_url: "http://localhost:11434"
```

### Google Gemini
```yaml
# config/config.yaml
llm:
  provider: "google"
  model: "gemini-pro"
  temperature: 0.3
  max_tokens: 4096
```

```bash
# .env
GOOGLE_API_KEY=AIzaSy...
```

## 문제 해결

### Ollama 포트 오류 (127.0.0.1:11434)
**에러 메시지**: "bind: Only one usage of each socket address is normally permitted"

**원인**: Ollama는 이미 백그라운드에서 실행 중.

**해결**:
```bash
# 정상 작동 확인
ollama list

# ollama serve 실행 X
```

`ollama list`가 정상적으로 모델 목록을 보여주면 문제 없습니다.
