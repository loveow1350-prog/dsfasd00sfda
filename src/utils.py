"""
Utilities: LLM interface, configuration loader, caching
"""
import os
import yaml
import redis
import json
import hashlib
from typing import Optional, Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
import colorlog, logging

# Load .env file for local development
# Note: In Google Colab, use Secrets instead
load_dotenv()

def setup_logger(name: str = None) -> logging.Logger:
    """Setup colorlog logger"""
    logger = colorlog.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False

    if not logger.handlers:
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    return logger

logger = setup_logger(__name__)

class Config:
    """Configuration manager"""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def get(self, key: str, default=None):
        """Get configuration value by dot notation (e.g., 'llm.provider')"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default


class LLMClient:
    """Unified LLM client supporting OpenAI, Anthropic, Google Gemini, Ollama, and Hugging Face"""

    def __init__(self, config: Config):
        self.config = config
        self.provider = config.get('llm.provider', 'openai')
        self.model = config.get('llm.model', 'gpt-4-turbo-preview')
        self.temperature = config.get('llm.temperature', 0.3)
        self.max_tokens = config.get('llm.max_tokens', 4096)

        if self.provider == 'openai':
            import openai
            self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        elif self.provider == 'anthropic':
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        elif self.provider == 'google':
            import google.generativeai as genai
            # Support both GEMINI_API_KEY (preferred) and GOOGLE_API_KEY (backward compatibility)
            api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model)
        elif self.provider == 'ollama':
            # Ollama runs locally, no API key needed
            self.base_url = config.get('llm.base_url', 'http://localhost:11434')
            self.client = None  # Will use requests
        elif self.provider == 'huggingface':
            from huggingface_hub import InferenceClient
            api_key = os.getenv('HUGGINGFACE_API_KEY')
            self.client = InferenceClient(token=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def supports_vision(self) -> bool:
        """Check if current provider/model supports vision"""
        vision_keywords = [
            'vision', 'gpt-4-turbo', 'gpt-4o', 'claude-3',
            'gemini-1.5', 'gemini-pro-vision',
            'llava', 'bakllava', 'moondream', 'vl', 'gemma'  # Ollama vision models
        ]
        return any(keyword in self.model.lower() for keyword in vision_keywords)

    def generate_with_image(
        self,
        prompt: str,
        image_data: bytes,
        system: Optional[str] = None
    ) -> str:
        """
        Generate text using VLM (Vision Language Model)

        Args:
            prompt: Text prompt
            image_data: Image bytes (PNG/JPEG)
            system: System message (optional)

        Returns:
            Generated text
        """
        try:
            if self.provider == 'openai':
                return self._openai_vision(prompt, image_data, system)
            elif self.provider == 'anthropic':
                return self._anthropic_vision(prompt, image_data, system)
            elif self.provider == 'google':
                return self._google_vision(prompt, image_data, system)
            elif self.provider == 'ollama':
                return self._ollama_vision(prompt, image_data, system)
            else:
                raise ValueError(f"Vision not supported for provider: {self.provider}")

        except Exception as e:
            logger.error(f"Vision generation failed: {e}")
            raise

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Generate text using LLM

        Args:
            prompt: User prompt
            system: System message (optional)

        Returns:
            Generated text
        """
        try:
            if self.provider == 'openai':
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content

            elif self.provider == 'anthropic':
                kwargs = {"model": self.model, "max_tokens": self.max_tokens}
                if system:
                    kwargs["system"] = system

                message = self.client.messages.create(
                    **kwargs,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature
                )
                return message.content[0].text

            elif self.provider == 'google':
                # Combine system and user prompt for Gemini
                full_prompt = prompt
                if system:
                    full_prompt = f"{system}\n\n{prompt}"

                generation_config = {
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }

                response = self.client.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
                return response.text

            elif self.provider == 'ollama':
                import requests

                # Combine system and user prompt
                full_prompt = prompt
                if system:
                    full_prompt = f"{system}\n\n{prompt}"

                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens
                        }
                    },
                    timeout=120
                )
                response.raise_for_status()
                return response.json()["response"]

            elif self.provider == 'huggingface':
                # Combine system and user prompt
                full_prompt = prompt
                if system:
                    full_prompt = f"{system}\n\n{prompt}"

                response = self.client.text_generation(
                    full_prompt,
                    model=self.model,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                return response

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def _openai_vision(self, prompt: str, image_data: bytes, system: Optional[str] = None) -> str:
        """OpenAI GPT-4V"""
        import base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')

        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                }
            ]
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    def _anthropic_vision(self, prompt: str, image_data: bytes, system: Optional[str] = None) -> str:
        """Anthropic Claude Vision"""
        import base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')

        kwargs = {"model": self.model, "max_tokens": self.max_tokens}
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(
            **kwargs,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }],
            temperature=self.temperature
        )
        return response.content[0].text

    def _google_vision(self, prompt: str, image_data: bytes, system: Optional[str] = None) -> str:
        """Google Gemini Vision"""
        from PIL import Image
        import io

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))

        # Combine system and prompt
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"

        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }

        response = self.client.generate_content(
            [full_prompt, image],
            generation_config=generation_config
        )
        return response.text

    def _ollama_vision(self, prompt: str, image_data: bytes, system: Optional[str] = None) -> str:
        """Ollama Vision (llava, bakllava, moondream, etc.)"""
        import requests
        import base64

        # Encode image to base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')

        # Combine system and prompt
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"

        estimated_tokens = len(full_prompt) // 4
        logger.debug(f"Estimated input tokens: {estimated_tokens}")

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "images": [image_b64],  # Ollama vision API
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            },
            timeout=120
        )
        response.raise_for_status()
        logger.debug(response.json()["response"])
        return response.json()["response"]


class LLMRouter:
    """
    Router for dual-model strategy
    Routes tasks to small or big model based on complexity
    """

    def __init__(self, config: Config):
        self.config = config
        self.enable_dual_model = config.get('llm.enable_dual_model', False)

        if self.enable_dual_model:
            # Create two separate LLM clients with potentially different providers
            logger.info("Initializing dual-model strategy")

            # Small model config (can have different provider)
            small_provider = config.get('llm.small_model_provider', config.get('llm.provider', 'ollama'))
            small_config_dict = {
                'llm': {
                    'provider': small_provider,
                    'model': config.get('llm.small_model', 'gemma2:2b'),
                    'temperature': config.get('llm.temperature', 0.7),
                    'max_tokens': config.get('llm.max_tokens', 2048),
                    'base_url': config.get('llm.small_model_base_url', config.get('llm.base_url', 'http://localhost:11434'))
                }
            }

            # Big model config (can have different provider)
            big_provider = config.get('llm.big_model_provider', config.get('llm.provider', 'google'))
            big_config_dict = {
                'llm': {
                    'provider': big_provider,
                    'model': config.get('llm.big_model', config.get('llm.model', 'gemini-pro')),
                    'temperature': config.get('llm.temperature', 0.7),
                    'max_tokens': config.get('llm.max_tokens', 2048),
                    'base_url': config.get('llm.base_url', 'http://localhost:11434')
                }
            }

            # Create temporary config objects
            class TempConfig:
                def __init__(self, config_dict):
                    self.config = config_dict

                def get(self, key, default=None):
                    keys = key.split('.')
                    value = self.config
                    for k in keys:
                        if isinstance(value, dict):
                            value = value.get(k)
                        else:
                            return default
                    return value if value is not None else default

            self.small_llm = LLMClient(TempConfig(small_config_dict))
            self.big_llm = LLMClient(TempConfig(big_config_dict))

            logger.info(f"Small model: {small_provider}/{config.get('llm.small_model', 'gemma2:2b')}")
            logger.info(f"Big model: {big_provider}/{config.get('llm.big_model', 'gemini-pro')}")

            # Evaluation model config (can be different from big/small)
            eval_provider = config.get('llm.eval_model_provider', big_provider)
            eval_model = config.get('llm.eval_model', config.get('llm.big_model', 'gemini-pro'))

            # Only create separate eval client if different from big model
            if eval_provider != big_provider or eval_model != config.get('llm.big_model'):
                eval_config_dict = {
                    'llm': {
                        'provider': eval_provider,
                        'model': eval_model,
                        'temperature': config.get('llm.temperature', 0.7),
                        'max_tokens': config.get('llm.max_tokens', 2048),
                        'base_url': config.get('llm.base_url', 'http://localhost:11434')
                    }
                }
                self.eval_llm = LLMClient(TempConfig(eval_config_dict))
                logger.info(f"Eval model: {eval_provider}/{eval_model}")
            else:
                # Use big model for evaluation
                self.eval_llm = self.big_llm
                logger.info(f"Eval model: using big model")

        else:
            # Fallback to single model
            logger.info("Using single model (dual-model disabled)")
            self.small_llm = LLMClient(config)
            self.big_llm = self.small_llm  # Same client
            self.eval_llm = self.small_llm  # Same client

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        complexity: str = "high"
    ) -> str:
        """
        Generate text using appropriate model based on complexity

        Args:
            prompt: User prompt
            system: System message (optional)
            complexity: "low" for small model, "high" for big model

        Returns:
            Generated text
        """
        if complexity == "low" and self.enable_dual_model:
            logger.debug(f"Using small model for low-complexity task")
            return self.small_llm.generate(prompt, system)
        else:
            logger.debug(f"Using big model for high-complexity task")
            return self.big_llm.generate(prompt, system)

    def generate_with_image(
        self,
        prompt: str,
        image_data: bytes,
        system: Optional[str] = None
    ) -> str:
        """
        Generate text using VLM (always uses big model)

        Args:
            prompt: Text prompt
            image_data: Image bytes (PNG/JPEG)
            system: System message (optional)

        Returns:
            Generated text
        """
        logger.debug("Using big model for VLM task")
        return self.big_llm.generate_with_image(prompt, image_data, system)

    def supports_vision(self) -> bool:
        """Check if big model supports vision"""
        return self.big_llm.supports_vision()

    def chat(
        self,
        messages: List[Dict[str, str]],
        response_format: str = "text",
        complexity: str = "high"
    ) -> str:
        """
        Chat completion with messages

        Args:
            messages: List of message dicts with 'role' and 'content'
            response_format: "text" or "json"
            complexity: "low" for small model, "high" for big model

        Returns:
            Generated text
        """
        # Extract system and user messages
        system_msg = None
        user_msg = ""

        for msg in messages:
            if msg.get('role') == 'system':
                system_msg = msg.get('content', '')
            elif msg.get('role') == 'user':
                user_msg = msg.get('content', '')

        # Add JSON instruction if needed
        if response_format == "json":
            if system_msg:
                system_msg += "\n\nRespond ONLY with valid JSON. No other text."
            else:
                system_msg = "Respond ONLY with valid JSON. No other text."

        # Route to appropriate model
        if complexity == "low" and self.enable_dual_model:
            return self.small_llm.generate(user_msg, system_msg)
        else:
            return self.big_llm.generate(user_msg, system_msg)

    def chat_eval(
        self,
        messages: List[Dict[str, str]],
        response_format: str = "text"
    ) -> str:
        """
        Chat completion for evaluation tasks (uses eval model)

        Args:
            messages: List of message dicts with 'role' and 'content'
            response_format: "text" or "json"

        Returns:
            Generated text
        """
        # Extract system and user messages
        system_msg = None
        user_msg = ""

        for msg in messages:
            if msg.get('role') == 'system':
                system_msg = msg.get('content', '')
            elif msg.get('role') == 'user':
                user_msg = msg.get('content', '')

        # Add JSON instruction if needed
        if response_format == "json":
            if system_msg:
                system_msg += "\n\nRespond ONLY with valid JSON. No other text."
            else:
                system_msg = "Respond ONLY with valid JSON. No other text."

        # Always use eval model
        logger.debug("Using eval model for evaluation task")
        return self.eval_llm.generate(user_msg, system_msg)


class CacheManager:
    """Redis-based cache manager"""

    def __init__(self, config: Config):
        self.redis_client = redis.Redis(
            host=config.get('redis.host', 'localhost'),
            port=config.get('redis.port', 6379),
            db=config.get('redis.db', 0),
            decode_responses=True
        )
        self.ttl_days = config.get('redis.cache_ttl_days', 7)
        self.ttl_seconds = self.ttl_days * 24 * 60 * 60

    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Cache get failed for {key}: {e}")
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cached value with TTL"""
        try:
            ttl = ttl or self.ttl_seconds
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(value, ensure_ascii=False)
            )
        except Exception as e:
            logger.warning(f"Cache set failed for {key}: {e}")

    def get_document_status(self, doc_id: str) -> Optional[Dict]:
        """Get pipeline status for document"""
        return self.get(f"doc:{doc_id}:status")

    def set_document_status(self, doc_id: str, status: Dict):
        """Set pipeline status for document"""
        self.set(f"doc:{doc_id}:status", status, ttl=3600)  # 1 hour TTL

    def get_artifact(self, doc_id: str, agent_name: str) -> Optional[Any]:
        """Get agent output artifact"""
        return self.get(f"doc:{doc_id}:artifact:{agent_name}")

    def set_artifact(self, doc_id: str, agent_name: str, artifact: Any):
        """Set agent output artifact"""
        self.set(f"doc:{doc_id}:artifact:{agent_name}", artifact)

    def get_technique_problems(self, technique: str) -> Optional[List[str]]:
        """Get cached problems for a technique"""
        key = f"technique:{self._hash(technique)}:problems"
        return self.get(key)

    def set_technique_problems(self, technique: str, problems: List[str]):
        """Cache problems for a technique"""
        key = f"technique:{self._hash(technique)}:problems"
        self.set(key, problems)

    def get_chunk_category(self, chunk_hash: str) -> Optional[str]:
        """Get cached category for a text chunk"""
        return self.get(f"chunk:{chunk_hash}:category")

    def set_chunk_category(self, chunk_hash: str, category: str):
        """Cache category for a text chunk"""
        self.set(f"chunk:{chunk_hash}:category", category)

    def _hash(self, text: str) -> str:
        """Create hash for cache key"""
        return hashlib.md5(text.encode()).hexdigest()


def load_prompts(prompts_path: str = "config/prompts.yaml") -> Dict[str, str]:
    """Load prompt templates from YAML file"""
    if not Path(prompts_path).exists():
        logger.warning(f"Prompts file not found: {prompts_path}")
        return {}

    with open(prompts_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# JSON Extraction Helper Functions

def extract_json_array(text: str) -> Optional[List]:
    """
    Extract JSON array from text, handling markdown code blocks

    This function is designed to handle LLM responses that may wrap JSON in markdown code blocks.
    It attempts multiple extraction strategies to maximize success rate.

    Args:
        text: Text that may contain JSON array, possibly wrapped in markdown

    Returns:
        List if JSON array found and parsed, None otherwise

    Examples:
        >>> extract_json_array('[1, 2, 3]')
        [1, 2, 3]
        >>> extract_json_array('```json\\n[1, 2, 3]\\n```')
        [1, 2, 3]
    """
    import re

    if not text or not text.strip():
        logger.warning("Empty text provided to extract_json_array")
        return None

    original_text = text
    logger.debug(f"Attempting to extract JSON array from text (length: {len(text)})")

    # 0. First, completely strip markdown code blocks
    # This is the most reliable way to handle Gemini's markdown output
    text = re.sub(r'```(?:json)?\s*', '', text)  # Remove opening ```json or ```
    text = re.sub(r'```\s*$', '', text)  # Remove closing ```
    text = text.strip()

    # 1. Remove any explanation text before code blocks or arrays
    text = re.sub(r'^[^`\[]*(?=```|\[)', '', text, flags=re.MULTILINE)

    # 2. Try multiple extraction patterns
    extraction_strategies = [
        # Strategy 1: Markdown code block with json tag (greedy)
        (r'```json\s*(\[[\s\S]*\])\s*```', "markdown with 'json' tag"),
        # Strategy 2: Markdown code block without tag (greedy)
        (r'```\s*(\[[\s\S]*\])\s*```', "markdown without tag"),
        # Strategy 3: Raw JSON array (greedy)
        (r'\[[\s\S]*\]', "raw JSON (greedy)"),
        # Strategy 4: Non-greedy fallback
        (r'\[[\s\S]*?\]', "raw JSON (non-greedy)"),
    ]

    json_str = None
    strategy_used = None

    for pattern, strategy_name in extraction_strategies:
        match = re.search(pattern, text)
        if match:
            if len(match.groups()) > 0:
                json_str = match.group(1)
            else:
                json_str = match.group(0)
            strategy_used = strategy_name
            logger.debug(f"✓ Extracted using strategy: {strategy_name}")
            break

    if not json_str:
        logger.warning(f"No JSON array pattern found in text")
        logger.debug(f"Text preview: {original_text[:500]}")
        return None

    # 3. Clean up the extracted string
    json_str = json_str.strip()

    # Check if JSON is truncated (missing closing ])
    if json_str.count('[') > json_str.count(']'):
        logger.warning("JSON array appears truncated (missing closing ])")
        # Try to fix by adding closing brackets
        missing_brackets = json_str.count('[') - json_str.count(']')

        # Remove incomplete last object if present
        last_open_brace = json_str.rfind('{')
        last_close_brace = json_str.rfind('}')

        if last_open_brace > last_close_brace:
            # Incomplete object at end, remove it
            json_str = json_str[:last_open_brace].rstrip(',').strip()
            logger.info(f"Removed incomplete object at end")

        # Add missing closing brackets
        json_str = json_str + (']' * missing_brackets)
        logger.info(f"Added {missing_brackets} closing bracket(s) to truncated JSON")

    # Remove any trailing text after the last ]
    last_bracket = json_str.rfind(']')
    if last_bracket != -1 and last_bracket < len(json_str) - 1:
        removed = json_str[last_bracket + 1:]
        json_str = json_str[:last_bracket + 1]
        logger.debug(f"Removed trailing text: {removed[:50]}")

    logger.debug(f"Attempting to parse JSON (length: {len(json_str)})")

    # 4. Try to parse
    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            logger.info(f"✅ Successfully parsed JSON array with {len(result)} items using {strategy_used}")
            return result
        else:
            logger.warning(f"Extracted JSON is not an array: {type(result)}")
            return None
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing failed: {e}")
        logger.debug(f"Failed JSON preview: {json_str}")

        # 5. Try to fix common issues
        fixes = [
            (r',\s*]', ']', "trailing comma before ]"),
            (r',\s*}', '}', "trailing comma before }"),
            (r'}\s*{', '},{', "missing comma between objects"),
            (r']\s*\[', '],[', "missing comma between arrays"),
        ]

        for pattern, replacement, fix_name in fixes:
            try:
                json_str_fixed = re.sub(pattern, replacement, json_str)
                if json_str_fixed != json_str:
                    logger.debug(f"Trying fix: {fix_name}")
                    result = json.loads(json_str_fixed)
                    if isinstance(result, list):
                        logger.info(f"✅ JSON parsed after fixing: {fix_name}")
                        return result
            except:
                continue

        # 6. Last resort: try to extract individual objects
        logger.debug("Attempting to extract individual JSON objects as last resort")
        try:
            # Find all {...} objects
            objects = re.findall(r'\{[^{}]*\}', json_str)
            if objects:
                parsed_objects = []
                for obj_str in objects:
                    try:
                        parsed_objects.append(json.loads(obj_str))
                    except:
                        continue
                if parsed_objects:
                    logger.info(f"✅ Extracted {len(parsed_objects)} individual objects")
                    return parsed_objects
        except:
            pass

        logger.error(f"All JSON extraction strategies failed")
        return None


def extract_json_object(text: str) -> Optional[Dict]:
    """
    Extract JSON object from text, handling markdown code blocks

    This function is designed to handle LLM responses that may wrap JSON in markdown code blocks.
    It attempts multiple extraction strategies to maximize success rate.

    Args:
        text: Text that may contain JSON object, possibly wrapped in markdown

    Returns:
        Dict if JSON object found and parsed, None otherwise

    Examples:
        >>> extract_json_object('{"key": "value"}')
        {'key': 'value'}
        >>> extract_json_object('```json\\n{"key": "value"}\\n```')
        {'key': 'value'}
    """
    import re

    if not text or not text.strip():
        logger.warning("Empty text provided to extract_json_object")
        return None

    # 0. First, strip markdown code blocks completely
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = text.strip()

    # 1. Try to extract from markdown code blocks first (just in case)
    json_pattern = r'```(?:json)?\s*(\{[\s\S]*\})\s*```'
    match = re.search(json_pattern, text)

    if match:
        json_str = match.group(1)
        logger.debug("Extracted JSON object from markdown block")
    else:
        # 2. Try to find raw JSON object (greedy)
        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            logger.debug(f"No JSON object found in: {text[:200]}")
            return None
        json_str = match.group(0)
        logger.debug("Extracted JSON object without markdown")

    json_str = json_str.strip()

    # Check if this is actually an array with a single object
    if json_str.startswith('[') and '{' in json_str:
        # Try to extract first object from array
        try:
            # Find first complete object
            brace_count = 0
            start_idx = json_str.find('{')
            if start_idx != -1:
                for i in range(start_idx, len(json_str)):
                    if json_str[i] == '{':
                        brace_count += 1
                    elif json_str[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found complete object
                            json_str = json_str[start_idx:i+1]
                            logger.debug("Extracted first object from array")
                            break
        except:
            pass

    # Check if JSON is truncated (missing closing })
    if json_str.count('{') > json_str.count('}'):
        logger.warning("JSON object appears truncated (missing closing })")
        # Find last complete key-value pair
        last_comma = json_str.rfind(',')
        last_quote = json_str.rfind('"')

        # If there's an incomplete key or value, remove it
        if last_comma > 0:
            # Truncate at last comma and add closing brace
            json_str = json_str[:last_comma].strip() + '}'
            logger.info("Removed incomplete key-value pair from truncated JSON")
        else:
            # Just add closing brace
            json_str = json_str + '}'
            logger.info("Added closing brace to truncated JSON")

    # Remove trailing text after last }
    last_brace = json_str.rfind('}')
    if last_brace != -1 and last_brace < len(json_str) - 1:
        json_str = json_str[:last_brace + 1]

    try:
        result = json.loads(json_str)
        if isinstance(result, dict):
            logger.debug("Successfully parsed JSON object")
            return result
        else:
            logger.warning(f"Extracted JSON is not an object: {type(result)}")
            return None
    except json.JSONDecodeError as e:
        logger.warning(f"JSON object parsing failed: {e}")
        logger.debug(f"Failed JSON preview: {json_str}")

        # Try to fix trailing commas
        try:
            json_str_fixed = re.sub(r',\s*}', '}', json_str)
            result = json.loads(json_str_fixed)
            if isinstance(result, dict):
                logger.info("JSON parsed after fixing trailing comma")
                return result
        except:
            pass

        return None

