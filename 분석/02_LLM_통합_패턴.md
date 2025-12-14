# LLM 통합 패턴

Cline 프로젝트에서 40개 이상의 LLM 프로바이더를 통합하는 핵심 패턴을 분석합니다.

---

## 1. Provider Factory Pattern

### 개요
단일 `ApiHandler` 인터페이스로 모든 LLM 프로바이더를 추상화합니다. 팩토리 함수를 통해 런타임에 적절한 프로바이더를 동적으로 선택합니다.

### 구현 구조

```
src/core/api/
├── index.ts                  # 팩토리 함수 및 인터페이스
├── retry.ts                  # 재시도 데코레이터
├── transform/
│   └── stream.ts             # 스트리밍 타입 정의
└── providers/                # 40+ 프로바이더 구현
    ├── anthropic.ts
    ├── openai.ts
    ├── gemini.ts
    ├── ollama.ts
    ├── bedrock.ts
    └── ...
```

### 핵심 인터페이스

```typescript
// src/core/api/index.ts

// 모든 프로바이더가 구현하는 핵심 인터페이스
export interface ApiHandler {
    // 메시지 생성 - 스트리밍 응답 반환
    createMessage(
        systemPrompt: string,
        messages: ClineStorageMessage[],
        tools?: ClineTool[],
        useResponseApi?: boolean
    ): ApiStream

    // 현재 모델 정보 반환
    getModel(): ApiHandlerModel

    // 사용량 통계 (선택적)
    getApiStreamUsage?(): Promise<ApiStreamUsageChunk | undefined>

    // 요청 취소 (선택적)
    abort?(): void
}

export interface ApiHandlerModel {
    id: string
    info: ModelInfo
}
```

### 팩토리 함수

```typescript
// 프로바이더별 핸들러 생성
function createHandlerForProvider(
    apiProvider: string | undefined,
    options: Omit<ApiConfiguration, "apiProvider">,
    mode: Mode,  // "plan" | "act"
): ApiHandler {
    switch (apiProvider) {
        case "anthropic":
            return new AnthropicHandler({
                onRetryAttempt: options.onRetryAttempt,
                apiKey: options.apiKey,
                anthropicBaseUrl: options.anthropicBaseUrl,
                // 모드별 모델/설정 분리
                apiModelId: mode === "plan"
                    ? options.planModeApiModelId
                    : options.actModeApiModelId,
                thinkingBudgetTokens: mode === "plan"
                    ? options.planModeThinkingBudgetTokens
                    : options.actModeThinkingBudgetTokens,
            })
        case "openai":
            return new OpenAiHandler({ /* ... */ })
        case "gemini":
            return new GeminiHandler({ /* ... */ })
        case "ollama":
            return new OllamaHandler({ /* ... */ })
        // 40+ 프로바이더 지원
        default:
            return new AnthropicHandler({ /* 기본값 */ })
    }
}

// 외부 호출용 빌더 함수
export function buildApiHandler(
    configuration: ApiConfiguration,
    mode: Mode
): ApiHandler {
    const { planModeApiProvider, actModeApiProvider, ...options } = configuration
    const apiProvider = mode === "plan" ? planModeApiProvider : actModeApiProvider

    // Thinking Budget 검증 및 클리핑
    // ...

    return createHandlerForProvider(apiProvider, options, mode)
}
```

### hdsp-agent 적용

```python
# backend/llm/handler.py
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
from dataclasses import dataclass

@dataclass
class ModelInfo:
    id: str
    max_tokens: int
    supports_streaming: bool = True

@dataclass
class StreamChunk:
    type: str  # "text" | "reasoning" | "usage" | "tool_calls"
    content: dict

class LLMHandler(ABC):
    """모든 LLM 프로바이더의 기본 인터페이스"""

    @abstractmethod
    async def create_message(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """스트리밍 메시지 생성"""
        pass

    @abstractmethod
    def get_model(self) -> ModelInfo:
        """모델 정보 반환"""
        pass

    async def abort(self) -> None:
        """요청 취소 (선택적)"""
        pass


# backend/llm/providers/gemini.py
class GeminiHandler(LLMHandler):
    def __init__(self, api_key: str, model_id: str = "gemini-pro"):
        self.api_key = api_key
        self.model_id = model_id
        self.client = None

    def _ensure_client(self):
        if not self.client:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_id)
        return self.client

    async def create_message(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        client = self._ensure_client()

        async for chunk in client.generate_content_stream(
            contents=self._format_messages(messages),
            generation_config={"system_instruction": system_prompt}
        ):
            yield StreamChunk(type="text", content={"text": chunk.text})

    def get_model(self) -> ModelInfo:
        return ModelInfo(id=self.model_id, max_tokens=8192)


# backend/llm/factory.py
def create_handler(
    provider: str,
    config: dict,
    mode: str = "act"  # "plan" | "act"
) -> LLMHandler:
    """팩토리 함수"""
    handlers = {
        "gemini": GeminiHandler,
        "openai": OpenAIHandler,
        "anthropic": AnthropicHandler,
        "vllm": VLLMHandler,
    }

    handler_class = handlers.get(provider)
    if not handler_class:
        raise ValueError(f"Unknown provider: {provider}")

    # 모드별 설정 선택
    model_id = config.get(f"{mode}_mode_model_id", config.get("model_id"))

    return handler_class(
        api_key=config.get("api_key"),
        model_id=model_id,
        **config.get("options", {})
    )
```

---

## 2. Normalized Streaming

### 개요
다양한 LLM API의 스트리밍 응답을 단일 타입으로 정규화합니다. `ApiStream` (AsyncGenerator)과 4가지 청크 타입을 사용합니다.

### 스트림 타입 정의

```typescript
// src/core/api/transform/stream.ts

// 스트림의 기본 타입 - AsyncGenerator
export type ApiStream = AsyncGenerator<ApiStreamChunk> & { id?: string }

// 4가지 청크 유형의 합집합
export type ApiStreamChunk =
    | ApiStreamTextChunk
    | ApiStreamThinkingChunk
    | ApiStreamUsageChunk
    | ApiStreamToolCallsChunk

// 1. 텍스트 청크 - 일반 응답
export interface ApiStreamTextChunk {
    type: "text"
    text: string
    id?: string
    signature?: string  // Gemini 서명
}

// 2. 사용량 청크 - 토큰 통계
export interface ApiStreamUsageChunk {
    type: "usage"
    inputTokens: number
    outputTokens: number
    cacheWriteTokens?: number
    cacheReadTokens?: number
    thoughtsTokenCount?: number  // OpenRouter
    totalCost?: number           // OpenRouter
    id?: string
}

// 3. 도구 호출 청크
export interface ApiStreamToolCallsChunk {
    type: "tool_calls"
    tool_call: ApiStreamToolCall
    id?: string
    signature?: string
}

export interface ApiStreamToolCall {
    call_id?: string
    function: {
        id?: string
        name?: string
        arguments?: any
    }
}

// 4. 추론/사고 청크 (Extended Thinking)
export interface ApiStreamThinkingChunk {
    type: "reasoning"
    reasoning: string           // "[REDACTED]" 가능
    details?: unknown           // OpenRouter 속성
    signature?: string          // API 재전송용
    redacted_data?: string
    id?: string
}
```

### 프로바이더별 변환 예시

```typescript
// src/core/api/providers/anthropic.ts

@withRetry()
async *createMessage(
    systemPrompt: string,
    messages: ClineStorageMessage[],
    tools?: AnthropicTool[]
): ApiStream {
    const client = this.ensureClient()
    const stream = await client.messages.create({ /* ... */ })

    // Anthropic 스트림 → 정규화된 ApiStream 변환
    for await (const event of stream) {
        if (event.type === "content_block_delta") {
            if (event.delta.type === "text_delta") {
                yield {
                    type: "text",
                    text: event.delta.text,
                } as ApiStreamTextChunk
            } else if (event.delta.type === "thinking_delta") {
                yield {
                    type: "reasoning",
                    reasoning: event.delta.thinking,
                } as ApiStreamThinkingChunk
            }
        } else if (event.type === "message_delta") {
            yield {
                type: "usage",
                inputTokens: event.usage.input_tokens,
                outputTokens: event.usage.output_tokens,
            } as ApiStreamUsageChunk
        }
    }
}
```

### hdsp-agent 적용

```python
# backend/llm/stream.py
from dataclasses import dataclass
from typing import AsyncGenerator, Any, Literal, Union

@dataclass
class TextChunk:
    type: Literal["text"] = "text"
    text: str = ""
    id: str | None = None

@dataclass
class UsageChunk:
    type: Literal["usage"] = "usage"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_cost: float | None = None

@dataclass
class ToolCallChunk:
    type: Literal["tool_calls"] = "tool_calls"
    call_id: str = ""
    function_name: str = ""
    arguments: dict = None

@dataclass
class ReasoningChunk:
    type: Literal["reasoning"] = "reasoning"
    reasoning: str = ""
    signature: str | None = None

# 합집합 타입
StreamChunk = Union[TextChunk, UsageChunk, ToolCallChunk, ReasoningChunk]
ApiStream = AsyncGenerator[StreamChunk, None]


# backend/llm/providers/gemini.py
async def create_message(self, ...) -> ApiStream:
    """Gemini 응답을 정규화된 스트림으로 변환"""
    async for chunk in self._raw_stream():
        # Gemini 형식 → 정규화된 TextChunk
        if chunk.text:
            yield TextChunk(text=chunk.text)

        # 사용량 정보
        if hasattr(chunk, 'usage_metadata'):
            yield UsageChunk(
                input_tokens=chunk.usage_metadata.prompt_token_count,
                output_tokens=chunk.usage_metadata.candidates_token_count
            )
```

---

## 3. Retry Mechanism

### 개요
지수 백오프와 Rate Limit 헤더 파싱을 지원하는 재시도 메커니즘입니다. 데코레이터 패턴으로 구현되어 모든 프로바이더에 일관되게 적용됩니다.

### 핵심 구현

```typescript
// src/core/api/retry.ts

interface RetryOptions {
    maxRetries?: number      // 기본값: 3
    baseDelay?: number       // 기본값: 1000ms
    maxDelay?: number        // 기본값: 10000ms
    retryAllErrors?: boolean // 기본값: false (429만 재시도)
}

const DEFAULT_OPTIONS: Required<RetryOptions> = {
    maxRetries: 3,
    baseDelay: 1_000,
    maxDelay: 10_000,
    retryAllErrors: false,
}

// 재시도 가능한 에러 클래스
export class RetriableError extends Error {
    status: number = 429
    retryAfter?: number

    constructor(message: string, retryAfter?: number, options?: ErrorOptions) {
        super(message, options)
        this.name = "RetriableError"
        this.retryAfter = retryAfter
    }
}

// 데코레이터 팩토리
export function withRetry(options: RetryOptions = {}) {
    const { maxRetries, baseDelay, maxDelay, retryAllErrors } = {
        ...DEFAULT_OPTIONS,
        ...options
    }

    return (_target: any, _propertyKey: string, descriptor: PropertyDescriptor) => {
        const originalMethod = descriptor.value

        // AsyncGenerator를 래핑하는 새 메서드
        descriptor.value = async function* (...args: any[]) {
            for (let attempt = 0; attempt < maxRetries; attempt++) {
                try {
                    yield* originalMethod.apply(this, args)
                    return  // 성공 시 종료
                } catch (error: any) {
                    const isRateLimit = error?.status === 429 ||
                                       error instanceof RetriableError
                    const isLastAttempt = attempt === maxRetries - 1

                    // 재시도 불가 조건
                    if ((!isRateLimit && !retryAllErrors) || isLastAttempt) {
                        throw error
                    }

                    // Rate Limit 헤더에서 대기 시간 파싱
                    const retryAfter =
                        error.headers?.["retry-after"] ||
                        error.headers?.["x-ratelimit-reset"] ||
                        error.headers?.["ratelimit-reset"] ||
                        error.retryAfter

                    let delay: number
                    if (retryAfter) {
                        const retryValue = parseInt(retryAfter, 10)
                        if (retryValue > Date.now() / 1000) {
                            // Unix timestamp
                            delay = retryValue * 1000 - Date.now()
                        } else {
                            // Delta seconds
                            delay = retryValue * 1000
                        }
                    } else {
                        // 지수 백오프: 1s, 2s, 4s, ... (최대 10s)
                        delay = Math.min(maxDelay, baseDelay * 2 ** attempt)
                    }

                    // 재시도 콜백 호출 (UI 업데이트 등)
                    const handlerInstance = this as any
                    if (handlerInstance.options?.onRetryAttempt) {
                        await handlerInstance.options.onRetryAttempt(
                            attempt + 1,
                            maxRetries,
                            delay,
                            error
                        )
                    }

                    await new Promise((resolve) => setTimeout(resolve, delay))
                }
            }
        }

        return descriptor
    }
}
```

### 사용 예시

```typescript
// 프로바이더에서 데코레이터 적용
export class AnthropicHandler implements ApiHandler {
    @withRetry()  // 기본 설정 사용
    async *createMessage(...): ApiStream {
        // Rate limit 발생 시 자동 재시도
    }
}

export class OllamaHandler implements ApiHandler {
    @withRetry({ retryAllErrors: true, maxRetries: 5 })  // 모든 에러 재시도
    async *createMessage(...): ApiStream {
        // 로컬 서버 불안정성 대응
    }
}
```

### hdsp-agent 적용

```python
# backend/llm/retry.py
import asyncio
from functools import wraps
from typing import Callable, Any
from dataclasses import dataclass

@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    retry_all_errors: bool = False

class RetriableError(Exception):
    """재시도 가능한 에러"""
    def __init__(self, message: str, retry_after: float = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.status = 429

def with_retry(config: RetryConfig = None):
    """AsyncGenerator용 재시도 데코레이터"""
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(config.max_retries):
                try:
                    async for chunk in func(*args, **kwargs):
                        yield chunk
                    return  # 성공
                except Exception as e:
                    is_rate_limit = (
                        getattr(e, 'status', None) == 429 or
                        isinstance(e, RetriableError)
                    )
                    is_last = attempt == config.max_retries - 1

                    if (not is_rate_limit and not config.retry_all_errors) or is_last:
                        raise

                    # 대기 시간 계산
                    retry_after = getattr(e, 'retry_after', None)
                    if retry_after:
                        delay = float(retry_after)
                    else:
                        delay = min(
                            config.max_delay,
                            config.base_delay * (2 ** attempt)
                        )

                    await asyncio.sleep(delay)

        return wrapper
    return decorator


# 사용 예시
class GeminiHandler(LLMHandler):
    @with_retry(RetryConfig(max_retries=3))
    async def create_message(self, ...):
        async for chunk in self._raw_stream():
            yield self._normalize(chunk)
```

---

## 4. Mode 기반 설정

### 개요
Plan 모드와 Act 모드에서 서로 다른 모델과 설정을 사용할 수 있습니다. 이를 통해 계획 단계에서는 비용 효율적인 모델을, 실행 단계에서는 고성능 모델을 사용할 수 있습니다.

### 설정 구조

```typescript
// src/shared/storage/types.ts
export type Mode = "plan" | "act"

// src/shared/api.ts
export interface ApiConfiguration {
    // 공통 설정
    apiKey?: string

    // Plan 모드 전용
    planModeApiProvider?: string
    planModeApiModelId?: string
    planModeThinkingBudgetTokens?: number
    planModeReasoningEffort?: string

    // Act 모드 전용
    actModeApiProvider?: string
    actModeApiModelId?: string
    actModeThinkingBudgetTokens?: number
    actModeReasoningEffort?: string

    // 재시도 콜백
    onRetryAttempt?: (attempt: number, max: number, delay: number, error: Error) => void
}
```

### 팩토리에서의 모드 분기

```typescript
function createHandlerForProvider(
    apiProvider: string,
    options: ApiConfiguration,
    mode: Mode  // 핵심: 모드 파라미터
): ApiHandler {
    switch (apiProvider) {
        case "anthropic":
            return new AnthropicHandler({
                // 모드에 따라 다른 모델 사용
                apiModelId: mode === "plan"
                    ? options.planModeApiModelId     // 예: claude-haiku
                    : options.actModeApiModelId,     // 예: claude-sonnet

                // 모드에 따라 다른 토큰 예산
                thinkingBudgetTokens: mode === "plan"
                    ? options.planModeThinkingBudgetTokens   // 적은 예산
                    : options.actModeThinkingBudgetTokens,   // 많은 예산
            })
    }
}

// 사용 예시
const planHandler = buildApiHandler(config, "plan")  // 저비용 모델
const actHandler = buildApiHandler(config, "act")    // 고성능 모델
```

### hdsp-agent 적용

```python
# backend/config/llm_config.py
from dataclasses import dataclass
from typing import Literal

Mode = Literal["plan", "act"]

@dataclass
class ModeConfig:
    provider: str
    model_id: str
    thinking_budget: int = 0
    reasoning_effort: str = "medium"

@dataclass
class LLMConfiguration:
    api_key: str
    plan_mode: ModeConfig
    act_mode: ModeConfig

    def get_config_for_mode(self, mode: Mode) -> ModeConfig:
        return self.plan_mode if mode == "plan" else self.act_mode


# backend/llm/factory.py
def build_handler(config: LLMConfiguration, mode: Mode) -> LLMHandler:
    """모드별 핸들러 생성"""
    mode_config = config.get_config_for_mode(mode)

    return create_handler(
        provider=mode_config.provider,
        config={
            "api_key": config.api_key,
            "model_id": mode_config.model_id,
            "thinking_budget": mode_config.thinking_budget,
        },
        mode=mode
    )


# 사용 예시
config = LLMConfiguration(
    api_key="...",
    plan_mode=ModeConfig(provider="gemini", model_id="gemini-flash"),
    act_mode=ModeConfig(provider="gemini", model_id="gemini-pro")
)

plan_handler = build_handler(config, "plan")  # gemini-flash
act_handler = build_handler(config, "act")    # gemini-pro
```

---

## 5. 프로바이더 구현 패턴

### 기본 구조

```typescript
// 모든 프로바이더의 공통 패턴
export class ExampleHandler implements ApiHandler {
    private options: ExampleHandlerOptions
    private client: ExampleClient | undefined

    constructor(options: ExampleHandlerOptions) {
        this.options = options
    }

    // 지연 초기화 (Lazy Initialization)
    private ensureClient(): ExampleClient {
        if (!this.client) {
            if (!this.options.apiKey) {
                throw new Error("API key is required")
            }
            this.client = new ExampleClient({
                apiKey: this.options.apiKey,
                baseURL: this.options.baseUrl,
            })
        }
        return this.client
    }

    // 재시도 데코레이터 적용
    @withRetry()
    async *createMessage(
        systemPrompt: string,
        messages: ClineStorageMessage[],
        tools?: Tool[]
    ): ApiStream {
        const client = this.ensureClient()

        // 프로바이더별 메시지 포맷 변환
        const formattedMessages = this.formatMessages(messages)

        // 스트리밍 요청
        const stream = await client.chat.completions.create({
            model: this.getModel().id,
            messages: formattedMessages,
            stream: true,
        })

        // 정규화된 청크로 변환
        for await (const chunk of stream) {
            yield this.normalizeChunk(chunk)
        }
    }

    getModel(): ApiHandlerModel {
        return {
            id: this.options.modelId || "default-model",
            info: modelRegistry[this.options.modelId] || defaultModelInfo,
        }
    }
}
```

---

## 요약

| 패턴 | 핵심 이점 | hdsp-agent 적용 우선순위 |
|------|----------|--------------------------|
| Provider Factory | 프로바이더 추가 용이, 런타임 선택 | 높음 |
| Normalized Streaming | 일관된 응답 처리, UI 단순화 | 높음 |
| Retry Mechanism | 안정성, Rate Limit 대응 | 높음 |
| Mode 설정 | 비용 최적화, 유연성 | 중간 |
| 지연 초기화 | 리소스 효율성 | 낮음 |

### 즉시 재사용 가능

1. **ApiStream 타입**: 4가지 청크 타입 (text, usage, tool_calls, reasoning) 그대로 사용
2. **@withRetry 데코레이터**: Python으로 포팅하여 동일 로직 적용
3. **Factory 패턴**: Gemini, OpenAI, vLLM 통합에 동일 구조 활용
