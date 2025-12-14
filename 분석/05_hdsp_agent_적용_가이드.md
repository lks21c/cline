# hdsp-agent 적용 가이드

Cline 프로젝트에서 추출한 패턴을 hdsp-agent (JupyterLab AI 확장)에 적용하는 실전 가이드입니다.

---

## 1. 적용 우선순위 매트릭스

### 높음 (즉시 적용)

| Cline 패턴 | hdsp-agent 적용 | 예상 효과 |
|-----------|----------------|----------|
| LLM Provider Factory | 동일 인터페이스 재사용 | 다중 LLM 통합 간소화 |
| Normalized Streaming | SSE 스트리밍에 적용 | UI 응답성 향상 |
| @withRetry 데코레이터 | Python 포팅 | API 안정성 확보 |
| useChatState 훅 | 채팅 상태 관리 | 코드 재사용성 |

### 중간 (1-2주 내)

| Cline 패턴 | hdsp-agent 적용 | 예상 효과 |
|-----------|----------------|----------|
| Host Abstraction | JupyterLabHost 구현 | IDE 독립적 코어 |
| StateManager | 노트북 상태 통합 | 영속성 개선 |
| useScrollBehavior | 채팅 스크롤 로직 | UX 향상 |

### 낮음 (추후 개선)

| Cline 패턴 | hdsp-agent 적용 | 예상 효과 |
|-----------|----------------|----------|
| Proto 메시지 | Pydantic 모델 | 타입 안전성 |
| 접근성 훅 | 스크린 리더 지원 | 사용성 확대 |
| 가상화 | 대량 메시지 처리 | 성능 최적화 |

---

## 2. LLM 통합 구현

### 2.1 기본 인터페이스 정의

```python
# backend/llm/base.py
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class ChunkType(str, Enum):
    TEXT = "text"
    REASONING = "reasoning"
    USAGE = "usage"
    TOOL_CALLS = "tool_calls"

@dataclass
class StreamChunk:
    type: ChunkType
    text: str = ""
    reasoning: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    tool_name: str = ""
    tool_args: Dict[str, Any] = None

@dataclass
class ModelInfo:
    id: str
    name: str
    max_tokens: int
    supports_streaming: bool = True
    supports_tools: bool = False

class LLMHandler(ABC):
    """모든 LLM 프로바이더의 기본 인터페이스"""

    @abstractmethod
    async def create_message(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """스트리밍 메시지 생성"""
        pass

    @abstractmethod
    def get_model(self) -> ModelInfo:
        """모델 정보 반환"""
        pass

    async def abort(self) -> None:
        """요청 취소"""
        pass
```

### 2.2 재시도 데코레이터

```python
# backend/llm/retry.py
import asyncio
from functools import wraps
from typing import Callable, TypeVar, Any
from dataclasses import dataclass

@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    retry_all_errors: bool = False

class RetriableError(Exception):
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
            last_error = None

            for attempt in range(config.max_retries):
                try:
                    async for chunk in func(*args, **kwargs):
                        yield chunk
                    return  # 성공
                except Exception as e:
                    last_error = e
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
                        # 지수 백오프
                        delay = min(
                            config.max_delay,
                            config.base_delay * (2 ** attempt)
                        )

                    await asyncio.sleep(delay)

            if last_error:
                raise last_error

        return wrapper
    return decorator
```

### 2.3 Gemini 프로바이더 구현

```python
# backend/llm/providers/gemini.py
import google.generativeai as genai
from typing import AsyncGenerator, Optional, List, Dict, Any
from ..base import LLMHandler, StreamChunk, ModelInfo, ChunkType
from ..retry import with_retry, RetryConfig

class GeminiHandler(LLMHandler):
    def __init__(
        self,
        api_key: str,
        model_id: str = "gemini-pro",
        thinking_budget: int = 0
    ):
        self.api_key = api_key
        self.model_id = model_id
        self.thinking_budget = thinking_budget
        self._client = None

    def _ensure_client(self):
        if not self._client:
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model_id)
        return self._client

    @with_retry(RetryConfig(max_retries=3))
    async def create_message(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        client = self._ensure_client()

        # 메시지 변환
        contents = self._format_messages(messages)

        # 스트리밍 요청
        response = await client.generate_content_async(
            contents,
            generation_config=genai.GenerationConfig(
                system_instruction=system_prompt
            ),
            stream=True
        )

        total_input = 0
        total_output = 0

        async for chunk in response:
            if chunk.text:
                yield StreamChunk(type=ChunkType.TEXT, text=chunk.text)

            # 사용량 추적
            if hasattr(chunk, 'usage_metadata'):
                total_input = chunk.usage_metadata.prompt_token_count
                total_output = chunk.usage_metadata.candidates_token_count

        # 최종 사용량
        yield StreamChunk(
            type=ChunkType.USAGE,
            input_tokens=total_input,
            output_tokens=total_output
        )

    def _format_messages(self, messages: List[Dict]) -> List[Dict]:
        """Cline 메시지 형식을 Gemini 형식으로 변환"""
        formatted = []
        for msg in messages:
            role = "user" if msg.get("role") == "user" else "model"
            formatted.append({
                "role": role,
                "parts": [{"text": msg.get("content", "")}]
            })
        return formatted

    def get_model(self) -> ModelInfo:
        return ModelInfo(
            id=self.model_id,
            name=f"Gemini {self.model_id}",
            max_tokens=8192,
            supports_streaming=True,
            supports_tools=True
        )
```

### 2.4 팩토리 함수

```python
# backend/llm/factory.py
from typing import Literal
from .base import LLMHandler
from .providers.gemini import GeminiHandler
from .providers.openai import OpenAIHandler
from .providers.vllm import VLLMHandler

Mode = Literal["plan", "act"]

def create_handler(
    provider: str,
    config: dict,
    mode: Mode = "act"
) -> LLMHandler:
    """프로바이더별 핸들러 생성"""
    handlers = {
        "gemini": GeminiHandler,
        "openai": OpenAIHandler,
        "vllm": VLLMHandler,
    }

    handler_class = handlers.get(provider)
    if not handler_class:
        raise ValueError(f"Unknown provider: {provider}")

    # 모드별 설정 선택
    model_id = config.get(f"{mode}_mode_model_id", config.get("model_id"))
    thinking_budget = config.get(f"{mode}_mode_thinking_budget", 0)

    return handler_class(
        api_key=config.get("api_key"),
        model_id=model_id,
        thinking_budget=thinking_budget,
    )
```

---

## 3. 상태 관리 구현

### 3.1 StateManager (Python)

```python
# backend/services/state_manager.py
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
import aiofiles

@dataclass
class AgentState:
    current_task_id: Optional[str] = None
    is_streaming: bool = False
    mode: str = "act"
    messages: list = None
    api_config: dict = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.api_config is None:
            self.api_config = {}

class StateManager:
    """계층적 상태 관리 (인메모리 + 디바운스 영속화)"""

    _instance = None
    DEBOUNCE_MS = 500

    def __init__(self, config_path: Path):
        self._state = AgentState()
        self._config_path = config_path
        self._pending_writes: Dict[str, asyncio.Task] = {}
        self._subscribers: Set[asyncio.Queue] = set()

    @classmethod
    def get_instance(cls) -> "StateManager":
        if cls._instance is None:
            config_path = Path.home() / ".jupyter" / "hdsp_agent_state.json"
            cls._instance = cls(config_path)
        return cls._instance

    # 동기적 읽기 (인메모리)
    def get_state(self) -> AgentState:
        return self._state

    def get_value(self, key: str) -> Any:
        return getattr(self._state, key, None)

    # 디바운스된 상태 업데이트
    async def set_value(self, key: str, value: Any):
        # 1. 즉시 인메모리 업데이트
        setattr(self._state, key, value)

        # 2. 구독자에게 알림
        await self._notify_subscribers()

        # 3. 기존 펜딩 취소
        if key in self._pending_writes:
            self._pending_writes[key].cancel()

        # 4. 디바운스된 디스크 저장
        self._pending_writes[key] = asyncio.create_task(
            self._debounced_persist(key)
        )

    async def _debounced_persist(self, key: str):
        await asyncio.sleep(self.DEBOUNCE_MS / 1000)
        await self._persist_to_disk()
        del self._pending_writes[key]

    async def _persist_to_disk(self):
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(self._config_path, "w") as f:
            await f.write(json.dumps(asdict(self._state), indent=2))

    async def load_from_disk(self):
        if self._config_path.exists():
            async with aiofiles.open(self._config_path) as f:
                data = json.loads(await f.read())
                self._state = AgentState(**data)

    # 구독 관리
    def subscribe(self) -> asyncio.Queue:
        queue = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        self._subscribers.discard(queue)

    async def _notify_subscribers(self):
        state_dict = asdict(self._state)
        for queue in self._subscribers:
            await queue.put(state_dict)
```

### 3.2 SSE 스트리밍 엔드포인트

```python
# backend/api/streaming.py
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from ..services.state_manager import StateManager
import json

router = APIRouter()

@router.get("/stream/state")
async def subscribe_to_state():
    """상태 변경 SSE 스트리밍"""
    state_manager = StateManager.get_instance()

    async def event_generator():
        queue = state_manager.subscribe()
        try:
            # 초기 상태 전송
            yield {
                "event": "state",
                "data": json.dumps(state_manager.get_state().__dict__)
            }

            # 변경 사항 스트리밍
            while True:
                state = await queue.get()
                yield {
                    "event": "state",
                    "data": json.dumps(state)
                }
        finally:
            state_manager.unsubscribe(queue)

    return EventSourceResponse(event_generator())
```

---

## 4. 프론트엔드 훅 구현

### 4.1 useChatState

```typescript
// frontend/hooks/useChatState.ts
import { useState, useMemo, useCallback, useRef } from "react"

export interface Message {
    id: string
    type: "user" | "assistant"
    content: string
    timestamp: number
    isStreaming?: boolean
}

export interface ChatState {
    inputValue: string
    setInputValue: (v: string) => void
    messages: Message[]
    addMessage: (msg: Message) => void
    updateLastMessage: (content: string) => void
    isStreaming: boolean
    setIsStreaming: (v: boolean) => void
    lastMessage: Message | undefined
    resetState: () => void
    textAreaRef: React.RefObject<HTMLTextAreaElement>
}

export function useChatState(): ChatState {
    const [inputValue, setInputValue] = useState("")
    const [messages, setMessages] = useState<Message[]>([])
    const [isStreaming, setIsStreaming] = useState(false)
    const textAreaRef = useRef<HTMLTextAreaElement>(null)

    const lastMessage = useMemo(() => messages.at(-1), [messages])

    const addMessage = useCallback((msg: Message) => {
        setMessages(prev => [...prev, msg])
    }, [])

    const updateLastMessage = useCallback((content: string) => {
        setMessages(prev => {
            if (prev.length === 0) return prev
            const updated = [...prev]
            updated[updated.length - 1] = {
                ...updated[updated.length - 1],
                content
            }
            return updated
        })
    }, [])

    const resetState = useCallback(() => {
        setInputValue("")
        setMessages([])
        setIsStreaming(false)
    }, [])

    return {
        inputValue, setInputValue,
        messages, addMessage, updateLastMessage,
        isStreaming, setIsStreaming,
        lastMessage, resetState, textAreaRef,
    }
}
```

### 4.2 useScrollBehavior

```typescript
// frontend/hooks/useScrollBehavior.ts
import { useRef, useCallback, useEffect, useMemo } from "react"
import debounce from "lodash/debounce"

export function useScrollBehavior(messagesLength: number) {
    const containerRef = useRef<HTMLDivElement>(null)
    const autoScrollEnabled = useRef(true)

    const scrollToBottom = useMemo(
        () => debounce(() => {
            if (autoScrollEnabled.current && containerRef.current) {
                containerRef.current.scrollTo({
                    top: containerRef.current.scrollHeight,
                    behavior: "smooth"
                })
            }
        }, 50),
        []
    )

    // 메시지 추가 시 자동 스크롤
    useEffect(() => {
        scrollToBottom()
    }, [messagesLength, scrollToBottom])

    // 수동 스크롤 감지
    const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
        const target = e.target as HTMLDivElement
        const isNearBottom =
            target.scrollHeight - target.scrollTop - target.clientHeight < 100
        autoScrollEnabled.current = isNearBottom
    }, [])

    // 휠 이벤트로 자동 스크롤 비활성화
    useEffect(() => {
        const handleWheel = (e: WheelEvent) => {
            if (e.deltaY < 0) {  // 위로 스크롤
                autoScrollEnabled.current = false
            }
        }
        window.addEventListener("wheel", handleWheel, { passive: true })
        return () => window.removeEventListener("wheel", handleWheel)
    }, [])

    return { containerRef, handleScroll, scrollToBottom }
}
```

### 4.3 AgentContext

```typescript
// frontend/context/AgentContext.tsx
import React, { createContext, useContext, useState, useEffect, useMemo, useCallback } from "react"
import { useChatState, ChatState, Message } from "../hooks/useChatState"

interface AgentConfig {
    provider: string
    modelId: string
    mode: "plan" | "act"
}

interface AgentContextType extends ChatState {
    config: AgentConfig
    setConfig: (c: Partial<AgentConfig>) => void
    isConnected: boolean
    sendMessage: (content: string) => Promise<void>
}

const AgentContext = createContext<AgentContextType | undefined>(undefined)

export function AgentProvider({ children }: { children: React.ReactNode }) {
    const chatState = useChatState()
    const [config, setConfigState] = useState<AgentConfig>({
        provider: "gemini",
        modelId: "gemini-pro",
        mode: "act"
    })
    const [isConnected, setIsConnected] = useState(false)

    // SSE 연결
    useEffect(() => {
        const eventSource = new EventSource("/hdsp-agent/stream/messages")

        eventSource.onopen = () => setIsConnected(true)
        eventSource.onerror = () => setIsConnected(false)

        eventSource.addEventListener("chunk", (event) => {
            const chunk = JSON.parse(event.data)
            if (chunk.type === "text") {
                chatState.updateLastMessage(
                    (chatState.lastMessage?.content || "") + chunk.text
                )
            }
        })

        eventSource.addEventListener("done", () => {
            chatState.setIsStreaming(false)
        })

        return () => eventSource.close()
    }, [])

    const sendMessage = useCallback(async (content: string) => {
        // 사용자 메시지 추가
        chatState.addMessage({
            id: crypto.randomUUID(),
            type: "user",
            content,
            timestamp: Date.now()
        })

        // 어시스턴트 메시지 플레이스홀더
        chatState.addMessage({
            id: crypto.randomUUID(),
            type: "assistant",
            content: "",
            timestamp: Date.now(),
            isStreaming: true
        })

        chatState.setIsStreaming(true)
        chatState.setInputValue("")

        // API 호출
        await fetch("/hdsp-agent/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content, config })
        })
    }, [config, chatState])

    const setConfig = useCallback((c: Partial<AgentConfig>) => {
        setConfigState(prev => ({ ...prev, ...c }))
    }, [])

    const value = useMemo(() => ({
        ...chatState,
        config, setConfig,
        isConnected, sendMessage,
    }), [chatState, config, isConnected, sendMessage])

    return (
        <AgentContext.Provider value={value}>
            {children}
        </AgentContext.Provider>
    )
}

export function useAgent() {
    const context = useContext(AgentContext)
    if (!context) {
        throw new Error("useAgent must be used within AgentProvider")
    }
    return context
}
```

---

## 5. 초기화 순서

### 5.1 Backend 초기화

```python
# backend/__init__.py
import logging
from .services.state_manager import StateManager
from .services.config_manager import ConfigManager
from .llm.factory import create_handler
from .api.routes import setup_routes

logger = logging.getLogger(__name__)

async def initialize_extension(server_app):
    """확장 초기화 순서"""

    # 1. 상태 관리자 초기화
    state_manager = StateManager.get_instance()
    await state_manager.load_from_disk()
    logger.info("StateManager initialized")

    # 2. 설정 로드
    config_manager = ConfigManager.get_instance()
    await config_manager.load_config()
    logger.info("ConfigManager initialized")

    # 3. LLM 핸들러 초기화 (지연 로딩)
    # create_handler는 실제 요청 시 호출됨
    logger.info("LLM handlers ready for lazy initialization")

    # 4. API 라우트 등록
    setup_routes(server_app.web_app)
    logger.info("API routes registered")

    # 5. 준비 완료
    logger.info("HDSP Agent extension loaded successfully")


def _load_jupyter_server_extension(server_app):
    """JupyterLab 서버 확장 진입점"""
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(initialize_extension(server_app))
```

### 5.2 Frontend 초기화

```typescript
// frontend/index.tsx
import { JupyterFrontEnd, JupyterFrontEndPlugin } from "@jupyterlab/application"
import { AgentProvider } from "./context/AgentContext"
import { AgentPanel } from "./components/AgentPanel"

const plugin: JupyterFrontEndPlugin<void> = {
    id: "hdsp-agent:plugin",
    autoStart: true,
    activate: async (app: JupyterFrontEnd) => {
        console.log("HDSP Agent extension activated")

        // 1. 패널 위젯 생성
        const panel = new AgentPanelWidget()

        // 2. 사이드바에 추가
        app.shell.add(panel, "right", { rank: 1000 })

        // 3. 명령어 등록
        app.commands.addCommand("hdsp-agent:toggle", {
            label: "Toggle HDSP Agent",
            execute: () => {
                if (panel.isVisible) {
                    panel.hide()
                } else {
                    panel.show()
                }
            }
        })

        console.log("HDSP Agent UI ready")
    }
}

export default plugin
```

---

## 6. 마이그레이션 체크리스트

### Phase 1: 기반 구축 (1주)

- [ ] LLMHandler 인터페이스 정의
- [ ] @with_retry 데코레이터 구현
- [ ] StreamChunk 타입 정의
- [ ] 첫 번째 프로바이더 (Gemini) 구현
- [ ] 팩토리 함수 구현

### Phase 2: 상태 관리 (1주)

- [ ] StateManager 싱글톤 구현
- [ ] 디바운스 영속화 로직
- [ ] SSE 스트리밍 엔드포인트
- [ ] 프론트엔드 Context 구현

### Phase 3: UI 통합 (1주)

- [ ] useChatState 훅 구현
- [ ] useScrollBehavior 훅 구현
- [ ] 채팅 컴포넌트 연결
- [ ] 메시지 렌더링 컴포넌트

### Phase 4: 추가 프로바이더 (1주)

- [ ] OpenAI 프로바이더
- [ ] vLLM 프로바이더
- [ ] 모드별 설정 (Plan/Act)
- [ ] 모델 선택 UI

---

## 7. 코드 재사용 매핑

| Cline 파일 | hdsp-agent 대응 | 재사용률 |
|-----------|----------------|---------|
| `src/core/api/index.ts` | `backend/llm/factory.py` | 90% |
| `src/core/api/transform/stream.ts` | `backend/llm/base.py` | 95% |
| `src/core/api/retry.ts` | `backend/llm/retry.py` | 85% |
| `hooks/useChatState.ts` | `frontend/hooks/useChatState.ts` | 80% |
| `hooks/useScrollBehavior.ts` | `frontend/hooks/useScrollBehavior.ts` | 75% |
| `context/ExtensionStateContext.tsx` | `frontend/context/AgentContext.tsx` | 70% |
| `src/hosts/HostProvider.ts` | `backend/hosts/jupyter_host.py` | 60% |

---

## 8. 주요 차이점 및 주의사항

### VS Code → JupyterLab

| 항목 | VS Code (Cline) | JupyterLab (hdsp-agent) |
|-----|-----------------|------------------------|
| 통신 | Message Passing + gRPC-like | REST + SSE |
| 상태 저장 | VS Code globalState | 파일 시스템 |
| UI 프레임워크 | Webview (React) | Lumino + React |
| 패키지 관리 | npm | pip + npm |
| 확장 API | vscode.* | @jupyterlab/* |

### 변환 시 주의사항

1. **비동기 처리**: Python의 asyncio와 TypeScript의 Promise 차이 고려
2. **타입 시스템**: TypeScript 인터페이스 → Python dataclass/Pydantic
3. **스트리밍**: gRPC streaming → SSE (Server-Sent Events)
4. **상태 영속화**: VS Code secrets → 환경 변수 또는 암호화 파일
5. **노트북 통합**: 셀 실행 컨텍스트, 커널 상태 연동 필요

---

## 요약

Cline 프로젝트에서 추출한 핵심 패턴을 hdsp-agent에 적용하면:

1. **LLM 통합 시간 단축**: Provider Factory + Normalized Streaming으로 다중 LLM 지원 간소화
2. **안정성 향상**: @withRetry 데코레이터로 API 오류 복원력 확보
3. **UX 개선**: 커스텀 훅 패턴으로 채팅 UI 응답성 향상
4. **유지보수성**: 계층적 상태 관리로 코드 구조화

4주 내 핵심 기능 구현 가능하며, 이후 점진적 개선을 통해 완성도를 높일 수 있습니다.
