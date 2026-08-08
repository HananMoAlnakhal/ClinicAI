from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    name: str

    @property
    def available(self) -> bool: ...

    def generate_sync(self, prompt: str, max_tokens: int) -> str: ...


@dataclass
class CallableProvider:
    name: str
    generator: Callable[[str, int], str]
    enabled: bool = True

    @property
    def available(self) -> bool:
        return self.enabled

    def generate_sync(self, prompt: str, max_tokens: int) -> str:
        return (self.generator(prompt, max_tokens) or "").strip()


class LocalFallbackProvider:
    """Deterministic final fallback: hand control back to the FSM safely."""

    name = "local_fsm"

    @property
    def available(self) -> bool:
        return True

    def generate_sync(self, prompt: str, max_tokens: int) -> str:
        return ""


class LLMRouter:
    def __init__(self, providers: list[LLMProvider], retries: int = 0):
        self.providers = providers
        self.retries = max(0, int(retries))
        self.last_provider: str | None = None

    @property
    def is_ready(self) -> bool:
        return any(provider.available for provider in self.providers)

    @property
    def active_provider(self) -> str | None:
        for provider in self.providers:
            if provider.available:
                return provider.name
        return None

    async def ask(self, prompt: str, max_tokens: int = 300) -> str:
        for provider in self.providers:
            if not provider.available:
                continue

            for attempt in range(self.retries + 1):
                if not provider.available:
                    break
                try:
                    result = await asyncio.to_thread(
                        provider.generate_sync, prompt, max_tokens
                    )
                    self.last_provider = provider.name
                    if result or provider.name == "local_fsm":
                        return result
                    break
                except Exception as exc:
                    logger.warning(
                        "LLM provider %s failed (attempt %s/%s): %s",
                        provider.name,
                        attempt + 1,
                        self.retries + 1,
                        exc,
                    )

        return ""
