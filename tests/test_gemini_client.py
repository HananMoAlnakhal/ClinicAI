"""Tests for LLM client: OpenRouter primary + Gemma fallback."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from tests.helpers import run_async


@pytest.fixture
def fresh_llm_client():
    with patch("nlp.gemini_client.OPENROUTER_AVAILABLE", True), patch(
        "nlp.gemini_client.GEMINI_AVAILABLE", True
    ), patch("nlp.gemini_client.gemini_client", object()), patch(
        "nlp.gemini_client.LLM_PRIMARY_MODEL", "openai/gpt-4.1-mini"
    ), patch(
        "nlp.gemini_client.LLM_FALLBACK_MODEL", "gemma-4-31b-it"
    ):
        from nlp.gemini_client import GeminiClient

        return GeminiClient()


def test_is_ready_stays_true_on_fallback_after_primary_quota(fresh_llm_client):
    client = fresh_llm_client
    client._enter_primary_cooldown(60.0, "test")
    assert client.is_ready is False

    client._activate_fallback("test")
    assert client.is_ready is True
    assert client.active_model == "gemma-4-31b-it"


def test_ask_switches_to_fallback_on_primary_quota_error(fresh_llm_client):
    client = fresh_llm_client
    calls: list[str] = []

    def fake_generate(model: str, prompt: str, max_tokens: int) -> str:
        calls.append(model)
        if model == "openai/gpt-4.1-mini":
            raise Exception("429 RESOURCE_EXHAUSTED quota exceeded")
        return "رد من gemma"

    client._generate_sync = fake_generate  # type: ignore[method-assign]

    reply = run_async(client.ask("مرحبا", max_tokens=50))
    assert reply == "رد من gemma"
    assert calls == ["openai/gpt-4.1-mini", "gemma-4-31b-it"]
    assert client.active_model == "gemma-4-31b-it"
    assert client.is_ready is True


def test_ask_uses_fallback_directly_after_switch(fresh_llm_client):
    client = fresh_llm_client
    client._activate_fallback("prior quota hit")
    calls: list[str] = []

    def fake_generate(model: str, prompt: str, max_tokens: int) -> str:
        calls.append(model)
        return "ok"

    client._generate_sync = fake_generate  # type: ignore[method-assign]

    reply = run_async(client.ask("test"))
    assert reply == "ok"
    assert calls == ["gemma-4-31b-it"]


def test_primary_restored_after_cooldown(fresh_llm_client):
    client = fresh_llm_client
    client._enter_primary_cooldown(60.0, "quota")
    client._activate_fallback("quota")
    client._primary_cooldown_until = time.monotonic() - 1
    client._maybe_restore_primary_model()
    assert client.active_model == "openai/gpt-4.1-mini"
    assert client._using_fallback is False


def test_is_transient_error_detects_winerror_10054():
    from nlp.gemini_client import GeminiClient

    exc = OSError("[WinError 10054] An existing connection was forcibly closed by the remote host")
    assert GeminiClient._is_transient_error(exc) is True


def test_ask_retries_primary_on_transient_then_succeeds(fresh_llm_client):
    client = fresh_llm_client
    attempts = {"count": 0}

    def fake_generate(model: str, prompt: str, max_tokens: int) -> str:
        if model != "openai/gpt-4.1-mini":
            return "fallback"
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise OSError("[WinError 10054] An existing connection was forcibly closed by the remote host")
        return "نجح بعد إعادة المحاولة"

    client._generate_sync = fake_generate  # type: ignore[method-assign]

    reply = run_async(client.ask("مرحبا", max_tokens=50))
    assert reply == "نجح بعد إعادة المحاولة"
    assert attempts["count"] == 2
    assert client.active_model == "openai/gpt-4.1-mini"


def test_ask_switches_to_fallback_on_transient_after_retries(fresh_llm_client):
    client = fresh_llm_client
    calls: list[str] = []

    def fake_generate(model: str, prompt: str, max_tokens: int) -> str:
        calls.append(model)
        if model == "openai/gpt-4.1-mini":
            raise OSError("[WinError 10054] An existing connection was forcibly closed by the remote host")
        return "رد من gemma"

    client._generate_sync = fake_generate  # type: ignore[method-assign]

    reply = run_async(client.ask("مرحبا", max_tokens=50))
    assert reply == "رد من gemma"
    assert calls.count("openai/gpt-4.1-mini") == 3
    assert "gemma-4-31b-it" in calls
    assert client.active_model == "gemma-4-31b-it"
