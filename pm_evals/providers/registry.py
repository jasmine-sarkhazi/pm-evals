"""Pick a provider from a model string.

* ``claude-*``                    -> Anthropic (``ANTHROPIC_API_KEY``)
* ``gpt-*``, ``o1*``, ``o3*``, ``o4*``, ``chatgpt-*`` -> OpenAI (``OPENAI_API_KEY``)
* ``gemini-*``                    -> Google's OpenAI-compatible endpoint (``GEMINI_API_KEY``)
* ``groq:<model>``                -> Groq (``GROQ_API_KEY``)
* ``mistral:<model>``             -> Mistral (``MISTRAL_API_KEY``)
* ``ollama:<model>``              -> local Ollama (no key)
* ``openai-compat:<model>``       -> any endpoint via ``OPENAI_COMPAT_BASE_URL`` / ``OPENAI_COMPAT_API_KEY``
* ``mock`` / ``mock:<behaviour>`` -> scripted provider for dry runs and tests

Judges only: ``jev`` / ``typesafe`` (or ``system-one``) route to TypeSafe's
System One model via :func:`make_judge`; ``jev:mock`` is an offline stand-in.
"""

from __future__ import annotations

import os
from typing import Any

from .base import LLMProvider, ProviderError

PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openai-compat": "OPENAI_COMPAT_API_KEY",
    "typesafe": "TYPESAFE_API_KEY",
}

# Judge-only model strings that route to TypeSafe's System One model (Jev).
_SYSTEM_ONE_NAMES = ("jev", "typesafe", "system-one", "system_one")

_COMPAT_BASES = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "ollama": "http://localhost:11434/v1",
}


def provider_for(model: str) -> str:
    m = model.lower()
    if m.startswith("mock"):
        return "mock"
    if ":" in m:
        prefix = m.split(":", 1)[0]
        if prefix in ("groq", "mistral", "ollama", "openai-compat", "openai", "anthropic", "gemini"):
            return prefix
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if m.startswith("gemini"):
        return "gemini"
    return "openai-compat" if os.environ.get("OPENAI_COMPAT_BASE_URL") else "anthropic"


def make_provider(model: str, **overrides: Any) -> LLMProvider:
    kind = provider_for(model)
    bare = model.split(":", 1)[1] if ":" in model and kind != "mock" else model
    if kind == "mock":
        from .mock import MockProvider

        behaviour = model.split(":", 1)[1] if ":" in model else "perfect"
        return MockProvider(model=model, behaviour=behaviour)
    if kind == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(bare, api_key=overrides.get("api_key"), base_url=overrides.get("base_url"))
    from .openai_provider import OpenAIProvider

    if kind == "openai":
        return OpenAIProvider(bare, api_key=overrides.get("api_key"), base_url=overrides.get("base_url"), label="openai")
    if kind in _COMPAT_BASES:
        key = overrides.get("api_key") or os.environ.get(PROVIDER_ENV.get(kind, ""), "")
        return OpenAIProvider(bare, api_key=key or None, base_url=overrides.get("base_url") or _COMPAT_BASES[kind], label=kind)
    base = overrides.get("base_url") or os.environ.get("OPENAI_COMPAT_BASE_URL")
    if not base:
        raise ProviderError(
            f"Don't know how to reach model '{model}'. Use a claude-/gpt-/gemini- model name, "
            "or set OPENAI_COMPAT_BASE_URL and use 'openai-compat:<model>'."
        )
    return OpenAIProvider(bare, api_key=overrides.get("api_key") or os.environ.get("OPENAI_COMPAT_API_KEY"), base_url=base, label="openai-compat")


def is_system_one_model(model: str) -> bool:
    """True for judge model strings that select TypeSafe's System One (Jev)."""
    m = (model or "").strip().lower()
    return m in _SYSTEM_ONE_NAMES or any(m.startswith(f"{n}:") for n in _SYSTEM_ONE_NAMES)


def make_judge(model: str, **overrides: Any) -> Any:
    """Build a judge from a model string.

    Routes System One model strings (``jev``/``typesafe``/``system-one``, and the
    offline ``jev:mock``) to :class:`TypeSafeJevJudge`; every other string is a
    normal :class:`LLMProvider` from :func:`make_provider`."""
    if is_system_one_model(model):
        from .typesafe_provider import MockSystemOneClient, TypeSafeJevJudge

        sub = model.split(":", 1)[1].lower() if ":" in model else ""
        if sub == "mock":
            return TypeSafeJevJudge(model=model, client=MockSystemOneClient())
        return TypeSafeJevJudge(model=model, api_key=overrides.get("api_key"))
    return make_provider(model, **overrides)


def configured_providers() -> dict[str, bool]:
    out = {k: bool(os.environ.get(v)) for k, v in PROVIDER_ENV.items()}
    out["ollama"] = True
    out["mock"] = True
    if os.environ.get("OPENAI_COMPAT_BASE_URL"):
        out["openai-compat"] = True
    return out
