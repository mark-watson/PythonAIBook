"""Provider registry and ``"provider/model-name"`` routing.

Port of ``providers.lisp`` from the Common Lisp litelm library. Every provider
litelm ships with speaks the OpenAI-compatible ``/chat/completions`` protocol,
so one code path serves them all and the prefix in the model string is the only
thing that picks an endpoint::

    litelm.completion("ollama/llama3.2:3b", "hi")                 # local
    litelm.completion("openai/gpt-5.4-nano", "hi")                # cloud
    litelm.completion("nvidia/meta/llama-3.1-8b-instruct", "hi")  # nested name

Model names may themselves contain slashes; only the *first* one separates the
provider from the model, so Fireworks and NVIDIA model ids survive intact.

Register any other OpenAI-compatible service at runtime::

    litelm.define_provider("groq", "https://api.groq.com/openai/v1",
                           env_keys=["GROQ_API_KEY"])
    litelm.completion("groq/llama-3.1-8b-instant", "hi")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .errors import LitelmError

__all__ = [
    "Provider",
    "bearer_headers",
    "define_provider",
    "find_provider",
    "parse_model",
    "provider_api_key",
    "provider_url",
    "providers",
]


@dataclass(frozen=True)
class Provider:
    """One OpenAI-compatible endpoint plus the env vars holding its key."""

    name: str
    base_url: str
    env_keys: tuple[str, ...] = field(default_factory=tuple)
    requires_key: bool = True


_REGISTRY: dict[str, Provider] = {}


def define_provider(
    name: str,
    base_url: str,
    *,
    env_keys: str | tuple[str, ...] | list[str] = (),
    requires_key: bool = True,
) -> Provider:
    """Register (or replace) a provider and return it.

    ``env_keys`` may be a single variable name or a sequence tried in order.
    Local servers such as Ollama pass ``requires_key=False``.
    """
    if isinstance(env_keys, str):
        keys: tuple[str, ...] = (env_keys,)
    else:
        keys = tuple(env_keys)
    provider = Provider(
        name=_provider_key(name),
        base_url=base_url.rstrip("/"),
        env_keys=keys,
        requires_key=requires_key,
    )
    _REGISTRY[provider.name] = provider
    return provider


def find_provider(name: str) -> Provider:
    """Look up a provider by name; raise listing the known ones."""
    key = _provider_key(name)
    try:
        return _REGISTRY[key]
    except KeyError:
        known = ", ".join(providers())
        raise LitelmError(
            f"Unknown provider {name!r}. Known providers: {known}"
        ) from None


def providers() -> list[str]:
    """Sorted names of every registered provider."""
    return sorted(_REGISTRY)


def parse_model(model: str, provider: str | None = None) -> tuple[Provider, str]:
    """Split ``"provider/model-name"`` into ``(Provider, "model-name")``.

    Passing ``provider`` overrides the string prefix, which is how a caller
    pins an endpoint regardless of how the model id is spelled.
    """
    if provider is not None:
        return find_provider(provider), model
    if not isinstance(model, str) or "/" not in model:
        raise LitelmError(f'Model {model!r} must be of the form "provider/model-name"')
    prefix, _, model_name = model.partition("/")
    if not prefix or not model_name:
        raise LitelmError(f'Model {model!r} must be of the form "provider/model-name"')
    return find_provider(prefix), model_name


def provider_api_key(provider: Provider, explicit_key: str | None = None) -> str | None:
    """Explicit key wins; otherwise scan ``provider.env_keys`` in order.

    Returns ``None`` for keyless local providers, and raises for a keyed
    provider whose environment variable is unset.
    """
    if explicit_key:
        return explicit_key
    for var in provider.env_keys:
        value = os.environ.get(var)
        if value:
            return value
    if provider.requires_key:
        names = ", ".join(provider.env_keys) or "an API key"
        raise LitelmError(
            f"No API key for provider {provider.name!r}. "
            f"Pass api_key=... or set {names}"
        )
    return None


def provider_url(provider: Provider, path: str, api_base: str | None = None) -> str:
    """Full URL for ``path`` on ``provider``, honouring a per-call override."""
    return (api_base.rstrip("/") if api_base else provider.base_url) + path


def bearer_headers(api_key: str | None) -> dict[str, str]:
    """JSON content type plus a bearer token when there is a key."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _provider_key(name: str) -> str:
    return name.strip().lower()


# The five providers the two LLM chapters actually use. The Common Lisp and
# Racket ports also ship deepseek and mistral; here they are one
# define_provider call away (see README.md), which keeps the routing table
# honest about what this book demonstrates.
define_provider(
    "openai", "https://api.openai.com/v1", env_keys=("OPENAI_API_KEY", "OPENAI_KEY")
)
define_provider(
    "gemini",
    "https://generativelanguage.googleapis.com/v1beta/openai",
    env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
)
define_provider(
    "fireworks-ai",
    "https://api.fireworks.ai/inference/v1",
    env_keys=("FIREWORKS_API_KEY",),
)
define_provider(
    "nvidia", "https://integrate.api.nvidia.com/v1", env_keys=("NVIDIA_API_KEY",)
)
define_provider("ollama", "http://localhost:11434/v1", env_keys=(), requires_key=False)
