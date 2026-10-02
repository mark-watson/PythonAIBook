"""Provider registry, model routing, and API key resolution."""

from __future__ import annotations

import pytest

import litelm

# The five providers the llm_local_models / llm_public_apis chapters use.
DEFAULTS = ["fireworks-ai", "gemini", "nvidia", "ollama", "openai"]


def test_exactly_the_chapter_providers_are_registered() -> None:
    """An exact set, not a subset: a provider silently disappearing from the
    registry must fail here."""
    assert litelm.providers() == DEFAULTS


def test_the_reference_ports_extras_are_one_call_away() -> None:
    """The Common Lisp and Racket ports also ship deepseek/mistral; this port
    documents them as register-it-yourself instead."""
    for name in ("deepseek", "mistral"):
        assert name not in litelm.providers()
    litelm.define_provider(
        "deepseek", "https://api.deepseek.com/v1", env_keys=["DEEPSEEK_API_KEY"]
    )
    provider, model = litelm.parse_model("deepseek/deepseek-chat")
    assert provider.base_url == "https://api.deepseek.com/v1"
    assert model == "deepseek-chat"


@pytest.mark.parametrize(
    ("model", "provider_name", "model_name"),
    [
        ("openai/gpt-5.4-nano", "openai", "gpt-5.4-nano"),
        ("ollama/llama3.2:3b", "ollama", "llama3.2:3b"),
        ("gemini/gemini-3-flash-preview", "gemini", "gemini-3-flash-preview"),
        (
            "nvidia/meta/llama-3.1-8b-instruct",
            "nvidia",
            "meta/llama-3.1-8b-instruct",
        ),
        (
            "fireworks-ai/accounts/fireworks/models/deepseek-v4-flash",
            "fireworks-ai",
            "accounts/fireworks/models/deepseek-v4-flash",
        ),
    ],
)
def test_parse_model_splits_on_first_slash(
    model: str, provider_name: str, model_name: str
) -> None:
    provider, parsed = litelm.parse_model(model)
    assert provider.name == provider_name
    assert parsed == model_name


def test_parse_model_honours_provider_override() -> None:
    provider, parsed = litelm.parse_model("meta/llama-3.1-8b-instruct", "nvidia")
    assert provider.name == "nvidia"
    assert parsed == "meta/llama-3.1-8b-instruct"


@pytest.mark.parametrize("model", ["gpt-5.4-nano", "openai/", "/gpt-5.4-nano"])
def test_parse_model_rejects_malformed_names(model: str) -> None:
    with pytest.raises(litelm.LitelmError):
        litelm.parse_model(model)


def test_unknown_provider_lists_the_known_ones() -> None:
    with pytest.raises(litelm.LitelmError) as excinfo:
        litelm.find_provider("acme")
    message = str(excinfo.value)
    assert "Unknown provider" in message
    assert "ollama" in message


def test_define_provider_registers_and_is_case_insensitive() -> None:
    provider = litelm.define_provider(
        "Groq", "https://api.groq.com/openai/v1", env_keys=["GROQ_API_KEY"]
    )
    assert provider.name == "groq"
    assert litelm.find_provider("GROQ") is provider
    assert "groq" in litelm.providers()
    assert provider.base_url == "https://api.groq.com/openai/v1"


def test_define_provider_accepts_a_single_env_key() -> None:
    provider = litelm.define_provider(
        "solo", "https://example.test/v1", env_keys="X_KEY"
    )
    assert provider.env_keys == ("X_KEY",)


def test_provider_url_strips_the_api_base_override() -> None:
    provider = litelm.find_provider("ollama")
    assert litelm.provider_url(provider, "/chat/completions") == (
        "http://localhost:11434/v1/chat/completions"
    )
    assert litelm.provider_url(
        provider, "/chat/completions", "http://box:9000/v1/"
    ) == ("http://box:9000/v1/chat/completions")


def test_api_key_prefers_the_explicit_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    provider = litelm.find_provider("openai")
    assert litelm.provider_api_key(provider, "explicit") == "explicit"


def test_api_key_falls_back_through_env_keys_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = litelm.find_provider("gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    assert litelm.provider_api_key(provider, None) == "google"
    monkeypatch.setenv("GEMINI_API_KEY", "gemini")
    assert litelm.provider_api_key(provider, None) == "gemini"


def test_api_key_missing_is_an_error_naming_the_variables() -> None:
    with pytest.raises(litelm.LitelmError) as excinfo:
        litelm.provider_api_key(litelm.find_provider("openai"), None)
    assert "OPENAI_API_KEY" in str(excinfo.value)


def test_keyless_provider_returns_none() -> None:
    assert litelm.provider_api_key(litelm.find_provider("ollama"), None) is None


def test_bearer_headers_include_authorization_only_with_a_key() -> None:
    assert litelm.bearer_headers(None) == {"Content-Type": "application/json"}
    assert litelm.bearer_headers("sk-1")["Authorization"] == "Bearer sk-1"


def test_empty_environment_variable_is_not_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(litelm.LitelmError):
        litelm.provider_api_key(litelm.find_provider("openai"), None)
