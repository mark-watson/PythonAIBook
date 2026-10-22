"""Text Adventure Game powered by Fireworks.ai LLMs.

This script runs a text-based adventure game using the Fireworks.ai API
with the deepseek-v4p1-flash model. The game master persona and setting are
defined in story.txt.

The LLM is reached through litellm (https://github.com/BerriAI/litellm), the
uniform interface to every provider, so the "fireworks_ai/" prefix in MODEL is
the only Fireworks-specific part.
"""

import sys
from typing import Any

import litellm

MODEL = "fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash"

DEFAULT_MAX_HISTORY_TURNS = 20
DEFAULT_MAX_CONTEXT_TOKENS = 16_000


def load_story() -> str:
    """Load the game master instructions from story.txt."""
    try:
        with open("story.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        print(
            "Error: story.txt not found. Please create story.txt with your adventure setting."
        )
        sys.exit(1)


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate the token count of a conversation, using tiktoken if available."""
    encoding: Any = None
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoding = None
    total = 0
    for message in messages:
        content = message.get("content") or ""
        if encoding is not None:
            total += len(encoding.encode(content))
        else:
            total += len(content) // 4
    return total


def trim_history(
    messages: list[dict[str, Any]],
    max_turns: int = DEFAULT_MAX_HISTORY_TURNS,
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
) -> list[dict[str, Any]]:
    """Trim conversation history down to the system prompt plus recent turns.

    Keeps the leading system message and the last ``max_turns`` user/assistant
    turns, dropping the oldest turns while the estimated token count exceeds
    ``max_tokens`` (the model context window budget).
    """
    if len(messages) < 2 or messages[0].get("role") != "system":
        return messages
    system_message = messages[0]
    turns = messages[1:]
    num_turns = len(turns) // 2
    while (
        num_turns > 0
        and len(turns) > 2
        and estimate_tokens([system_message, *turns]) > max_tokens
    ):
        turns = turns[2:]
        num_turns -= 1
    if num_turns > max_turns:
        turns = turns[-(2 * max_turns) :]
    return [system_message, *turns]


def get_ai_response(messages: list[dict[str, Any]]) -> str:
    """Send conversation history to the model and return its reply.

    Trims the history to the system prompt plus the most recent turns that fit
    the model's context window before sending, and updates the caller's list so
    the full history stays bounded.
    """
    trimmed = trim_history(messages)
    messages[:] = trimmed
    response = litellm.completion(model=MODEL, messages=messages)
    assert isinstance(response, litellm.ModelResponse), (
        "Expected a non-streaming response"
    )
    content = response.choices[0].message.content
    assert content is not None, "Model returned empty response"
    return content


def main() -> None:
    story_text = load_story()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": story_text},
    ]

    print("=" * 60)
    print("  TEXT ADVENTURE GAME")
    print("  Powered by Fireworks.ai — deepseek-v4p1-flash")
    print("=" * 60)
    print()
    print("Commands: /help  /restart  /quit")
    print()

    # Get the opening scene
    messages.append({"role": "user", "content": "Start the adventure."})
    try:
        reply = get_ai_response(messages)
    except Exception as e:
        # Catch broadly (auth, rate limits, timeouts, JSON errors, provider
        # outages, ...) and route to the same handling instead of letting the
        # game loop crash.
        print(f"Error connecting to Fireworks.ai: {e}")
        sys.exit(1)
    messages.append({"role": "assistant", "content": reply})
    print(reply)

    # Game loop
    while True:
        print()
        user_input = input("> ").strip()

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/q"):
                print("Thanks for playing!")
                break
            elif cmd == "/restart":
                messages = [
                    {"role": "system", "content": story_text},
                    {"role": "user", "content": "Start the adventure."},
                ]
                try:
                    reply = get_ai_response(messages)
                except Exception as e:
                    print(f"Error: {e}")
                    break
                messages.append({"role": "assistant", "content": reply})
                print("\n--- Restarted ---\n")
                print(reply)
                continue
            elif cmd == "/help":
                print("Commands: /help  /restart  /quit")
                print("Type your action or choice to advance the story.")
                continue
            else:
                print(f"Unknown command: {user_input}")
                continue

        messages.append({"role": "user", "content": user_input})
        try:
            reply = get_ai_response(messages)
        except Exception as e:
            print(f"Error: {e}")
            print("Try again or type /quit to exit.")
            messages.pop()  # Remove the failed user message
            continue
        messages.append({"role": "assistant", "content": reply})
        print()
        print(reply)


if __name__ == "__main__":
    main()
