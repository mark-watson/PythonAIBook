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


def get_ai_response(messages: list[dict[str, Any]]) -> str:
    """Send conversation history to the model and return its reply."""
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
    # litellm's specific errors (AuthenticationError, RateLimitError,
    # APIConnectionError, ...) all derive from OpenAIError, so this one clause
    # covers a missing key as well as a provider outage.
    except litellm.exceptions.OpenAIError as e:
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
                except litellm.exceptions.OpenAIError as e:
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
        except litellm.exceptions.OpenAIError as e:
            print(f"Error: {e}")
            print("Try again or type /quit to exit.")
            messages.pop()  # Remove the failed user message
            continue
        messages.append({"role": "assistant", "content": reply})
        print()
        print(reply)


if __name__ == "__main__":
    main()
