"""Text Adventure Game powered by Fireworks.ai LLMs.

This script runs a text-based adventure game using the Fireworks.ai API
with the deepseek-v4p1-flash model. The game master persona and setting are
defined in story.txt.

The LLM is reached through litelm, the book's uniform interface, so the
"fireworks-ai/" prefix in MODEL is the only Fireworks-specific part.
"""

import sys

import litelm

MODEL = "fireworks-ai/accounts/fireworks/models/deepseek-v4p1-flash"


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


def get_ai_response(messages: list[litelm.Message]) -> str:
    """Send conversation history to the model and return its reply."""
    response = litelm.completion(MODEL, messages)
    content = response.content
    assert content is not None, "Model returned empty response"
    return content


def main():
    story_text = load_story()

    messages: list[litelm.Message] = [
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
    except litelm.LitelmError as e:
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
                except litelm.LitelmError as e:
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
        except litelm.LitelmError as e:
            print(f"Error: {e}")
            print("Try again or type /quit to exit.")
            messages.pop()  # Remove the failed user message
            continue
        messages.append({"role": "assistant", "content": reply})
        print()
        print(reply)


if __name__ == "__main__":
    main()
