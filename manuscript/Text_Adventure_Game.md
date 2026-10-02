# Text Adventure Game with an LLM Game Master

In the 1970s, text adventure games like *Colossal Cave Adventure* and *Zork* were the closest thing we had to virtual reality. You typed commands like `go north` or `take lamp` and the game described what happened next. Those games were built with hand-crafted parsers and painstakingly authored room descriptions. Today we can build something far more flexible in under 150 lines of Python by handing the storytelling to a large language model.

In the mid-1970s I wrote a popular open source Apple Basic text adventure game. I wrote the game initially on a huge sheet of butcher block paper, each location being a bubble with descriptive text and arrows between the bubbles defining states of a transition network. It was a lot of work writing it out on paper and then manually converting the data to a Basic program. Here, we get better results with a short Python program.

In this chapter we build a terminal-based text adventure game where an LLM acts as the Game Master. The model generates scenes, presents choices, tracks your inventory, and adapts the story to your decisions, all in real time. There is no pre-scripted plot. Every run through the game is different.

The examples for this chapter are in the directory **source-code/text-adventure-game**.

## How It Works

The architecture is simple. We maintain a conversation history as a list of message objects: a system message that defines the game world and rules, then alternating user and assistant messages as the player and Game Master take turns. Each time the player types an action, we append it to the history, send the whole conversation to the model, and append the response. The model's entire memory of the game state lives in that growing conversation history.

```
┌──────────┐     ┌─────────────────────┐     ┌──────────────┐
│  Player  │────▶│  Conversation       │────▶│  Fireworks   │
│  input   │     │  History (messages) │     │  API (LLM)   │
└──────────┘     └─────────────────────┘     └──────────────┘
                        │                           │
                        │  append reply             │
                        ◀───────────────────────────┘
```

The system prompt is the secret sauce. It defines the Game Master's personality, the setting, the rules, and the narrative style. We load it from a separate file so you can swap in new worlds without touching the code.

## The System Prompt

The file `story.txt` contains the instructions that shape the entire game experience. Let's look at it:

```python
# story.txt Game Master instructions and world definition

You are the Game Master of a text adventure game. Your role is to:

1. Describe the current scene vividly but concisely (2-4 sentences).
2. Present 3-4 clear choices the player can take, labeled as options.
3. Respond to the player's choice by advancing the story in an interesting direction.
4. Keep the story coherent and remember what has happened.
5. Occasionally introduce surprises, puzzles, or challenges.
6. End with "What do you do?" after presenting options.

The setting: You are in a mysterious ancient valley rumored to hold
a powerful artifact. Strange creatures and forgotten magic linger here.
The player is an adventurer seeking the artifact.

Start the game by describing the entry to the valley and offering
initial choices. Keep track of the player's inventory (they start
with a map, a dagger, and 3 gold coins). Make choices matter with some
lead to treasure, others to peril. Include at least one NPC the player
can encounter.
```

This prompt does several important things at once. It establishes a *persona* (Game Master), sets *constraints* (2-4 sentence descriptions, 3-4 choices), defines a *world*, gives the player *starting equipment*, and instructs the model to include *NPCs* and *consequences*. The numbered list format helps the model follow the rules consistently.

{class: tip}
The system prompt is the most important part of any LLM-powered application. When building your own adventures, spend time iterating on the prompt. Add rules, remove rules, change the tone. A small wording change can dramatically alter the game's personality.

## The Game Engine

The Python code is straightforward. It loads the story, calls the model through [**litellm**](https://github.com/BerriAI/litellm), and runs a game loop. Here is the complete program:

```python
# game.py  Text Adventure Game powered by Fireworks.ai LLMs

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
```

### Walking Through the Code

**The LLM call** goes through `litellm.completion`, the uniform interface we met in the previous two chapters. The model string `fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash` is the only Fireworks-specific part of the program: the `fireworks_ai/` prefix tells litellm which provider to call, and litellm finds the endpoint and reads `FIREWORKS_API_KEY` for us. The reply comes back as a `litellm.ModelResponse`, and the text we want lives at `response.choices[0].message.content` — the same shape for every provider litellm talks to.

**The model** is DeepSeek V4 Flash, which is fast and inexpensive. Fast responses are important qualities when a player is waiting for the next scene. I prototyped this example using a local model running on Ollama and the game play was not much fun because the responses were very slow. You can substitute any model litellm can reach by changing the `MODEL` constant — litellm dispatches on the provider prefix, so the local `ollama_chat/llama3.2:3b` from the previous chapter is just a different string.

**The conversation history** is a Python list of dictionaries, each with a `role` and `content`. The system message goes in first to set the ground rules. The opening scene is generated by sending `"Start the adventure."` as the first user message. After that, every player input and model response is appended to the list.

**The game loop** is a `while True` loop that reads input, checks for commands (lines starting with `/`), sends non-command input to the model, and prints the response. Slash commands let the player quit, restart, or get help without those strings being sent to the model as game actions.

**Restarting** works by resetting the message list to just the system prompt and a fresh `"Start the adventure."` so the model generates a completely new opening scene and the adventure begins again.

**Error handling** wraps the API calls in try/except blocks that catch `litellm.exceptions.OpenAIError`. That is litellm's base class for provider failures — `AuthenticationError`, `RateLimitError`, `APIConnectionError` and the rest all derive from it — so one clause covers both a missing API key and a provider outage. If a request fails, we print the error and give the player a chance to continue rather than crashing out of the game. Catch the finer-grained subclasses instead when you want to react differently to a bad key or a throttled request.

## Playing the Game

Here is an example session to give you a feel for how the game plays:

```
$ uv run game.py 
============================================================
  TEXT ADVENTURE GAME
  Powered by Fireworks.ai — deepseek-v4p1-flash
============================================================

Commands: /help  /restart  /quit

You stand at the mouth of a narrow pass, its walls of jagged stone rising high on either side. A cold mist curls around your boots, and ancient carvings-weathered swirls and symbols that cover the cliff faces. Somewhere ahead, a faint, rhythmic sound echoes, like water dripping into stone. The air smells of damp earth and old moss.

Where do you begin?

**Option 1:** Enter the pass cautiously, following the sound of dripping water.
**Option 2:** Study the carvings on the cliffs, hoping they reveal a clue.
**Option 3:** Unfold your map and search for a marked path or warning.
**Option 4:** Call out into the mist, testing if anything answers.

What do you do?

> what do the carvings on the cliff look like?

You step closer, squinting through the mist. The carvings spiral and weave like the roots of an immense tree. Among the abstract patterns, you make out recurring shapes: a curled serpent, a crescent moon, and a single eye with a vertical pupil. Some sections seem to glow faintly when you brush away moss, as if faintly enchanted. Beneath the largest eye symbol, a line of smaller marks might be letters or a warning but the script is unfamiliar.

**Option 1:** Trace the glowing symbols with your finger, hoping to activate something.
**Option 2:** Copy the strange script into your journal to decipher later.
**Option 3:** Move on entering the pass toward the dripping sound.
**Option 4:** Press your ear against the stone and listen for anything hidden.

What do you do?

> copy the strange script in journal and move towards the dripping sound

You quickly scratch the strange script into your journal, the charcoal smudging slightly as you work. The symbols feel oddly familiar in your hand, as if your fingers knew the shapes before your eyes did. You tuck the journal away and step deeper into the pass.

The mist thins as you walk, revealing a winding stone corridor that opens into a small, circular clearing. In the center, a weathered stone well stands, its rim draped in pale moss. The dripping sound is louder here, it comes from within the well, though no water spills over the edge. A faint silver light pulses from the depth.

Near the well, a small, rusted iron bell hangs from a crooked branch. There is no rope, no clapper, just the bell, motionless.

**Option 1:** Peer into the well to see what glows below.
**Option 2:** Ring the bell and see what happens.
**Option 3:** Search the edges of the clearing for another path or clue.
**Option 4:** Drop one of your gold coins into the well as an offering.

What do you do?

> /quit
Thanks for playing!
```

Every time you play the game the example program generates a unique story. The model remembers your choices; if you befriend an NPC early on, they may reappear later. If you lose your map crossing a river, the model won't let you consult it later without some creative problem-solving.

## Customizing Your Adventure

The best part of this architecture is how easy it is to create entirely different games. Just edit `story.txt`. Here are a few ideas:

**Sci-fi thriller:** "You are the security officer aboard the starship *Event Horizon*. A distress signal has been received from a derelict research station. Strange energy readings suggest something is still alive on board..."

**Noir detective:** "You are a private eye in 1940s Chicago. A dame in a red dress just walked into your office with a case that smells wrong. The city is baking under a heat wave and your ceiling fan is on its last legs..."

**Fantasy tavern:** "You are the innkeeper of the Grumpy Griffin, the rowdiest tavern in the realm. Tonight a hooded stranger sits in the corner, a bar brawl is brewing, and the king's guards just kicked in your door..."

The model adapts to whatever world you define. The numbered rules keep the game structure consistent; the setting paragraph gives the model everything it needs to paint a scene.

{class: tip}
For the best results, keep your system prompt between 200 and 500 words. Too short and the model won't have enough context to maintain a consistent world. Too long and it may struggle to follow all the rules at once.

## Why This Matters

Beyond being fun, this project demonstrates a pattern you will use in many real-world LLM applications: maintaining a conversation history where the system prompt defines behavior, user messages carry input, and assistant messages carry responses. This same structure powers customer support chatbots, interactive tutorials, code assistants, and creative writing tools.

The key insight is that the LLM's context window serves as both its *memory* and its *state*. There is no separate database tracking the player's inventory or location so the model tracks everything implicitly in the conversation. For a game, this works beautifully. For production applications with millions of users, you would eventually want to store state externally and summarize older conversation turns to manage context window limits. But for a personal project or prototype, the pure conversation-history approach is remarkably capable.

## Running the Example

Set your Fireworks API key and run the game:

```bash
export FIREWORKS_API_KEY="your-api-key"
uv sync
uv run python game.py
```

`uv sync` installs the project's dependencies, including litellm from PyPI, into the project's virtual environment. Then lose yourself in the ancient valley for a while. The dagger and three gold coins won't spend themselves.

## Summary

You have now built an AI-powered text adventure game. The complete program is under 150 lines of Python, yet it can generate infinite stories across any setting you can describe. The techniques you used like system prompt design, conversation history management, and the game loop pattern all transfer directly to chatbots, interactive fiction, and any application where an LLM needs to maintain state across multiple turns of conversation.

Try swapping in different models, experimenting with the system prompt, or adding features like a save/load system that persists the message history to a JSON file. The framework is simple enough that you can extend it in an afternoon, and the results are genuinely entertaining.
