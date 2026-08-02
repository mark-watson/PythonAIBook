# Object Oriented AI Agents with NVIDIA's OO Agents Framework

Most agent frameworks in circulation today share the same basic shape. You write a system prompt as a string, you register a list of tools as JSON schemas, and you hand both to a runtime that decides when to call which tool. Everything about the agent is external metadata: the prompt is a text blob, the tools are dictionaries, the state lives in whatever data structures you happen to pass around, and the interface between the model and your Python code is a serialization boundary that you write by hand.

NVIDIA's Object Oriented Agents framework, packaged as `nooa`, takes a different approach. An agent is a Python class. Its system prompt is the class docstring. Each of its capabilities is a method. Some of those methods are ordinary deterministic Python and behave as tools the language model may call. Other methods have `...` as their body, and the framework fills in the implementation at runtime by prompting the language model with the method's signature, docstring, and return type. State is just class fields with type annotations. There is no separate tool schema, no manual JSON glue, and no drift between what the prompt promises and what the code enforces.

This chapter builds a complete example on top of that idea: a travel planner agent that recommends a destination, prices a trip against a budget, drafts a day by day itinerary, offers packing advice, and produces a structured trip plan. It runs on NVIDIA's free NIM inference endpoint using the same `NVIDIA_client.py` helper introduced in the previous chapter.

## Why represent an agent as a class

Consider the operations a travel planner needs to perform. Some are unambiguous and cheap:

* Look up the flight cost to a city.
* Multiply the hotel nightly rate by the number of nights.
* Filter destinations to those under a budget.

Others require judgment, taste, or synthesis:

* Pick the single best city for a traveler whose interests are "hiking, ancient history, and lively food markets".
* Draft a plausible seven day itinerary in Cusco.
* Write a memorable one sentence packing tip.

The first group belongs in Python. Nothing is gained, and much is lost, by asking a language model to multiply 180 by 6. The second group belongs to the language model, because the space of plausible answers is huge and no closed form procedure captures what "best" means.

In a traditional agent framework you would express this split by writing Python functions for the deterministic operations, wrapping them in JSON tool schemas, and writing prompts for the generative operations as separate strings. In `nooa` the split shows up directly in the class body. Deterministic operations are regular methods. Generative operations are async methods with `...` for a body. Both live on `self`, so the language model can call the deterministic tools while producing an answer for the generative ones. The type annotations on each method double as the interface contract the framework enforces on the model's output.

## What we will build

The finished program is a single file, `travel_planner_agent.py`, driven by a `Makefile`, with a small `README.md` for reference. The `plan_trip` method is the top level orchestrator. It calls one language model method to pick a destination, a second to draft an itinerary, a third for a packing tip, and finally one direct call to the low level `NVIDIA_client.complete` helper for a local greeting phrase. Along the way it uses the deterministic methods for cost math.

The eventual return value is a Pydantic `TravelPlan` with a numeric total, a boolean flag for whether the plan fits the budget, and a list of `DayPlan` records for each day of the trip.

## The mock destination database

The example ships with a hand written dictionary of five destinations. In a real application this would be a call to a booking API, but keeping it inline makes the example easy to run and easy to hack. Here is a representative entry so you can see the shape before you read the code that consumes it:

```python
"Cusco": {
    "country": "Peru",
    "flight": 900,
    "hotel_night": 90,
    "vibe": "Inca ruins, Andean markets, coca-tea culture",
}
```

Each destination has a country, a round trip flight cost in USD, a nightly hotel rate in USD, and a short "vibe" string that captures the city's character in a form the language model can reason over. Five entries is deliberately small; the goal is that a reader can add a sixth destination in ten seconds and immediately see the agent consider it.

## Structured output: the Pydantic models

Before the agent class itself, the file defines three Pydantic models. These are the return types the language model must produce. The framework parses model output against these classes, so any field that is missing, mistyped, or out of range triggers an automatic retry or a raised exception rather than silently corrupt data flowing forward.

```python
class DayPlan(BaseModel):
    day: int = Field(ge=1)
    theme: str
    morning: str
    afternoon: str
    evening: str


class Itinerary(BaseModel):
    days: list[DayPlan]


class TravelPlan(BaseModel):
    destination: str
    country: str
    total_cost_usd: float = Field(ge=0)
    duration_days: int = Field(ge=1)
    within_budget: bool
    days: list[DayPlan]
    packing_tip: str
    local_phrase: str
```

Notice `Itinerary` wraps a `list[DayPlan]`. This is a small defensive choice. Structured output from language models works most reliably when the top level type is a single object, so `draft_itinerary` returns `Itinerary` and the orchestrator unwraps `.days` before packing them into the final `TravelPlan`. The `Field(ge=1)` and `Field(ge=0)` constraints let Pydantic reject nonsensical outputs like a day zero or a negative cost.

## Configuring the language model

`nooa` uses the `litellm` library under the hood, so it can point at any OpenAI compatible endpoint by combining the `openai/<model>` prefix with an `api_base` and `api_key`. Reusing the constants from the previous chapter's `NVIDIA_client.py` file, the model client is:

```python
from NVIDIA_client import DEFAULT_MODEL, _BASE_URL, complete

from nooa import Agent
from nooa.unifiedllm.registry import get_llm_client

llm = get_llm_client(
    f"openai/{DEFAULT_MODEL}",
    api_base=_BASE_URL,
    api_key=os.getenv("NVIDIA_API_KEY"),
)
```

This `llm` object is then passed to the class definition itself with `class TravelPlannerAgent(Agent, llm=llm):`. Every generative method on the class will use this client. Because we are importing `_BASE_URL` and `DEFAULT_MODEL` from the earlier chapter's file, swapping models is a one line change in one place.

## The agent class, method by method

The class is short enough that we can walk through it in three passes: the docstring, the deterministic tools, and the generative methods.

### The docstring as a system prompt

```python
class TravelPlannerAgent(Agent, llm=llm):
    """You are a seasoned travel planner. You blend hard cost math from the
    deterministic helper methods with creative, culturally-aware itinerary
    design. Never invent prices - always call the helper methods for numeric
    facts. Prefer destinations whose vibe genuinely matches the traveller's
    stated interests."""
```

Two things worth noticing. First, the docstring is the system prompt. There is no separate `system_prompt=` argument. Second, the docstring names the helper methods by name and instructs the model to prefer them for numeric facts. This is how you steer the model toward using your deterministic tools rather than guessing.

### Deterministic tools

Five ordinary methods act as tools the language model can call from any generative step:

```python
def list_destinations(self) -> list[str]:
    """Every city you can plan a trip to."""
    return list(DESTINATIONS.keys())

def get_vibe(self, city: str) -> str:
    """One-line character description of the city."""
    return str(DESTINATIONS[city]["vibe"])

def get_country(self, city: str) -> str:
    """Country the city sits in."""
    return str(DESTINATIONS[city]["country"])

def estimate_cost(self, city: str, nights: int) -> float:
    """Round-trip flight + nights * hotel, in USD."""
    d = DESTINATIONS[city]
    return float(d["flight"]) + float(d["hotel_night"]) * nights

def cheapest_within(self, budget: float, nights: int) -> list[str]:
    """Cities whose total (flight + nights * hotel) is <= budget."""
    return [c for c in DESTINATIONS if self.estimate_cost(c, nights) <= budget]
```

These are plain Python. There is no `@tool` decorator, no schema declaration, no registration step. `nooa`'s runtime gives the language model a REPL-like environment with `self` bound, and the model calls these methods by writing Python such as `self.cheapest_within(2500, 6)`. Because the signatures and docstrings are already in the class, the model has everything it needs to pick the right call without additional glue.

### Generative methods

The three generative methods are declared the same way as normal Python methods, but their bodies are just `...`:

```python
async def recommend_destination(
    self, interests: str, budget: float, nights: int
) -> str:
    """Pick the SINGLE best city for `interests` within `budget` for
    `nights` nights. First call `cheapest_within` to filter, then
    compare `get_vibe` for each candidate. Return only the city name -
    no punctuation, no explanation."""
    ...

async def draft_itinerary(self, city: str, nights: int) -> Itinerary:
    """Produce a `nights + 1` day itinerary for `city`. Use `get_vibe`
    for local flavor. Number days starting at 1. Each day should have
    a theme plus morning/afternoon/evening activities."""
    ...

async def packing_tip(self, city: str) -> str:
    """One vivid sentence of packing advice tailored to `city`."""
    ...
```

At runtime, `nooa` intercepts every call to one of these methods, packages the class docstring, the method docstring, the method signature, and the return type into a prompt, sends that prompt to the configured `llm`, and validates the reply against the annotated return type. If the return type is a Pydantic model the reply is parsed into an instance. If validation fails the framework can retry.

The docstrings do more than describe intent. In `recommend_destination` the docstring names the two deterministic helper methods the model should call, so the model plans, filters, and compares before answering. In `draft_itinerary` the docstring establishes the shape of a day and the numbering convention.

### The orchestrator

The last method is regular async Python. It calls the generative methods, calls the deterministic methods, and calls the sibling `NVIDIA_client.complete` helper directly for a bonus local greeting phrase:

```python
async def plan_trip(
    self, interests: str, budget: float, nights: int
) -> TravelPlan:
    """End-to-end: choose city -> cost it -> itinerary -> packing tip
    -> a local greeting phrase (via a direct NVIDIA_client call)."""
    raw = (await self.recommend_destination(interests, budget, nights)).strip()
    city = raw.strip('"').splitlines()[0].strip()
    if city not in DESTINATIONS:
        affordable = self.cheapest_within(budget, nights)
        city = affordable[0] if affordable else next(iter(DESTINATIONS))

    cost = self.estimate_cost(city, nights)
    itinerary = await self.draft_itinerary(city, nights)
    tip = await self.packing_tip(city)

    phrase = complete(
        f"Give ONE short local greeting phrase a traveller could use in "
        f"{city}, {self.get_country(city)}, with a phonetic pronunciation "
        f"in parentheses. Reply with the phrase only - no preamble."
    ).strip()

    return TravelPlan(
        destination=city,
        country=self.get_country(city),
        total_cost_usd=cost,
        duration_days=nights + 1,
        within_budget=cost <= budget,
        days=itinerary.days,
        packing_tip=tip,
        local_phrase=phrase,
    )
```

Two design choices in this method deserve attention. First, even though `recommend_destination` is instructed to return only a city name, the orchestrator defensively strips quotes, takes the first line, and falls back to the cheapest affordable option if the reply is not a known city. Real language model output is noisy, and a wrapper of a few lines is far cheaper than a corrupted downstream stage. Second, the `complete` call at the end is deliberately synchronous. It shows that the object oriented framework composes cleanly with plain HTTP style calls to the same endpoint; you do not have to route everything through the agent to benefit from it.

## Complete file listing

The full `travel_planner_agent.py`:

```python
# travel_planner_agent.py - Trip Planner built with NVIDIA OO Agents (nooa)

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "nooa",
#   "openai>=1.0",
#   "pydantic>=2",
# ]
# ///

import asyncio
import os
import sys
from pathlib import Path

from pydantic import BaseModel, Field

_SIBLING = Path(__file__).resolve().parent.parent / "llm_public_apis"
sys.path.insert(0, str(_SIBLING))
from NVIDIA_client import DEFAULT_MODEL, _BASE_URL, complete  # noqa: E402

from nooa import Agent  # noqa: E402
from nooa.unifiedllm.registry import get_llm_client  # noqa: E402


if not os.getenv("NVIDIA_API_KEY"):
    raise SystemExit("Set NVIDIA_API_KEY first (free key at https://build.nvidia.com)")

llm = get_llm_client(
    f"openai/{DEFAULT_MODEL}",
    api_base=_BASE_URL,
    api_key=os.getenv("NVIDIA_API_KEY"),
)


DESTINATIONS: dict[str, dict[str, object]] = {
    "Kyoto":     {"country": "Japan",    "flight": 1400, "hotel_night": 180, "vibe": "zen temples, bamboo forests, matcha rituals"},
    "Reykjavik": {"country": "Iceland",  "flight":  650, "hotel_night": 220, "vibe": "geothermal lagoons, aurora borealis, glacier hikes"},
    "Cusco":     {"country": "Peru",     "flight":  900, "hotel_night":  90, "vibe": "Inca ruins, Andean markets, coca-tea culture"},
    "Lisbon":    {"country": "Portugal", "flight":  550, "hotel_night": 140, "vibe": "tile-clad hillsides, fado music, pastel de nata"},
    "Marrakech": {"country": "Morocco",  "flight":  700, "hotel_night": 110, "vibe": "labyrinth souks, Sahara excursions, mint-tea evenings"},
}


class DayPlan(BaseModel):
    day: int = Field(ge=1)
    theme: str
    morning: str
    afternoon: str
    evening: str


class Itinerary(BaseModel):
    days: list[DayPlan]


class TravelPlan(BaseModel):
    destination: str
    country: str
    total_cost_usd: float = Field(ge=0)
    duration_days: int = Field(ge=1)
    within_budget: bool
    days: list[DayPlan]
    packing_tip: str
    local_phrase: str


class TravelPlannerAgent(Agent, llm=llm):
    """You are a seasoned travel planner. You blend hard cost math from the
    deterministic helper methods with creative, culturally-aware itinerary
    design. Never invent prices - always call the helper methods for numeric
    facts. Prefer destinations whose vibe genuinely matches the traveller's
    stated interests."""

    def list_destinations(self) -> list[str]:
        """Every city you can plan a trip to."""
        return list(DESTINATIONS.keys())

    def get_vibe(self, city: str) -> str:
        """One-line character description of the city."""
        return str(DESTINATIONS[city]["vibe"])

    def get_country(self, city: str) -> str:
        """Country the city sits in."""
        return str(DESTINATIONS[city]["country"])

    def estimate_cost(self, city: str, nights: int) -> float:
        """Round-trip flight + nights * hotel, in USD."""
        d = DESTINATIONS[city]
        return float(d["flight"]) + float(d["hotel_night"]) * nights

    def cheapest_within(self, budget: float, nights: int) -> list[str]:
        """Cities whose total (flight + nights * hotel) is <= budget."""
        return [c for c in DESTINATIONS if self.estimate_cost(c, nights) <= budget]

    async def recommend_destination(
        self, interests: str, budget: float, nights: int
    ) -> str:
        """Pick the SINGLE best city for `interests` within `budget` for
        `nights` nights. First call `cheapest_within` to filter, then
        compare `get_vibe` for each candidate. Return only the city name."""
        ...

    async def draft_itinerary(self, city: str, nights: int) -> Itinerary:
        """Produce a `nights + 1` day itinerary for `city`. Use `get_vibe`
        for local flavor. Number days starting at 1."""
        ...

    async def packing_tip(self, city: str) -> str:
        """One vivid sentence of packing advice tailored to `city`."""
        ...

    async def plan_trip(
        self, interests: str, budget: float, nights: int
    ) -> TravelPlan:
        """End-to-end: choose city, cost it, itinerary, packing tip, phrase."""
        raw = (await self.recommend_destination(interests, budget, nights)).strip()
        city = raw.strip('"').splitlines()[0].strip()
        if city not in DESTINATIONS:
            affordable = self.cheapest_within(budget, nights)
            city = affordable[0] if affordable else next(iter(DESTINATIONS))

        cost = self.estimate_cost(city, nights)
        itinerary = await self.draft_itinerary(city, nights)
        tip = await self.packing_tip(city)

        phrase = complete(
            f"Give ONE short local greeting phrase a traveller could use in "
            f"{city}, {self.get_country(city)}, with a phonetic pronunciation "
            f"in parentheses. Reply with the phrase only - no preamble."
        ).strip()

        return TravelPlan(
            destination=city,
            country=self.get_country(city),
            total_cost_usd=cost,
            duration_days=nights + 1,
            within_budget=cost <= budget,
            days=itinerary.days,
            packing_tip=tip,
            local_phrase=phrase,
        )


async def main() -> None:
    agent = TravelPlannerAgent()
    plan = await agent.plan_trip(
        interests="hiking, ancient history, and lively food markets",
        budget=2500.0,
        nights=6,
    )
    print(plan.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

## The Makefile and PEP 723 script metadata

The block near the top of the file with `# /// script` is a PEP 723 header. It tells `uv` which Python version and which packages the script needs. Because of this header, `uv run travel_planner_agent.py` will create an ephemeral virtual environment on first run, install `nooa`, `openai`, and `pydantic` into it, and cache the environment for subsequent runs. No `pyproject.toml` is needed and no `uv sync` step is required.

The `Makefile` uses this feature directly:

```makefile
.PHONY: run clean

run:
	uv run travel_planner_agent.py

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
```

## Running the example

After exporting your NVIDIA API key and running `make run`, the output is a single JSON document. Every language model call is nondeterministic, so your exact wording will differ, but the structure and the numeric fields are stable. Here is a representative run:

```json
{
  "destination": "Cusco",
  "country": "Peru",
  "total_cost_usd": 1440.0,
  "duration_days": 7,
  "within_budget": true,
  "days": [
    {
      "day": 1,
      "theme": "Arrival and gentle acclimatization",
      "morning": "Land in Cusco and check in slowly; sip coca tea to settle the altitude.",
      "afternoon": "Amble through San Blas, ducking into artisan studios and Inca stonework alleys.",
      "evening": "Early quinoa-and-trout dinner near the Plaza de Armas."
    },
    {
      "day": 2,
      "theme": "Sacred Valley markets",
      "morning": "Ride to Pisac; wander the Sunday market stacked with alpaca weavings and roasted corn.",
      "afternoon": "Climb the Pisac ruins for a first taste of Inca terracing.",
      "evening": "Return to Cusco for pisco sours and lomo saltado."
    },
    {
      "day": 3,
      "theme": "Andean day hike",
      "morning": "Drive to the Rainbow Mountain trailhead and start the climb at first light.",
      "afternoon": "Descend slowly; picnic near a glacial stream.",
      "evening": "Recover with a hearty aji de gallina back in Cusco."
    },
    {
      "day": 4,
      "theme": "Ollantaytambo and the train",
      "morning": "Explore Ollantaytambo's fortress and its living Inca street grid.",
      "afternoon": "Board the afternoon train to Aguas Calientes at the foot of Machu Picchu.",
      "evening": "Riverside dinner and an early night before sunrise."
    },
    {
      "day": 5,
      "theme": "Machu Picchu",
      "morning": "First bus up the switchbacks; walk the classic circuit as mist lifts off the ruins.",
      "afternoon": "Hike to the Sun Gate for a wider view.",
      "evening": "Return train to Ollantaytambo, taxi back to Cusco."
    },
    {
      "day": 6,
      "theme": "Food markets and museums",
      "morning": "Graze the San Pedro market; try chicha morada and salteñas.",
      "afternoon": "Visit the Inka Museum for context on the ruins you have walked.",
      "evening": "Farewell dinner at a Novoandino tasting menu."
    },
    {
      "day": 7,
      "theme": "Departure",
      "morning": "Slow breakfast, last coca tea, final walk around the Plaza de Armas.",
      "afternoon": "Airport transfer.",
      "evening": "Fly home."
    }
  ],
  "packing_tip": "Layer merino for icy dawn ascents that surrender to strong Andean sun by noon.",
  "local_phrase": "Allillanchu (ah-lee-YAHN-choo)"
}
```

## Interpreting the output

Several things in this output are worth pausing on.

The `total_cost_usd` is `1440.0`, which is exactly `900 + 90 * 6`, matching the Cusco entry in `DESTINATIONS`. The number was computed by `estimate_cost`, not invented by the language model. This is the payoff of writing deterministic tool methods: numeric fields are always right, because they never leave Python.

The `within_budget` flag is `true`, because `1440.0 <= 2500.0`. Again, this is a plain Python comparison in the orchestrator, not a judgment call by the model. If the recommended city had come back over budget, this flag would be `false` and the caller could take corrective action.

The `destination` is Cusco. The traveler's stated interests were "hiking, ancient history, and lively food markets". Compare against the vibe strings for each destination: Kyoto is temples and matcha, Reykjavik is glaciers and lagoons, Lisbon is tiles and pastries, Marrakech is souks and Sahara excursions. Cusco is the only destination whose vibe hits all three of hiking, ancient history, and food markets at once. The model made a defensible pick using only the short vibe descriptions in the database, filtered against the budget by way of `cheapest_within`.

The itinerary is seven `DayPlan` objects, matching `nights + 1 = 7`. Every day has `theme`, `morning`, `afternoon`, and `evening` populated as strings. This is the Pydantic contract at work. If the model had returned a day with a missing field or an integer where a string was expected, `nooa` would have raised or retried, and the caller would never see a half formed record.

The `local_phrase` came from the direct `complete` call rather than from the agent. This is important because it shows that you can freely mix framework calls and low level calls. Both hit the same NVIDIA NIM endpoint using the same `NVIDIA_API_KEY`.

## Wrap Up

The design principle behind `nooa` is that the boundary between "code the developer wrote" and "code the model wrote" should be a method boundary, not a serialization boundary. Once you accept that framing, most of the machinery of traditional agent frameworks becomes unnecessary. There is no tool registry because methods are already registered by being on the class. There is no prompt file because docstrings are prompts. There is no output parser because return types are already annotated and Pydantic already knows how to validate against them.

The travel planner in this chapter has fewer than two hundred lines of code and demonstrates six framework capabilities: class based agent definition, docstring driven prompting, deterministic tool methods, language model generation methods, structured Pydantic output, and clean composition with plain HTTP calls to the same endpoint. Every one of those capabilities is expressed as ordinary Python. That is the point.

The example is deliberately small enough to hack. Add a destination and the recommender considers it on the next run. Add a helper method such as `weather_score(city, month) -> float` and any generative method that mentions it in its docstring can call it. Add a whole new generative method with a `...` body and a signature, and it works from the first call. The framework fades into the background and leaves you writing Python.

## Optional Practice Problems

The following exercises range from small edits to more ambitious extensions. All of them build on the code in this chapter.

1. **Add a destination.** Append a new entry to `DESTINATIONS`, for example Reykjavik replaced with Queenstown or Ushuaia, and change the traveler's interests to something that clearly favors your new city. Confirm from the JSON output that the model picks it.

2. **Add a deterministic tool method.** Write a `season_score(self, city: str, month: str) -> float` method that returns a number between zero and one based on any rules you like (for example, prefer Kyoto in April, Cusco in June, Marrakech in October). Update the docstring on `recommend_destination` to mention the new tool and rerun. Inspect the output to confirm the picks shift with the month.

3. **Add a new generation method.** Declare `async def budget_swap_suggestions(self, city: str, target_savings: float) -> list[str]:` with a `...` body and a docstring instructing the model to propose two or three concrete substitutions (for example, "swap the Andean train for a shared van, save around one hundred fifty USD"). Wire it into `plan_trip` and add the returned list to `TravelPlan`.

4. **Persist the plan to disk.** Extend the orchestrator to write the returned `TravelPlan` as JSON to a file whose name includes the destination and the current date. Confirm the file round trips through `TravelPlan.model_validate_json`.

5. **Swap the model.** Change the `DEFAULT_MODEL` constant in `../llm_public_apis/NVIDIA_client.py` to a different model available on NVIDIA NIM. Rerun and compare the tone and structure of the itineraries.

6. **Multi city trip.** Extend the agent with a `plan_multi_city_trip(interests, budget, nights, cities: int)` generative method that returns a list of `TravelPlan` objects, one per leg. You will need to think carefully about how to split the budget between legs; a deterministic helper method that allocates the budget proportionally to leg length is a good starting point.

7. **Better recovery from a bad recommendation.** The orchestrator's current fallback picks the first affordable city if the model returns something unrecognized. Improve this by writing a second attempt: prompt the model with the specific list of affordable cities and ask it to pick from that list. Only fall back to the arbitrary first pick if that second attempt also fails.

8. **Add a small evaluation harness.** Write a script that calls `plan_trip` with three different `(interests, budget, nights)` inputs, checks that each returned plan is within budget and has the expected number of days, and prints a pass/fail summary. This is a good foundation for regression testing your agent as you extend it.
