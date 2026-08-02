# travel_planner_agent.py - Trip Planner built with NVIDIA OO Agents (nooa)
#
# This example demonstrates the key features of NVIDIA's Object Oriented Agents
# framework by building a small but realistic Travel Planner agent:
#
#   1. `Agent` subclass with a docstring that becomes the system prompt.
#   2. Ordinary Python methods that act as deterministic "tools" the LLM
#      may call while it works (cost math, mock DB lookups).
#   3. LLM-driven generation methods whose bodies are just `...` - the
#      framework fills them in at runtime from the signature + docstring.
#   4. Structured output via Pydantic.
#   5. An async orchestrator method that stitches everything together.
#   6. A bonus direct call to the sibling `NVIDIA_client.complete()` helper,
#      showing the framework composes cleanly with plain HTTP calls.
#
# The LLM backend is NVIDIA's free OpenAI-compatible NIM endpoint, reached by
# passing `_BASE_URL` and `DEFAULT_MODEL` from `../llm_public_apis/NVIDIA_client.py`
# through litellm's `openai/<model>` prefix.
#
# Hack it: edit DESTINATIONS to add cities of your own, change `interests` and
# `budget` in `main()`, or add new deterministic tool methods and watch the
# LLM start calling them.

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "nooa @ git+https://github.com/NVIDIA-NeMo/labs-OO-Agents",
#   "openai>=1.0",
#   "pydantic>=2",
# ]
# ///

import asyncio
import os
import sys
from pathlib import Path

from pydantic import BaseModel, Field

# Reuse endpoint + model from the sibling NVIDIA_client demo
_SIBLING = Path(__file__).resolve().parent.parent / "llm_public_apis"
sys.path.insert(0, str(_SIBLING))
from NVIDIA_client import DEFAULT_MODEL, _BASE_URL, complete  # noqa: E402

from nooa import Agent  # noqa: E402
from nooa.unifiedllm.registry import get_llm_client  # noqa: E402


if not os.getenv("NVIDIA_API_KEY"):
    raise SystemExit("Set NVIDIA_API_KEY first (free key at https://build.nvidia.com)")

# litellm's "openai/" prefix means "any OpenAI-compatible endpoint", so
# combining it with NVIDIA_client's constants routes every call to NVIDIA NIM.
llm = get_llm_client(
    f"openai/{DEFAULT_MODEL}",
    api_base=_BASE_URL,
    api_key=os.getenv("NVIDIA_API_KEY"),
)


# ── Mock "destination database" - edit freely to add your own cities ──────
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

    # ── Deterministic "tool" methods (regular Python, no `...`) ──────────
    #    The LLM can call any of these on `self` while it works.

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
        return float(d["flight"]) + float(d["hotel_night"]) * nights  # pyright: ignore[reportArgumentType]

    def cheapest_within(self, budget: float, nights: int) -> list[str]:
        """Cities whose total (flight + nights * hotel) is <= budget."""
        return [c for c in DESTINATIONS if self.estimate_cost(c, nights) <= budget]

    # ── LLM-driven generation methods (body is `...`) ────────────────────
    #    Signature + docstring become the contract; the framework generates
    #    the implementation at call time using the configured `llm`.

    async def recommend_destination(
        self, interests: str, budget: float, nights: int
    ) -> str:
        """Pick the SINGLE best city for `interests` within `budget` for
        `nights` nights. First call `cheapest_within` to filter, then
        compare `get_vibe` for each candidate. Return only the city name -
        no punctuation, no explanation."""
        ...

    async def draft_itinerary(self, city: str, nights: int) -> list[DayPlan]:
        """Produce a `nights + 1` day itinerary for `city`. Use `get_vibe`
        for local flavor. Number days starting at 1. Each day should have
        a theme plus morning/afternoon/evening activities."""
        ...

    async def packing_tip(self, city: str) -> str:
        """One vivid sentence of packing advice tailored to `city`."""
        ...

    # ── Orchestrator: plain async Python that composes the above ─────────

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
        itinerary = Itinerary(days=await self.draft_itinerary(city, nights))
        tip = await self.packing_tip(city)

        # Bonus flourish: one direct sync call to the sibling NVIDIA_client.
        # Demonstrates that the OO agent framework composes cleanly with
        # plain HTTP-level LLM calls hitting the same endpoint.
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
