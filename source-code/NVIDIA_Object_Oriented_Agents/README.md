# NVIDIA Object Oriented Agents - Travel Planner Example

A single-file demonstration of the [NVIDIA-NeMo/labs-OO-Agents](https://github.com/NVIDIA-NeMo/labs-OO-Agents)
framework (`nooa`) built on top of NVIDIA's free NIM inference endpoint.

The agent recommends a destination, prices a trip against a budget, drafts a
day-by-day itinerary, suggests packing advice, and finishes with a local
greeting phrase - all from a single `plan_trip()` call.

## What this shows

| Feature | Where |
|---|---|
| `Agent` subclass, docstring as system prompt | `TravelPlannerAgent` |
| Deterministic Python "tool" methods on `self` | `list_destinations`, `estimate_cost`, `cheapest_within`, ... |
| LLM-driven generation methods (body = `...`) | `recommend_destination`, `draft_itinerary`, `packing_tip` |
| Structured output with Pydantic | `TravelPlan`, `Itinerary`, `DayPlan` |
| Async orchestration composing tools + LLM calls | `plan_trip` |
| Reusing the sibling `NVIDIA_client.py` HTTP helper | `complete(...)` call inside `plan_trip` |

## Setup

1. Get a free NVIDIA NIM key at <https://build.nvidia.com>.
2. Install [`uv`](https://docs.astral.sh/uv/) if you don't have it.
3. Export the key:

   ```bash
   export NVIDIA_API_KEY="your-key"
   ```

## Run

```bash
make run
```

The script uses a PEP 723 header, so `uv run` will install `nooa`,
`openai`, and `pydantic` into a cached ephemeral venv on the first run.
No `uv sync` step is required.

Sample output (truncated):

```json
{
  "destination": "Cusco",
  "country": "Peru",
  "total_cost_usd": 1440.0,
  "duration_days": 7,
  "within_budget": true,
  "days": [
    {"day": 1, "theme": "Acclimatize to the altitude", ...},
    ...
  ],
  "packing_tip": "Layer merino for icy mornings that yield to Andean sun.",
  "local_phrase": "Allillanchu (ah-lee-YAHN-choo)"
}
```

## Hack it

- **Add a city.** Append an entry to `DESTINATIONS` at the top of
  `travel_planner_agent.py` - the LLM will immediately consider it.
- **Add a new tool.** Write a regular Python method on `TravelPlannerAgent`
  (e.g. `weather_score(city, month) -> float`). The LLM sees it via
  `self.<name>` and can call it while planning.
- **Add a new LLM step.** Declare an `async def` with a docstring and a
  `...` body (e.g. `async def budget_swap_suggestions(...) -> list[str]:`).
  The framework fills in the implementation from the signature.
- **Swap the model.** Change `DEFAULT_MODEL` in
  `../llm_public_apis/NVIDIA_client.py`, or import a different constant
  and pass it to `get_llm_client(...)`.

## Requirements

- Python 3.10+
- `uv` (for the PEP 723 script runner)
- `NVIDIA_API_KEY` environment variable
- The sibling file `../llm_public_apis/NVIDIA_client.py` (imported at runtime)

## Files

```
NVIDIA_Object_Oriented_Agents/
├── travel_planner_agent.py   # the example (PEP 723 inline deps)
├── Makefile                  # `make run`, `make clean`
└── README.md                 # this file
```
