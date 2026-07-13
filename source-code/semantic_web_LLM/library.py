# /usr/bin/env python3
"""
library.py - Shared utilities for SPARQL + LLM question answering.

This module provides general-purpose functions used by the knowledge-base
specific scripts (DBPedia.py, Wikidata.py, DBPedia_and_Wikidata.py):

  * LLM utilities: Fireworks.ai client, entity extraction, answer synthesis
  * SPARQL utilities: generic query execution, relationship detection
  * CLI helper: a reusable main() that delegates to a caller-supplied
    answer_question function

Set environment variable:
    export FIREWORKS_API_KEY="your-api-key"
"""

import os
import json
import re
from collections.abc import Callable

import requests
from openai import OpenAI


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.getenv("FIREWORKS_API_KEY"),
)

MODEL_ID = "accounts/fireworks/models/deepseek-v4-flash"

# Descriptive User-Agent so SPARQL endpoints (especially Wikidata) do not
# rate-limit us as an unidentified bot.
_USER_AGENT = (
    "PythonAIBook/1.0 (educational project; +https://github.com/markw/PythonAIBook)"
)

# Canonical entity-type ordering used by all KB-specific scripts.
ENTITY_TYPES = ["PERSON", "ORG", "GPE", "MISC"]


def llm_complete(prompt: str, max_tokens: int = 3000, temperature: float = 0) -> str:
    """Send a single user message to the Fireworks.ai LLM and return the text.

    Thin wrapper around the OpenAI-compatible chat-completions API so callers
    do not have to repeat the boilerplate client/model/messages dance.
    """
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract named entities from text using the Fireworks.ai LLM.

    Uses a one-shot prompt so the model sees an example input/output pair
    before classifying the user's text. Returns a dict mapping entity types
    (PERSON, ORG, GPE, MISC) to lists of entity names.
    """
    one_shot_prompt = """You are an expert named-entity recognition system.

Identify the named entities in the user's text and classify each one into exactly one of these types:
- PERSON  (people: individuals, fictional characters)
- ORG     (organizations: companies, bands, teams, government bodies)
- GPE     (geo-political entities: countries, states, cities, regions)
- MISC    (anything else worth looking up: scientific fields, products, works of art, concepts)

Return ONLY a JSON object of the form {"TYPE": ["Name", ...]}. Omit types with no entities. No prose, no markdown fences.

# Example
Text: "Where does Bill Gates work and what is the population of Paris?"
Output: {"PERSON": ["Bill Gates"], "GPE": ["Paris"]}

# Now classify this text
Text: "{text}"
Output: """

    try:
        raw = llm_complete(
            one_shot_prompt.replace("{text}", text),
            max_tokens=3000,
            temperature=0,
        )
        # Strip markdown code fences if the model wrapped its output
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if fence_match:
            raw = fence_match.group(1).strip()
        entities = json.loads(raw)
        # Normalize: ensure values are lists of strings
        cleaned = {}
        for etype, names in entities.items():
            if isinstance(names, str):
                names = [names]
            cleaned[etype] = [str(n) for n in names]
        return cleaned
    except Exception as e:
        print(f"[Warning] LLM entity extraction failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Answer synthesis
# ---------------------------------------------------------------------------


def synthesize_answer(
    question: str, context: str, source_label: str = "the knowledge base"
) -> str:
    """Use the LLM to synthesize a natural-language answer from context.

    *question* is the user's original question.  *context* is the structured
    text retrieved from one or more SPARQL endpoints.  *source_label* is a
    human-readable name for the source (e.g. "DBpedia", "Wikidata",
    "DBpedia and Wikidata") so the prompt can name it.
    """
    short_context = context[:3000] if len(context) > 3000 else context
    prompt = (
        f'Given question: "{question}"\n\n'
        f"Knowledge from {source_label} about relevant entities:\n"
        f"{short_context}\n\n"
        "Please answer the question based on this knowledge. "
        "Be specific, address every entity the user asked about, and "
        "cite entity names where possible."
    )
    try:
        return llm_complete(prompt, max_tokens=3500)
    except Exception as e:
        print(f"[Warning] LLM synthesis failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# SPARQL utilities
# ---------------------------------------------------------------------------


def query_sparql(endpoint: str, sparql_query: str) -> list[dict[str, dict[str, str]]]:
    """Execute a SPARQL query against *endpoint* and return result bindings.

    Uses the ``requests`` library so we can set a descriptive User-Agent
    header, which Wikidata requires and DBpedia appreciates. Returns a list
    of dicts, one per result row, where keys are variable names and values
    are dicts with 'type' and 'value' keys (the standard SPARQL JSON
    Results format).
    """
    print(f"SPARQL query ({endpoint}):\n{sparql_query}")
    resp = requests.get(
        endpoint,
        params={"query": sparql_query, "format": "json"},
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/sparql-results+json",
        },
        timeout=60,
    )
    resp.raise_for_status()
    ret = resp.json()["results"]["bindings"]
    print(f"Results: {len(ret)} bindings\n")
    return ret


def detect_relationship(
    question: str, relationship_table: dict[str, tuple[str, str]]
) -> tuple[str, str] | None:
    """Detect a relationship query in *question* using *relationship_table*.

    Scans the lowercased question for any key in the table. Multi-word keys
    are checked first (sorted by descending length) so "capital of" wins over
    bare "capital". Returns (property_uri, verb) or None.
    """
    lower = question.lower()
    for key in sorted(relationship_table, key=len, reverse=True):
        if key in lower:
            uri, verb = relationship_table[key]
            return uri, verb
    return None


def resolve_value(obj: str, obj_label: str | None) -> str:
    """Convert a SPARQL object value into a human-readable string.

    If *obj_label* is present (e.g. resolved via rdfs:label or the Wikidata
    label service), use it. Otherwise, if *obj* is a URI, extract the last
    path segment and replace underscores with spaces. Otherwise return *obj*
    as-is (literals: dates, numbers, plain strings).
    """
    if obj_label:
        return obj_label
    if obj.startswith("http"):
        return obj.rsplit("/", 1)[-1].replace("_", " ")
    return obj


# ---------------------------------------------------------------------------
# Shared CLI
# ---------------------------------------------------------------------------


def run_cli(answer_fn: Callable[[str], tuple[str, str]], script_name: str):
    """Reusable command-line / interactive entry point.

    *answer_fn* is a callable(question: str) -> (answer: str, context: str).
    *script_name* is displayed in the interactive prompt header.
    """
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print(f"{script_name} - QA with SPARQL + LLM")
        print("-" * 50)
        question = input("Enter your question: ")

    print(f"\nQuestion: {question}")
    print("-" * 50)

    answer, context = answer_fn(question)

    print("\nAnswer:")
    print(answer)
