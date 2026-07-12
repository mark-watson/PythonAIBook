# /usr/bin/env python3
"""
DBPedia_and_Wikidata.py - Semantic Web QA combining two knowledge bases.

This script answers natural-language questions by querying **both** DBpedia
and Wikidata, then merging the retrieved facts into a single context that
the LLM synthesizes into an answer.  This demonstrates a key advantage of
the Semantic Web: because both knowledge bases use URIs and RDF, a client
application can federate queries across multiple endpoints and combine
the results.

Pipeline:
1. Fireworks.ai LLM extracts named entities from the question
2. Both DBPedia.py and Wikidata.py retrieve context for those entities
3. The two context strings are merged (DBpedia first, Wikidata second)
4. The LLM synthesizes a single answer from the combined knowledge

Usage:
    uv run DBPedia_and_Wikidata.py "tell me about the following: IBM, Microsoft"

Set environment variable:
    export FIREWORKS_API_KEY="your-api-key"
"""

from library import (
    extract_entities,
    synthesize_answer,
    run_cli,
)

import DBPedia
import Wikidata


# ---------------------------------------------------------------------------
# Combined answer pipeline
# ---------------------------------------------------------------------------


def answer_question(question: str) -> tuple[str, str]:
    """Answer a question using both DBpedia and Wikidata.

    Queries both knowledge bases for the same set of entities, then merges
    the retrieved context and asks the LLM to synthesize an answer that
    draws on both sources.

    Returns (answer_text, combined_context).
    """
    entities = extract_entities(question)
    print(f"[DEBUG] Entities found: {entities}")

    # --- DBpedia context ---
    print("\n--- Querying DBpedia ---")
    dbpedia_context = DBPedia.get_entity_context(entities)
    print(f"[DEBUG] DBpedia context: {len(dbpedia_context)} chars")

    # --- Wikidata context ---
    print("\n--- Querying Wikidata ---")
    wikidata_context = Wikidata.get_entity_context(entities)
    print(f"[DEBUG] Wikidata context: {len(wikidata_context)} chars")

    # --- Merge contexts ---
    # Label each source so the LLM (and the reader) can tell where each
    # block of facts came from. DBpedia is queried first because it tends
    # to have richer descriptions; Wikidata adds structured P-properties
    # that DBpedia may not carry.
    parts = []
    if dbpedia_context.strip():
        parts.append("=== Knowledge from DBpedia ===\n" + dbpedia_context)
    if wikidata_context.strip():
        parts.append("=== Knowledge from Wikidata ===\n" + wikidata_context)

    combined_context = "\n\n".join(parts)

    if not combined_context.strip():
        return ("I couldn't find relevant entities in either knowledge base.", "")

    print(f"[DEBUG] Combined context: {len(combined_context)} chars")

    # --- LLM synthesis over the merged context ---
    answer = synthesize_answer(
        question,
        combined_context,
        source_label="DBpedia and Wikidata",
    )
    if answer:
        return answer, combined_context

    # --- Fallback: return raw combined context if LLM fails ---
    return combined_context, combined_context


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_cli(answer_question, "DBPedia_and_Wikidata.py")
