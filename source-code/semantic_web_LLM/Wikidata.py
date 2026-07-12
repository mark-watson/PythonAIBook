# /usr/bin/env python3
"""
Wikidata.py - Semantic Web QA using the Wikidata knowledge base.

Answers natural-language questions by:
1. Using a Fireworks.ai LLM to extract named entities
2. Detecting relationship queries and querying Wikidata properties directly
3. Retrieving entity descriptions and structured facts via SPARQL
4. Synthesizing a natural-language answer with the LLM

Wikidata uses a different data model from DBpedia: entities are identified
by Q-numbers (e.g., Q2283 for Microsoft) and properties by P-numbers
(e.g., P159 for headquarters location). The Wikidata query service also
requires a descriptive User-Agent header and enforces stricter rate
limits, so the SPARQL queries here use the Wikidata label service to
resolve Q-numbers to human-readable labels in a single round trip.

Usage:
    uv run Wikidata.py "tell me about the following: IBM, Microsoft"

Set environment variable:
    export FIREWORKS_API_KEY="your-api-key"
"""

import time

from library import (
    ENTITY_TYPES,
    extract_entities,
    synthesize_answer,
    query_sparql,
    detect_relationship,
    resolve_value,
    run_cli,
)


# ---------------------------------------------------------------------------
# Wikidata endpoint
# ---------------------------------------------------------------------------

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

# Wikidata enforces strict rate limits (often 1 request / minute on the
# public endpoint). We insert a short delay between SPARQL calls to avoid
# HTTP 429 responses.
SPARQL_DELAY = 1.0  # seconds between queries


def _query_wikidata(sparql_query: str) -> list[dict]:
    """Execute a SPARQL query against Wikidata with rate-limit courtesy."""
    results = query_sparql(WIKIDATA_ENDPOINT, sparql_query)
    time.sleep(SPARQL_DELAY)
    return results


# ---------------------------------------------------------------------------
# SPARQL templates (Wikidata-specific)
# ---------------------------------------------------------------------------

# Entity lookup by English label. Returns the Q-number entity and its short
# Wikidata description. We sort by Q-number (ascending) so the most
# prominent entity (lowest Q-number) comes first, filtering out
# disambiguation pages.
SPARQL_LOOKUP_TEMPLATE = """
SELECT ?item ?itemLabel ?description WHERE {{
  ?item rdfs:label '{name}'@en .
  ?item schema:description ?description .
  FILTER(LANG(?description) = 'en') .
  FILTER(CONTAINS(STR(?item), '/entity/Q')) .
  BIND(REPLACE(STR(?item), '.*Q', '') AS ?qnum) .
  OPTIONAL {{ ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel) = 'en') . }}
}}
ORDER BY xsd:integer(?qnum)
LIMIT 5
"""

# Relationship lookup: given an entity Q-number and a Wikidata property
# (e.g. wdt:P36 for capital), fetch the related entity's English label.
SPARQL_RELATIONSHIP_TEMPLATE = """
SELECT DISTINCT ?obj ?objLabel WHERE {{
  wd:{qid} wdt:{pid} ?obj .
  ?obj rdfs:label ?objLabel .
  FILTER(LANG(?objLabel) = 'en') .
}}
LIMIT 10
"""

# Enrichment: given an entity Q-number, fetch a curated set of property
# values with human-readable labels. Uses the Wikidata meta-model: each
# direct claim property (wdt:Pxxx) has a corresponding property entity
# (wd:Pxxx) whose rdfs:label gives us the property name. Object values
# that are themselves entities are resolved via rdfs:label.
SPARQL_ENRICHMENT_TEMPLATE = """
SELECT DISTINCT ?propLabel ?obj ?objLabel WHERE {{
  VALUES (?p ?propEntity) {{
    {values}
  }}
  wd:{qid} ?p ?obj .
  ?propEntity rdfs:label ?propLabel .
  FILTER(LANG(?propLabel) = 'en') .
  OPTIONAL {{
    ?obj rdfs:label ?objLabel .
    FILTER(LANG(?objLabel) = 'en') .
  }}
}}
LIMIT 80
"""


# ---------------------------------------------------------------------------
# Relationship property mapping (English phrase -> Wikidata property ID)
# ---------------------------------------------------------------------------

# Wikidata properties use P-numbers. Each entry maps an English phrase to
# (property_id_without_prefix, human_readable_verb). The SPARQL templates
# prepend "wdt:" to the property ID at query time.
RELATIONSHIP_PROPERTIES = {
    "capital": ("P36", "capital"),
    "capital of": ("P36", "capital"),
    "birthplace": ("P19", "birthplace"),
    "born": ("P19", "birthplace"),
    "born in": ("P19", "birthplace"),
    "deathplace": ("P20", "deathplace"),
    "died": ("P20", "deathplace"),
    "died in": ("P20", "deathplace"),
    "spouse": ("P26", "spouse"),
    "married to": ("P26", "spouse"),
    "founded by": ("P112", "founder"),
    "founded": ("P112", "founder"),
    "founder": ("P112", "founder"),
    "who founded": ("P112", "founder"),
    "industry": ("P452", "industry"),
    "headquartered": ("P159", "headquarters location"),
    "headquarters": ("P159", "headquarters location"),
    "country": ("P17", "country"),
    "population": ("P1082", "population"),
    "currency": ("P38", "currency"),
    "language": ("P37", "official language"),
    "official language": ("P37", "official language"),
    "area": ("P2046", "area"),
    "subsidiary": ("P355", "subsidiary"),
    "chairperson": ("P488", "chairperson"),
    "ceo": ("P169", "CEO"),
}


# ---------------------------------------------------------------------------
# Entity type -> Wikidata "instance of" property (P31) values
# ---------------------------------------------------------------------------

# Wikidata does not have a single "type" ontology like DBpedia. Instead,
# entities are linked to classes via the P31 (instance of) property. We
# use these to constrain the lookup query so "Washington" the person is
# distinguished from "Washington" the state. The Q-numbers here are:
#   Q5       = human (person)
#   Q43229   = organization
#   Q6256    = country
#   Q515     = city
#   Q56061   = state of the United States
# We keep the mapping simple: GPE covers countries, cities, and regions.
ENTITY_TYPE_TO_WIKIDATA_CLASS = {
    "PERSON": "Q5",
    "ORG": "Q43229",
    "GPE": "",  # GPE is too broad for a single class; rely on label match
    "MISC": "",
}


# ---------------------------------------------------------------------------
# Enrichment properties per entity type (Wikidata P-numbers)
# ---------------------------------------------------------------------------

# Each entry maps a human-readable label to a (property_id, property_entity)
# pair. The property_id is the direct-claim prefix (wdt:Pxxx), and
# property_entity is the property item (wd:Pxxx) whose rdfs:label gives us
# the human-readable property name.
ENRICHMENT_PROPERTIES = {
    "ORG": {
        "inception": "P571",
        "founded by": "P112",
        "industry": "P452",
        "headquarters location": "P159",
        "country": "P17",
        "chairperson": "P488",
        "CEO": "P169",
        "subsidiary": "P355",
        "product or material produced": "P1056",
        "net profit": "P2295",
    },
    "PERSON": {
        "date of birth": "P569",
        "place of birth": "P19",
        "date of death": "P570",
        "place of death": "P20",
        "occupation": "P106",
        "country of citizenship": "P27",
        "spouse": "P26",
        "educated at": "P69",
        "employer": "P108",
        "award received": "P166",
        "member of": "P463",
    },
    "GPE": {
        "population": "P1082",
        "area": "P2046",
        "country": "P17",
        "capital": "P36",
        "head of state": "P35",
        "head of government": "P6",
        "official language": "P37",
        "currency": "P38",
    },
    "MISC": {},
}


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def lookup_entity_qid(name: str) -> str | None:
    """Look up the Wikidata Q-number for an entity by its English label.

    Returns the Q-number (e.g. "Q2283") or None if not found. Disambiguation
    pages are excluded by filtering on the presence of schema:description.
    Results are ordered by ascending Q-number so the most prominent entity
    (typically the one with the lowest Q-number) is returned first.
    """
    query = SPARQL_LOOKUP_TEMPLATE.format(name=name.replace("'", "\\'"))
    try:
        results = _query_wikidata(query)
    except Exception as e:
        print(f"[Warning] Wikidata lookup failed for {name}: {e}")
        return None

    for r in results:
        item_uri = r.get("item", {}).get("value", "")
        desc = r.get("description", {}).get("value", "")
        # Skip disambiguation pages
        if "disambiguation" in desc.lower():
            continue
        # Extract Q-number from the URI
        qid = item_uri.rsplit("/", 1)[-1]
        if qid.startswith("Q"):
            return qid
    return None


def query_relationship(qid: str, pid: str) -> list[dict]:
    """Query Wikidata for the object of a relationship property.

    *qid* is the entity Q-number (e.g. "Q142") and *pid* is the property
    ID without the wdt: prefix (e.g. "P36" for capital).
    """
    query = SPARQL_RELATIONSHIP_TEMPLATE.format(qid=qid, pid=pid)
    try:
        results = _query_wikidata(query)
    except Exception as e:
        print(f"[Warning] Wikidata relationship query failed: {e}")
        return []

    out = []
    for r in results:
        entry = {"label": r.get("objLabel", {}).get("value", "")}
        out.append(entry)
    return out


def enrich_entity(qid: str, entity_type: str) -> list[str]:
    """Fetch curated property/value facts for a Wikidata entity.

    Returns a list of "propLabel: value" strings. Entity-valued objects
    are resolved to their English rdfs:label when available.
    """
    props = ENRICHMENT_PROPERTIES.get(entity_type, {})
    if not props:
        return []

    # Build the VALUES block: pairs of (wdt:Pxxx, wd:Pxxx)
    values_lines = []
    for label, pid in props.items():
        values_lines.append(f"(wdt:{pid} wd:{pid})")
    values_block = "\n    ".join(values_lines)

    query = SPARQL_ENRICHMENT_TEMPLATE.format(qid=qid, values=values_block)
    try:
        results = _query_wikidata(query)
    except Exception as e:
        print(f"[Warning] Wikidata enrichment failed for {qid}: {e}")
        return []

    facts = []
    seen = set()
    for r in results:
        label = r.get("propLabel", {}).get("value", "")
        obj = r.get("obj", {}).get("value", "")
        obj_label = r.get("objLabel", {}).get("value")
        value = resolve_value(obj, obj_label)
        if not value:
            continue
        key = (label, value)
        if key in seen:
            continue
        seen.add(key)
        facts.append(f"{label}: {value}")
    return facts


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------


def get_entity_context(entities: dict) -> str:
    """Execute SPARQL queries for extracted entities and build context text.

    For each entity, looks up the Wikidata Q-number, fetches the short
    description, and runs an enrichment query for structured facts.
    Returns a structured context string with one paragraph per entity.
    """
    context_parts = []

    for entity_type in ENTITY_TYPES:
        if entity_type not in entities:
            continue

        for name in entities[entity_type]:
            qid = lookup_entity_qid(name)
            if not qid:
                print(f"[Warning] No Wikidata Q-number found for {name}")
                continue

            print(f"[DEBUG] {name} -> Wikidata {qid}")

            # Fetch the short description
            desc = ""
            try:
                query = SPARQL_LOOKUP_TEMPLATE.format(name=name.replace("'", "\\'"))
                results = _query_wikidata(query)
                for r in results:
                    if r.get("item", {}).get("value", "").endswith(qid):
                        desc = r.get("description", {}).get("value", "")
                        break
            except Exception as e:
                print(f"[Warning] Description fetch failed for {name}: {e}")

            # Fetch enrichment facts
            facts = enrich_entity(qid, entity_type)

            if desc and facts:
                context_parts.append(f"{name} ({qid}): {desc}\n" + "\n".join(facts))
            elif desc:
                context_parts.append(f"{name} ({qid}): {desc}")
            elif facts:
                context_parts.append(f"{name} ({qid}):\n" + "\n".join(facts))

    return "\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Answer pipeline
# ---------------------------------------------------------------------------


def answer_question(question: str) -> tuple[str, str]:
    """Answer a question using Wikidata + Fireworks.ai.

    Returns (answer_text, context_used).
    """
    entities = extract_entities(question)
    print(f"[DEBUG] Entities found: {entities}")

    # Relationship detection -> direct Wikidata property lookup
    relationship = detect_relationship(question, RELATIONSHIP_PROPERTIES)
    if relationship:
        pid, verb = relationship
        print(f"[DEBUG] Relationship detected: {verb} (P{pid})")
        all_names = [n for names in entities.values() for n in names]
        if not all_names:
            all_names = [w for w in question.replace("?", "").split() if w[0].isupper()]
        for name in all_names:
            qid = lookup_entity_qid(name)
            if not qid:
                continue
            rel_results = query_relationship(qid, pid)
            if rel_results:
                parts = [r["label"] for r in rel_results[:5] if r["label"]]
                if parts:
                    answer = f"The {verb} of {name} is: {', '.join(parts)}."
                    return answer, "\n".join(parts)

    # General entity lookup + enrichment
    context = get_entity_context(entities)
    print(f"[DEBUG] Context text length: {len(context)} chars")

    if not context.strip():
        return "I couldn't find relevant entities in the knowledge base.", ""

    # LLM synthesis
    answer = synthesize_answer(question, context, source_label="Wikidata")
    if answer:
        return answer, context

    return "Unable to generate an answer.", context


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_cli(answer_question, "Wikidata.py")
