# /usr/bin/env python3
"""
DBPedia.py - Semantic Web QA using the DBpedia knowledge base.

Answers natural-language questions by:
1. Using a Fireworks.ai LLM to extract named entities
2. Detecting relationship queries and querying DBpedia properties directly
3. Retrieving entity descriptions and structured facts via SPARQL
4. Synthesizing a natural-language answer with the LLM

Usage:
    uv run DBPedia.py "tell me about the following: IBM, Microsoft"

Set environment variable:
    export FIREWORKS_API_KEY="your-api-key"
"""

from library import (
    client, MODEL_ID, ENTITY_TYPES,
    llm_complete, extract_entities, synthesize_answer,
    query_sparql, detect_relationship, resolve_value, run_cli,
)


# ---------------------------------------------------------------------------
# DBpedia endpoint
# ---------------------------------------------------------------------------

DBPEDIA_ENDPOINT = "http://dbpedia.org/sparql"


# ---------------------------------------------------------------------------
# SPARQL templates (DBpedia-specific)
# ---------------------------------------------------------------------------

# Entity lookup by English label. Tries multiple descriptive predicates and
# filters out Category: resources. The rdf:type constraint is OPTIONAL so
# entities whose LLM-tagged type does not exactly match the DBpedia ontology
# still resolve.
SPARQL_QUERY_TEMPLATE = """
SELECT DISTINCT ?s ?p ?comment WHERE {{
  ?s <http://www.w3.org/2000/01/rdf-schema#label> '{name}'@en .
  FILTER(!CONTAINS(STR(?s), 'dbpedia.org/resource/Category:')) .
  ?s ?p ?comment .
  FILTER (lang(?comment) = 'en') .
  VALUES ?p {{
    <http://dbpedia.org/ontology/abstract>
    <http://www.w3.org/2000/01/rdf-schema#comment>
    <http://dbpedia.org/ontology/description>
  }}
  {dbpedia_type}
}} ORDER BY ?p LIMIT 15
"""

# Relationship lookup: given an entity label and a DBpedia property URI,
# fetch the related entity's label and a short description.
SPARQL_RELATIONSHIP_TEMPLATE = """
SELECT DISTINCT ?obj ?objLabel ?objComment WHERE {{
  ?s <http://www.w3.org/2000/01/rdf-schema#label> '{name}'@en .
  ?s <{property_uri}> ?obj .
  FILTER (isURI(?obj)) .
  ?obj <http://www.w3.org/2000/01/rdf-schema#label> ?objLabel .
  FILTER (lang(?objLabel) = 'en') .
  OPTIONAL {{
    ?obj ?pc ?objComment .
    FILTER (lang(?objComment) = 'en') .
    VALUES ?pc {{
      <http://dbpedia.org/ontology/description>
      <http://www.w3.org/2000/01/rdf-schema#comment>
      <http://dbpedia.org/ontology/abstract>
    }}
  }}
}} LIMIT 10
"""

# Enrichment: fetch a curated set of property/value pairs in one query.
# URI-valued objects are resolved to their English rdfs:label.
SPARQL_ENRICHMENT_TEMPLATE = """
SELECT DISTINCT ?propLabel ?obj ?objLabel WHERE {{
  ?s <http://www.w3.org/2000/01/rdf-schema#label> '{name}'@en .
  VALUES (?prop ?propLabel) {{
    {values}
  }}
  ?s ?prop ?obj .
  OPTIONAL {{
    ?obj <http://www.w3.org/2000/01/rdf-schema#label> ?objLabel .
    FILTER(lang(?objLabel) = 'en')
  }}
}} LIMIT 80
"""


# ---------------------------------------------------------------------------
# Relationship property mapping (English phrase -> DBpedia ontology URI)
# ---------------------------------------------------------------------------

RELATIONSHIP_PROPERTIES = {
    "capital": ("http://dbpedia.org/ontology/capital", "capital"),
    "capital of": ("http://dbpedia.org/ontology/capital", "capital"),
    "birthplace": ("http://dbpedia.org/ontology/birthPlace", "birthplace"),
    "born": ("http://dbpedia.org/ontology/birthPlace", "birthplace"),
    "born in": ("http://dbpedia.org/ontology/birthPlace", "birthplace"),
    "deathplace": ("http://dbpedia.org/ontology/deathPlace", "deathplace"),
    "died": ("http://dbpedia.org/ontology/deathPlace", "deathplace"),
    "died in": ("http://dbpedia.org/ontology/deathPlace", "deathplace"),
    "spouse": ("http://dbpedia.org/ontology/spouse", "spouse"),
    "married to": ("http://dbpedia.org/ontology/spouse", "spouse"),
    "founded by": ("http://dbpedia.org/ontology/foundedBy", "founder"),
    "founded": ("http://dbpedia.org/ontology/foundedBy", "founder"),
    "founder": ("http://dbpedia.org/ontology/foundedBy", "founder"),
    "who founded": ("http://dbpedia.org/ontology/foundedBy", "founder"),
    "industry": ("http://dbpedia.org/ontology/industry", "industry"),
    "location": ("http://dbpedia.org/ontology/locationCity", "location"),
    "headquartered": ("http://dbpedia.org/ontology/locationCity", "headquarters location"),
    "headquarters": ("http://dbpedia.org/ontology/locationCity", "headquarters location"),
    "country": ("http://dbpedia.org/ontology/country", "country"),
    "population": ("http://dbpedia.org/ontology/populationTotal", "population"),
    "leader": ("http://dbpedia.org/ontology/leaderName", "leader"),
    "president": ("http://dbpedia.org/ontology/leaderName", "president/leader"),
    "prime minister": ("http://dbpedia.org/ontology/leaderName", "leader"),
    "currency": ("http://dbpedia.org/ontology/currency", "currency"),
    "area": ("http://dbpedia.org/ontology/areaTotal", "area"),
    "language": ("http://dbpedia.org/ontology/language", "language"),
    "official language": ("http://dbpedia.org/ontology/language", "official language"),
}


# ---------------------------------------------------------------------------
# Entity type -> DBpedia ontology URI
# ---------------------------------------------------------------------------

ENTITY_TYPE_TO_DBPEDIA_URI = {
    "PERSON": "<http://dbpedia.org/ontology/Person>",
    "ORG": "<http://dbpedia.org/ontology/Organisation>",
    "GPE": "<http://dbpedia.org/ontology/Place>",
    "MISC": "",
}


# ---------------------------------------------------------------------------
# Enrichment properties per entity type
# ---------------------------------------------------------------------------

ENRICHMENT_PROPERTIES = {
    "ORG": {
        "foundingDate": "http://dbpedia.org/ontology/foundingDate",
        "foundingYear": "http://dbpedia.org/ontology/foundingYear",
        "foundedBy": "http://dbpedia.org/ontology/foundedBy",
        "founder": "http://dbpedia.org/ontology/founder",
        "industry": "http://dbpedia.org/ontology/industry",
        "type": "http://dbpedia.org/ontology/type",
        "service": "http://dbpedia.org/ontology/service",
        "product": "http://dbpedia.org/ontology/product",
        "keyPerson": "http://dbpedia.org/ontology/keyPerson",
        "numberOfEmployees": "http://dbpedia.org/ontology/numberOfEmployees",
        "revenue": "http://dbpedia.org/ontology/revenue",
        "netIncome": "http://dbpedia.org/ontology/netIncome",
        "operatingIncome": "http://dbpedia.org/ontology/operatingIncome",
        "locationCity": "http://dbpedia.org/ontology/locationCity",
        "locationCountry": "http://dbpedia.org/ontology/locationCountry",
        "subsidiary": "http://dbpedia.org/ontology/subsidiary",
    },
    "PERSON": {
        "birthDate": "http://dbpedia.org/ontology/birthDate",
        "birthPlace": "http://dbpedia.org/ontology/birthPlace",
        "deathDate": "http://dbpedia.org/ontology/deathDate",
        "deathPlace": "http://dbpedia.org/ontology/deathPlace",
        "occupation": "http://dbpedia.org/ontology/occupation",
        "nationality": "http://dbpedia.org/ontology/nationality",
        "spouse": "http://dbpedia.org/ontology/spouse",
        "almaMater": "http://dbpedia.org/ontology/almaMater",
        "knownFor": "http://dbpedia.org/ontology/knownFor",
        "employer": "http://dbpedia.org/ontology/employer",
        "award": "http://dbpedia.org/ontology/award",
    },
    "GPE": {
        "populationTotal": "http://dbpedia.org/ontology/populationTotal",
        "areaTotal": "http://dbpedia.org/ontology/areaTotal",
        "country": "http://dbpedia.org/ontology/country",
        "capital": "http://dbpedia.org/ontology/capital",
        "leaderName": "http://dbpedia.org/ontology/leaderName",
        "type": "http://dbpedia.org/ontology/type",
        "language": "http://dbpedia.org/ontology/language",
        "currency": "http://dbpedia.org/ontology/currency",
    },
    "MISC": {},
}


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def query_dbpedia(sparql_query: str) -> list[dict]:
    """Execute a SPARQL query against the DBpedia endpoint."""
    return query_sparql(DBPEDIA_ENDPOINT, sparql_query)


def build_sparql_query(name: str, dbpedia_type_uri: str) -> str:
    """Build a SPARQL entity-lookup query for *name* with an optional type
    constraint."""
    if dbpedia_type_uri:
        type_clause = (
            f"OPTIONAL {{ ?s "
            f"<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
            f"{dbpedia_type_uri} . }}"
        )
    else:
        type_clause = ""
    return SPARQL_QUERY_TEMPLATE.format(name=name, dbpedia_type=type_clause)


def query_relationship(entity_name: str, property_uri: str) -> list[dict]:
    """Query DBpedia for the object of a relationship property on an entity."""
    query = SPARQL_RELATIONSHIP_TEMPLATE.format(
        name=entity_name, property_uri=property_uri
    )
    results = query_dbpedia(query)
    out = []
    for r in results:
        entry = {"label": r.get("objLabel", {}).get("value", "")}
        if "objComment" in r:
            entry["comment"] = r["objComment"]["value"]
        out.append(entry)
    return out


def enrich_entity(name: str, entity_type: str) -> list[str]:
    """Fetch curated property/value facts for an entity from DBpedia.

    Returns a list of "propLabel: value" strings. URI objects are rendered
    using their English rdfs:label when available.
    """
    props = ENRICHMENT_PROPERTIES.get(entity_type, {})
    if not props:
        return []

    values_block = "\n    ".join(
        f'(<{uri}> "{label}")' for label, uri in props.items()
    )
    query = SPARQL_ENRICHMENT_TEMPLATE.format(name=name, values=values_block)
    try:
        results = query_dbpedia(query)
    except Exception as e:
        print(f"[Warning] Enrichment query failed for {name}: {e}")
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

    Returns a structured context string with one paragraph per entity,
    combining the best description available and enrichment facts.
    """
    context_parts = []

    for entity_type in ENTITY_TYPES:
        if entity_type not in entities:
            continue
        dbpedia_uri = ENTITY_TYPE_TO_DBPEDIA_URI.get(entity_type)
        if dbpedia_uri is None:
            continue

        for name in entities[entity_type]:
            try:
                query = build_sparql_query(name, dbpedia_uri)
                results = query_dbpedia(query)
                best = None
                for result in results:
                    if "comment" in result and "value" in result["comment"]:
                        value = result["comment"]["value"]
                        if best is None or len(value) > len(best):
                            best = value
                facts = enrich_entity(name, entity_type)
                if best and facts:
                    context_parts.append(
                        f"{name}: {best}\n" + "\n".join(facts))
                elif best:
                    context_parts.append(f"{name}: {best}")
                elif facts:
                    context_parts.append(f"{name}:\n" + "\n".join(facts))
            except Exception as e:
                print(f"[Warning] SPARQL query failed for {name}: {e}")

    return "\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Answer pipeline
# ---------------------------------------------------------------------------

def answer_question(question: str) -> tuple[str, str]:
    """Answer a question using DBpedia + Fireworks.ai.

    Returns (answer_text, context_used).
    """
    entities = extract_entities(question)
    print(f"[DEBUG] Entities found: {entities}")

    # Relationship detection -> direct SPARQL property lookup
    relationship = detect_relationship(question, RELATIONSHIP_PROPERTIES)
    if relationship:
        property_uri, verb = relationship
        print(f"[DEBUG] Relationship detected: {verb} ({property_uri})")
        all_names = [n for names in entities.values() for n in names]
        if not all_names:
            all_names = [w for w in question.replace("?", "").split()
                         if w[0].isupper()]
        for name in all_names:
            try:
                rel_results = query_relationship(name, property_uri)
                if rel_results:
                    parts = []
                    for r in rel_results[:5]:
                        label = r["label"]
                        comment = r.get("comment", "")
                        if comment:
                            parts.append(f"{label} ({comment})")
                        else:
                            parts.append(label)
                    answer = f"The {verb} of {name} is: {', '.join(parts)}."
                    return answer, "\n".join(parts)
            except Exception as e:
                print(f"[Warning] Relationship query failed for {name}: {e}")

    # General entity lookup + enrichment
    context = get_entity_context(entities)
    print(f"[DEBUG] Context text length: {len(context)} chars")

    if not context.strip():
        return "I couldn't find relevant entities in the knowledge base.", ""

    # LLM synthesis
    answer = synthesize_answer(question, context, source_label="DBpedia")
    if answer:
        return answer, context

    # Fallback: raw descriptions
    fallback_parts = []
    for entity_type in ENTITY_TYPES:
        if entity_type not in entities:
            continue
        dbpedia_uri = ENTITY_TYPE_TO_DBPEDIA_URI.get(entity_type)
        if dbpedia_uri is None:
            continue
        for name in entities[entity_type]:
            try:
                query = build_sparql_query(name, dbpedia_uri)
                results = query_dbpedia(query)
                best = None
                for result in results:
                    if "comment" in result and "value" in result["comment"]:
                        value = result["comment"]["value"]
                        if best is None or len(value) > len(best):
                            best = value
                if best:
                    fallback_parts.append(f"{name}: {best}")
            except Exception as e:
                print(f"[Warning] Query failed for {name}: {e}")

    if fallback_parts:
        return "\n\n".join(fallback_parts), context
    return "Unable to generate an answer.", context


# ---------------------------------------------------------------------------
# Multi-turn chat
# ---------------------------------------------------------------------------

def chat_with_context(system_prompt: str = None):
    """Create a multi-turn conversation with DBpedia knowledge."""
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        while True:
            user_input = input("\n> ")
            if user_input.lower() in ["quit", "exit", "bye"]:
                break

            entities = extract_entities(user_input)
            context = get_entity_context(entities)

            message_content = f"Question: {user_input}"
            if context.strip():
                message_content += f"\n\nContext from DBpedia:\n{context[:2000]}"

            messages.append({"role": "user", "content": message_content})

            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=3500,
            )
            answer = response.choices[0].message.content.strip()
            print(f"\n{answer}")
            messages.append({"role": "assistant", "content": answer})

    except KeyboardInterrupt:
        print("\nGoodbye!")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_cli(answer_question, "DBPedia.py")
