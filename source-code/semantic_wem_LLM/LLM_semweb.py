# /usr/bin/env python3
"""
LLM_semweb.py - Semantic Web Question Answering with SPARQL and Fireworks.ai

This utility answers natural language questions using:
1. Fireworks.ai LLM for entity extraction and typing
2. DBpedia SPARQL endpoint for knowledge base queries
3. Fireworks.ai deepseek-v4-flash model to augment/refine SPARQL queries

Usage with uv:
    uv add openai SPARQLWrapper click

Set environment variable:
    export FIREWORKS_API_KEY="your-api-key"

Example questions this can answer:
    "Where does Bill Gates work?"
    "What is the population of Paris?"
    "Who is Bill Clinton married to?"
"""

import os
import json
import re
from openai import OpenAI
from SPARQLWrapper import SPARQLWrapper, JSON


# Initialize Fireworks.ai client
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.getenv("FIREWORKS_API_KEY"),
)

MODEL_ID = "accounts/fireworks/models/deepseek-v4-flash"


# SPARQL query template for DBpedia entity lookup by label.
# Tries multiple descriptive predicates (description, comment, abstract) since
# DBpedia's endpoint has changed over time, and makes the rdf:type constraint
# OPTIONAL so entities whose LLM-tagged type doesn't exactly match the
# DBpedia ontology (e.g. Pepsi tagged as Organisation but typed as dbo:Beverage)
# still resolve.
SPARQL_QUERY_TEMPLATE = """
SELECT DISTINCT ?s ?comment WHERE {{
  ?s <http://www.w3.org/2000/01/rdf-schema#label> '{name}'@en .
  ?s ?p ?comment .
  FILTER (lang(?comment) = 'en') .
  VALUES ?p {{
    <http://dbpedia.org/ontology/description>
    <http://www.w3.org/2000/01/rdf-schema#comment>
    <http://dbpedia.org/ontology/abstract>
  }}
  {dbpedia_type}
}} LIMIT 15
"""

# SPARQL query template for looking up a relationship/property of an entity.
# Given an entity label and a DBpedia property URI, this fetches the related
# entity's label and a short description so we can answer "what is the capital
# of France" style questions.
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

# Mapping from common English words/phrases to DBpedia ontology property URIs.
# Used to translate relationship questions ("what is the capital of France?")
# into SPARQL property lookups. Keys are lowercase; matching is substring-based.
# Each entry maps to a (property_uri, human_readable_verb) tuple.
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


def query_dbpedia(sparql_query: str):
    """Execute a SPARQL query against DBpedia and return results in JSON format."""
    print(f"SPARQL query: {sparql_query}")
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    ret = sparql.query().convert()["results"]["bindings"]
    print(f"Results:\n{ret}\n")
    return ret


def detect_relationship(question: str) -> tuple[str, str] | None:
    """Detect a relationship query in the question and return (property_uri, verb).

    Scans the question for any key in RELATIONSHIP_PROPERTIES. Multi-word keys
    are checked first so "capital of" wins over bare "capital". Returns None if
    no known relationship is found.
    """
    lower = question.lower()
    # Sort by descending key length so multi-word phrases match first
    for key in sorted(RELATIONSHIP_PROPERTIES, key=len, reverse=True):
        if key in lower:
            uri, verb = RELATIONSHIP_PROPERTIES[key]
            return uri, verb
    return None


def query_relationship(entity_name: str, property_uri: str) -> list[dict]:
    """Query DBpedia for the object of a relationship property on an entity.

    Returns a list of dicts with keys 'label' and 'comment' for each related
    entity found.
    """
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


def extract_entities(text: str) -> dict:
    """Extract named entities from text using the Fireworks.ai LLM.

    Uses a one-shot prompt so the model sees an example input/output pair
    before classifying the user's text. Returns a dict mapping entity types
    (PERSON, ORG, GPE, MISC) to lists of entity names, matching the shape the
    rest of the pipeline expects.
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
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": one_shot_prompt.replace("{text}", text)}],
            max_tokens=300,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
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


def build_sparql_query(name: str, dbpedia_type_uri: str) -> str:
    """Build a SPARQL query for entities by name and type.

    When dbpedia_type_uri is empty (e.g. for MISC entities with no single
    DBpedia type), the rdf:type OPTIONAL clause is omitted entirely.
    """
    if dbpedia_type_uri:
        type_clause = (
            f"OPTIONAL {{ ?s "
            f"<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
            f"{dbpedia_type_uri} . }}"
        )
    else:
        type_clause = ""
    return SPARQL_QUERY_TEMPLATE.format(
        name=name, dbpedia_type=type_clause
    )


# Mapping from LLM entity types to DBpedia ontology URIs.
# MISC has no single DBpedia type, so we omit the type filter entirely by
# passing an empty OPTIONAL pattern (the label match alone resolves it).
ENTITY_TYPE_TO_DBPEDIA_URI = {
    "PERSON": "<http://dbpedia.org/ontology/Person>",
    "ORG": "<http://dbpedia.org/ontology/Organisation>",
    "GPE": "<http://dbpedia.org/ontology/Place>",
    "MISC": "",
}


def get_entity_context(entities: dict) -> str:
    """Execute SPARQL queries for extracted entities and build context text.

    Returns concatenated commentary about each entity found in DBpedia.
    """
    context_parts = []

    def process_entities(entity_type):
        if entity_type not in entities:
            return ""
        dbpedia_uri = ENTITY_TYPE_TO_DBPEDIA_URI.get(entity_type)
        if dbpedia_uri is None:
            return ""

        for name in entities[entity_type]:
            try:
                query = build_sparql_query(name, dbpedia_uri)
                results = query_dbpedia(query)
                for result in results:
                    if "comment" in result and "value" in result["comment"]:
                        context_parts.append(result["comment"]["value"])
            except Exception as e:
                print(f"[Warning] SPARQL query failed for {name}: {e}")

        return ""

    # Process in a stable order for deterministic output
    for entity_type in ["PERSON", "ORG", "GPE", "MISC"]:
        process_entities(entity_type)

    return " ".join(context_parts)


def refine_query_with_llm(question: str, context: str) -> str:
    """Use Fireworks.ai to refine or clarify the question for better SPARQL queries.

    The LLM can help by:
    - Identifying what specific information is being asked
    - Clarifying ambiguous references (like "it", "they")
    - Suggesting DBpedia properties to query

    Returns a refined/clarified version of the input question.
    """
    if not context.strip():
        return question  # No point refining without context

    prompt = f"""You are an expert in semantic web and SPARQL querying. 

Given this natural language question: "{question}"

And this knowledge base content extracted from DBpedia (use only entity names/types here):

{context[:2000] if len(context) > 2000 else context}

Clarify the question to make it more specific for SPARQL querying. Focus on:
- What exact information is being asked?
- Which entities are relevant?
- What relationship is being sought?

Return ONLY a single refined question string, no explanation or formatting."""

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Warning] LLM refinement failed: {e}")
        return question


def generate_sparql_with_llm(question: str, entities: dict) -> list[str]:
    """Use the LLM to suggest SPARQL patterns or help generate queries.

    Returns a list of potential SPARQL query strings the model would like us to try.
    """
    entity_info = ", ".join(
        f"{etype}: {names}" for etype, names in entities.items()
    ) if entities else "No clear entities"

    prompt = f"""You are an expert SPARQL query generator familiar with DBpedia ontology.

Question: "{question}"

Detected entities:
{entity_info}

DBpedia property hints (common properties):
- dbo:birthPlace, dbo:deathPlace for people
- dbo:location, dbo:headquarterSettlement for organizations
- dbo:populationTotal for places
- dbo:spouse for person relationships
- dbo:capital for countries/cities with capitals

Please generate SPARQL query patterns to answer this question. 
Return a Python-style list of query strings using DBpedia property URIs like:
<http://dbpedia.org/ontology/location>
<http://dbpedia.org/sparql>

Each query should use ?s and ?comment as variables, querying for entity labels."""

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
        )

        content = response.choices[0].message.content.strip()

        # Try to parse the response for query strings
        queries = []
        return queries  # For now, return empty list and fall back to our templates

    except Exception as e:
        print(f"[Warning] LLM SPARQL generation failed: {e}")
        return []


def answer_question(question: str) -> tuple[str, str]:
    """Main function to answer a question using SPARQL + DBpedia + Fireworks.ai.

    Returns a tuple of (answer_text, context_used).
    """
    # Step 1: Extract entities from the question using the LLM
    entities = extract_entities(question)
    print(f"[DEBUG] Entities found: {entities}")

    # Step 1b: Check if the question asks about a known relationship/property
    # (e.g. "capital of France", "born in", "spouse of"). If so, query the
    # specific DBpedia property directly rather than just fetching descriptions.
    relationship = detect_relationship(question)
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

    # Step 2: Query DBpedia for entity information to build context
    context = get_entity_context(entities)
    print(f"[DEBUG] Context text length: {len(context)} chars")

    if not context.strip():
        return "I couldn't find relevant entities in the knowledge base.", ""

    # Step 3: Use LLM to potentially refine the question or generate additional queries
    refined_question = refine_query_with_llm(question, context)
    print(f"[DEBUG] Refined question: {refined_question}")

    # Step 4: Answer using the original approach (simplified - first query that has results)
    for entity_type in ["PERSON", "ORG", "GPE", "MISC"]:
        if entity_type not in entities:
            continue

        dbpedia_uri = ENTITY_TYPE_TO_DBPEDIA_URI.get(entity_type)
        if dbpedia_uri is None:
            continue

        for name in entities[entity_type]:
            try:
                query = build_sparql_query(name, dbpedia_uri)
                results = query_dbpedia(query)

                if results:
                    # Return the matching comment/description for this entity
                    for result in results[:3]:  # Top 3 results
                        if "comment" in result and "value" in result["comment"]:
                            return result["comment"]["value"], context + "\n\n" + name

            except Exception as e:
                print(f"[Warning] Query failed for {name}: {e}")
                continue

    # Step 5: Fall back to LLM-based reasoning if no direct match found
    try:
        short_context = context[:3000] if len(context) > 3000 else context
        llm_prompt = (
            f'Given question: "{question}"\n\n'
            f"Knowledge from DBpedia about relevant entities:\n{short_context}\n\n"
            "Please answer the question based on this knowledge. Be specific and cite entity names where possible."
        )
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": llm_prompt}],
            max_tokens=500,
        )
        return response.choices[0].message.content.strip(), context
    except Exception as e:
        print(f"[Error] LLM answering failed: {e}")
        return "Unable to generate an answer.", context


def main():
    """Command-line interface for the semantic web QA utility."""
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("LLM_semweb.py - QA with SPARQL + DBpedia")
        print("-" * 50)
        question = input("Enter your question: ")

    print(f"\nQuestion: {question}")
    print("-" * 50)

    answer, context = answer_question(question)

    print("\nAnswer:")
    print(answer)


# Additional utility functions for Fireworks.ai integration

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
                max_tokens=500,
            )

            answer = response.choices[0].message.content.strip()
            print(f"\n{answer}")

            messages.append({"role": "assistant", "content": answer})

    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
