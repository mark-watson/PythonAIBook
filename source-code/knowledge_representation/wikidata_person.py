# wikidata_person.py - Query Wikidata for information about a person
#
# Demonstrates querying the Wikidata SPARQL endpoint for structured
# knowledge about a specific person, including birth place, birth date,
# and occupations. Wikidata uses numeric entity and property identifiers
# (e.g., Q5 = "human", P19 = "place of birth").
#
# Requirements: uv pip install sparqlwrapper
# Run: uv run wikidata_person.py [person name]

import sys
from SPARQLWrapper import SPARQLWrapper, JSON

QUERY_TEMPLATE = """
SELECT ?personLabel ?birthPlaceLabel ?birthDate
       (GROUP_CONCAT(DISTINCT ?occupationLabel; SEPARATOR=", ") AS ?occupations)
WHERE {{
    ?person wdt:P31 wd:Q5 .
    ?person rdfs:label "{name}"@en .
    OPTIONAL {{ ?person wdt:P19 ?birthPlace . }}
    OPTIONAL {{ ?person wdt:P569 ?birthDate . }}
    OPTIONAL {{ ?person wdt:P106 ?occupation . }}
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
GROUP BY ?personLabel ?birthPlaceLabel ?birthDate
LIMIT 5
"""


def fetch_person(name: str) -> list[dict[str, str]]:
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.addCustomHttpHeader("User-Agent", "PythonAIBook/1.0")
    sparql.setQuery(QUERY_TEMPLATE.format(name=name))
    sparql.setReturnFormat(JSON)
    results = sparql.queryAndConvert()

    bindings = results.get("results", {}).get("bindings", [])
    people = []
    for r in bindings:
        people.append({
            "name": r.get("personLabel", {}).get("value", "unknown"),
            "birth_place": r.get("birthPlaceLabel", {}).get("value", ""),
            "birth_date": r.get("birthDate", {}).get("value", "")[:10],
            "occupations": r.get("occupations", {}).get("value", ""),
        })
    return people


if __name__ == "__main__":
    person_name = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Albert Einstein"
    try:
        people = fetch_person(person_name)
        if not people:
            print(f"No results found for '{person_name}'.")
        else:
            for p in people:
                print(f"  Name: {p['name']}")
                if p["birth_place"]:
                    print(f"  Born: {p['birth_place']}")
                if p["birth_date"]:
                    print(f"  Date: {p['birth_date']}")
                if p["occupations"]:
                    print(f"  Occupations: {p['occupations']}")
                print()
    except Exception as e:
        print(f"Error querying Wikidata: {e}")
