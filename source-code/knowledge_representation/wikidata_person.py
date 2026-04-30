# wikidata_person.py - Query Wikidata for information about a person
#
# Demonstrates querying the Wikidata SPARQL endpoint for structured
# knowledge about a specific person, including birth place, birth date,
# and occupations. Wikidata uses numeric entity and property identifiers
# (e.g., Q5 = "human", P19 = "place of birth").
#
# Requirements: uv pip install sparqlwrapper
# Run: uv run wikidata_person.py

from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.addCustomHttpHeader("User-Agent", "PythonAIBook/1.0")

queryString = """
SELECT ?personLabel ?birthPlaceLabel ?birthDate ?occupationLabel
WHERE {
    ?person wdt:P31 wd:Q5 .            # instance of human
    ?person rdfs:label "Albert Einstein"@en .
    OPTIONAL { ?person wdt:P19 ?birthPlace . }
    OPTIONAL { ?person wdt:P569 ?birthDate . }
    OPTIONAL { ?person wdt:P106 ?occupation . }
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
}
LIMIT 10
"""

sparql.setQuery(queryString)
sparql.setReturnFormat(JSON)
results = sparql.queryAndConvert()

for r in results["results"]["bindings"]:
    print(f"  Name: {r['personLabel']['value']}")
    if 'birthPlaceLabel' in r:
        print(f"  Born: {r['birthPlaceLabel']['value']}")
    if 'birthDate' in r:
        print(f"  Date: {r['birthDate']['value'][:10]}")
    if 'occupationLabel' in r:
        print(f"  Occupation: {r['occupationLabel']['value']}")
    print()
