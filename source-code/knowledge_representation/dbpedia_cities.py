# dbpedia_cities.py - Query DBPedia for city data
#
# Demonstrates querying the public DBPedia SPARQL endpoint for structured
# knowledge about cities, including population and country. DBPedia mirrors
# Wikipedia's structured content as RDF triples.
#
# Requirements: uv pip install sparqlwrapper
# Run: uv run dbpedia_cities.py

from SPARQLWrapper import SPARQLWrapper, JSON

QUERY_STRING = """
SELECT ?city_uri ?dbpedia_label ?population ?country_label
WHERE {
    ?city_uri
        <http://dbpedia.org/ontology/type>
        <http://dbpedia.org/resource/City> .
    ?city_uri
        <http://dbpedia.org/property/populationEst>
        ?population .
    ?city_uri
         <http://www.w3.org/2000/01/rdf-schema#label>
         ?dbpedia_label FILTER (lang(?dbpedia_label) = 'en') .
    OPTIONAL {
        ?city_uri <http://dbpedia.org/ontology/country> ?country .
        ?country <http://www.w3.org/2000/01/rdf-schema#label>
                 ?country_label FILTER (lang(?country_label) = 'en') .
    }
}
ORDER BY DESC(?population)
LIMIT 10
"""


def fetch_cities() -> list[dict[str, str]]:
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.addCustomHttpHeader("User-Agent", "PythonAIBook/1.0")
    sparql.setQuery(QUERY_STRING)
    sparql.setReturnFormat(JSON)
    results = sparql.queryAndConvert()

    bindings = results.get("results", {}).get("bindings", [])
    cities = []
    for r in bindings:
        cities.append({
            "city": r.get("dbpedia_label", {}).get("value", "unknown"),
            "population": int(r.get("population", {}).get("value", 0)),
            "country": r.get("country_label", {}).get("value", "unknown"),
        })
    return cities


if __name__ == "__main__":
    try:
        cities = fetch_cities()
        if not cities:
            print("No results returned from DBpedia.")
        else:
            for c in cities:
                print(f"  {c['city']} ({c['country']}): population {c['population']:,}")
    except Exception as e:
        print(f"Error querying DBpedia: {e}")
