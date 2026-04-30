# dbpedia_cities.py - Query DBPedia for city data
#
# Demonstrates querying the public DBPedia SPARQL endpoint for structured
# knowledge about cities, including population and country. DBPedia mirrors
# Wikipedia's structured content as RDF triples.
#
# Requirements: uv pip install sparqlwrapper
# Run: uv run dbpedia_cities.py

from SPARQLWrapper import SPARQLWrapper, JSON

queryString = """
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

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setQuery(queryString)
sparql.setReturnFormat(JSON)
results = sparql.queryAndConvert()

for r in results["results"]["bindings"]:
    city = r['dbpedia_label']['value']
    pop = int(r['population']['value'])
    country = r.get('country_label', {}).get('value', 'unknown')
    print(f"  {city} ({country}): population {pop:,}")
