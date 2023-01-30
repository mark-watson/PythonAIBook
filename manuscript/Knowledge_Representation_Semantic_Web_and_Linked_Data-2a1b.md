
Of particular use is the matched result:

```
?p	?o
<http://www.w3.org/2002/07/owl#sameAs>
  <http://dbpedia.org/resource/Hillary_Rodham_Clinton>
```

This lets us combine data from OpenCyc with DBPedia (limit to just 2500 results). This is not a good example since we are in no way tying together data from OpenCyc to DBPedia (we will combine the results later), rather we are just doing two separate queries:

```sparql
SELECT *
WHERE {
  <http://sw.opencyc.org/concept/Mx4rvV7SqpwpEbGdrcN5Y29ycA> ?p ?o .

  SERVICE <http://dbpedia.org/sparql?timeout=30000> {
    <http://dbpedia.org/resource/Hillary_Rodham_Clinton>
      ?p_dbpedia_1
      ?o_dbpedia_1 .
    ?s_dbpedia_2
      ?p_dbpedia_2
      <http://dbpedia.org/resource/Hillary_Rodham_Clinton> .
  }
} limit 2500
```

Two results chosen that only used English language (DBPedia has triples containing text in many human languages):



```
?p	?o	?p_dbpedia_1	?o_dbpedia_1	?s_dbpedia_2	?p_dbpedia_2

<http://www.w3.org/2002/07/owl#sameAs>
  <http://dbpedia.org/resource/Hillary_Rodham_Clinton>
  <http://www.w3.org/2000/01/rdf-schema#label>
  "Hillary Clinton"@en
  <http://dbpedia.org/resource/American_Academy_of_Arts_and_Sciences_members>
  <http://dbpedia.org/ontology/wikiPageWikiLink>

<http://www.w3.org/2002/07/owl#sameAs>
  <http://dbpedia.org/resource/Hillary_Rodham_Clinton>
  <http://www.w3.org/2000/01/rdf-schema#label>
  "Hillary Rodham Clinton"@en
  <http://dbpedia.org/resource/Lincoln_Bedroom_for_contributors_controversy>
  <http://dbpedia.org/ontology/wikiPageWikiLink>
```

If we don't link data from two RDF services then we are obviously better off doing two separate queries and combining the results in our application.

Before linking data for OpenCyc and DBPedia, let's look at a Python SPARQL query example:

```python
import rdflib
from SPARQLWrapper import SPARQLWrapper, JSON
from pprint import pprint

queryString = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbo: <http://dbpedia.org/ontology/>
SELECT *
WHERE {
    <http://sw.opencyc.org/concept/Mx4rvV7SqpwpEbGdrcN5Y29ycA> 
      rdfs:label ?label
      FILTER (lang(?label) = 'en') .
    <http://sw.opencyc.org/concept/Mx4rvV7SqpwpEbGdrcN5Y29ycA>
      <http://www.w3.org/2002/07/owl#sameAs> ?dbpedia_uri
      filter(strstarts(str(?dbpedia_uri), "http://dbpedia.org/resource")) .
}
LIMIT 5
"""

sparql = SPARQLWrapper("http://localhost:3030/opencyc")
sparql.setQuery(queryString)
sparql.setReturnFormat(JSON)
sparql.setMethod('POST')
ret = sparql.queryAndConvert()
for r in ret["results"]["bindings"]:
    pprint(r)
```

