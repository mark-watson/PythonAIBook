When I need to collect text on an entity I often look for comment data on DBPedia. In this case, there were no English language comments so no comment results are in the returned JSON:

```JSON
 $ python opencyc_example_3.py 
{'dbpedia_label': {'type': 'literal',
                   'value': 'Hillary Rodham Clinton',
                   'xml:lang': 'en'},
 'dbpedia_uri': {'type': 'uri',
                 'value': 'http://dbpedia.org/resource/Hillary_Rodham_Clinton'},
 'label': {'type': 'literal', 'value': 'Hillary Clinton', 'xml:lang': 'en'}}```
```

If we didn't use the OPTIONAL operator then we would not have retrieved data for the first pattern in the WHERE clause.

We now leave our discussion of using the antiquated and no longer updated OpenCyc data and look at Python code in the next section that uses the Wikidata SPARQL server rather than DBPedia.

## Examples Using Wikidata Instead of DBPedia 

Wikidata uses abstract URIs instead of human readable URIs that DBPedia uses. Because of Wikidata's abstract URIs I usually use DBPedia when experimenting with new ideas. That said, there is more data in Wikidata. The examples in this section will get you started if you want to experiment with Wikidata.

As with DBPedia, start with Wikidata's public SPARQL endpoint [https://query.wikidata.org](https://query.wikidata.org). I want to walk you through resolving abstract URIs to something human readable by starting with a SPARQL query:

![Wikidata public SPARQL endpoint](wikidata1.png)

In the SPARQL results there are three matching subjects:

- [wd:Q37156](https://www.wikidata.org/wiki/Q37156) that has the URI value of [https://www.wikidata.org/wiki/Q37156](https://www.wikidata.org/wiki/Q37156). Click on this link. This is the entity we want but also try clicking on these links:
- [wd:Q5968787](https://www.wikidata.org/wiki/Q5968787)
- [wd:Q19874511](https://www.wikidata.org/wiki/Q19874511)

Here is a simple Python program that uses Wikidata:

```python
## Test client for Wikidata SPARQL endpoint

from SPARQLWrapper import SPARQLWrapper, JSON
from pprint import pprint

queryString = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT *
WHERE {
  ?subject skos:altLabel "International Business Machines"@en .
}
LIMIT 4
"""

uris = []
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setQuery(queryString)
sparql.setReturnFormat(JSON)
ret = sparql.query().convert()
for r in ret["results"]["bindings"]:
  pprint(r)
  if 'subject' in r:
    if 'value' in r['subject']:
      uri = r['subject']['value']
      print(uri)
      uris.append(uri)

queryString2 = """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT *
WHERE {
  <A_URI> wdt:P31 ?entity_label . # wdt:P31 is instanceOf
  ?entity_label skos:altLabel ?entity_human_readable_label
    FILTER (lang(?entity_human_readable_label) = 'en') .
}
LIMIT 5
"""

def wd_helper(an_ibm_uri):
    print(f"\n *** {an_ibm_uri} ***\n")
    query = queryString2.replace("A_URI", an_ibm_uri)
    #print(query)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    ret = sparql.query().convert()['results']['bindings']
    for r in ret:
      print(r['entity_human_readable_label']['value'])

for uri in uris:
  wd_helper(uri)
```

The output is:


```
$ python wikidata1.py
{'subject': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q37156'}}
http://www.wikidata.org/entity/Q37156
{'subject': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q5968787'}}
http://www.wikidata.org/entity/Q5968787
{'subject': {'type': 'uri',
             'value': 'http://www.wikidata.org/entity/Q19874511'}}
http://www.wikidata.org/entity/Q19874511

 *** http://www.wikidata.org/entity/Q37156 ***

device
hw
hardware
computer component
computer accessory

 *** http://www.wikidata.org/entity/Q5968787 ***

edifice
buildings

 *** http://www.wikidata.org/entity/Q19874511 ***

lab
research laboratory
research facility
research lab
laboratories
```

I would like you to have a few takeaways from this material:

- When using public Knowledge Graphs like DBPedia and Wikipedia, you want to start by using the public SPARQL endpoints to explore the data to understand what might be useful for your project.
- Write low-level libraries to make SPARQL queries and filter and transform the JSON query results data to a form that you can easily use.
- Given a foundation of data access and transformation tools, then write your application.

In the next section we look at a tool I wrote for exploring Knowledge Graphs.
