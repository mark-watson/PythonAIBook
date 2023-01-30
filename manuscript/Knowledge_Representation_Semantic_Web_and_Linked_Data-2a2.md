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

## Knowledge Graph Navigator: Use English to Explore DBPedia

When I need to use DBPedia data in applications before I start writing any code I explore possibly useful information using:

- [https://dbpedia.org/sparql](https://dbpedia.org/sparql) DBPedia SPARQL endpoint.
- [https://dbpedia.org/fct/](https://dbpedia.org/fct/) OPEN LINK Software text search and entity search.

The following example is my effort to create a tool that quickly identifies entities like people, places, and organization and the relations between these discovered entities.


I published the **kgn** command line Python app to PyPy: [https://pypi.org/project/kgn/](https://pypi.org/project/kgn/). The GitHub repository is [https://github.com/mark-watson/kgn](https://github.com/mark-watson/kgn).

We will look at a few snippets of the code. Here is a roadmap by source file in alphabetical order:

- cache.py caches SPARQL query results in an SQLite database.
- cli.py is the top level command line tool.
- colorize.py colorizes generated SPARQL queries to make them more readable.
- kgn.py is the main logic for the Knowledge Graph Navigator.
- kgnutils.py contains a function for resolving text entity names into DBPedia URIs.
- relationships.py takes a list of N entity URIs and performs an exhaustive search to find relationships between pairs of entities. This code runs O(N^2) so it is best to not input more than 5 or 6 text entity names.
- sparql.py is a collection of reusable SPARQL utilities.
- textui.py contains helper functions for the text-based user interface.

This is a fairly long example but if you followed the previous Python + SPARQL query examples, then it should be fairly clear how this works. When we identify entities in input text we generate SPARQL queries to match the literal entity names. If this is possible, then we have the DBPedia URIs for entities in the input text and it is straightforward to get comment text for entities and search for properties (relationships) that link any two entity URIs using a SPARQL matching pattern like:

```sparql
  <entity_1_URI> ?p <entity_2_URI> .
```

Listing of the main application logic in kgn.py:

```python
from pprint import pprint
from .kgnutils import dbpedia_get_entities_by_name
from .textui import select_entities, get_query
from .relationships import entity_results_to_relationship_links
import spacy

try:
  nlp_model = spacy.load('en_core_web_sm')
except:
  print("Loading spaCy model file...")
  from os import system
  system("python -m spacy download en_core_web_sm")
  nlp_model = spacy.load('en_core_web_sm')


def entities_in_text(s):
    " use spaCY to find entity names in text "
    doc = nlp_model(s)
    ret = {}
    for [ename, etype] in [[entity.text, entity.label_] for entity in doc.ents]:
        if etype in ret:
            ret[etype] = ret[etype] + [ename]
        else:
            ret[etype] = [ename]
    return ret

entity_type_to_type_uri = {'PERSON': '<http://dbpedia.org/ontology/Person>',
    'GPE': '<http://dbpedia.org/ontology/Place>', 'ORG':
    '<http://dbpedia.org/ontology/Organisation>'}
short_comment_to_uri = {}

def shorten_comment(comment, uri):
    sc = comment[0:70:None] + '...'
    short_comment_to_uri[sc] = uri
    return sc

query = ''

def kgn():
    print("Knowledge Graph Navigator (note: only runs in a terminal)")
    while True:
        query = get_query()
        if query == 'quit' or query == 'q':
            break
        elist = entities_in_text(query)
        people_found_on_dbpedia = []
        places_found_on_dbpedia = []
        organizations_found_on_dbpedia = []
        global short_comment_to_uri
        short_comment_to_uri = {}
        for key in elist:
            type_uri = entity_type_to_type_uri[key]
            for name in elist[key]:
                dbp = dbpedia_get_entities_by_name(name, type_uri)
                for d in dbp:
                    short_comment = shorten_comment(d[1][1], d[0][1])
                    people_found_on_dbpedia.extend([name + ' || ' +
                        short_comment]) if key == 'PERSON' else None
                    places_found_on_dbpedia.extend([name + ' || ' +
                        short_comment]) if key == 'GPE' else None
                    organizations_found_on_dbpedia.extend([name + ' || ' +
                        short_comment]) if key == 'ORG' else None
        user_selected_entities = select_entities(people_found_on_dbpedia,
            places_found_on_dbpedia, organizations_found_on_dbpedia)
        uri_list = []
        for entity in user_selected_entities['entities']:
            short_comment = entity[4 + entity.index(' || '):None:None]
            uri_list.extend([short_comment_to_uri[short_comment]])
        print("\n\nEntity data:")
        pprint(user_selected_entities)
        print("\n\n")
        relation_data = (
            entity_results_to_relationship_links(uri_list))
        print('\n\nDiscovered relationship links:\n')
        for relationship in relation_data:
            print(relationship[0] + ' --> ' + relationship[2][1] +
                  ' --> ' + relationship[1])
```

This command line tool does not run very well in a shell in IDEs like PyCharm to pay attention to the printed prompt in line 44 and run **kgn** in a terminal window that properly renders unicode characters and colored/styled text.

A listing of kgnutils.py that uses a SPARQL query to resolve entity names to DBPedia URIs:

```python
from .sparql import dbpedia_sparql
from .colorize import colorize_sparql

def dbpedia_get_entities_by_name(name, dbpedia_type):
    sparql = (
        'select distinct ?s ?comment {{ ?s ?p "{}"@en . ?s <http://www.w3.org/2000/01/rdf-schema#comment>  ?comment  . FILTER  (lang(?comment) = \'en\') . ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> {} . }} limit 15'
        .format(name, dbpedia_type))
    print('Generated SPARQL to get DBPedia entity URIs from a name:')
    print(colorize_sparql(sparql))
    return dbpedia_sparql(sparql)
```

The listing of relationships.py that finds RDF properties that link any two entity URIs:

```python
from .sparql import dbpedia_sparql
from .colorize import colorize_sparql

def flatten(a_list):
    return [item for items in a_list for item in items]

def dbpedia_get_relationships(s_uri, o_uri):
    query = (
        "SELECT DISTINCT ?p {{  {} ?p {} . FILTER (!regex(str(?p), 'wikiPage', 'i')) }} LIMIT 5"
        .format(s_uri, o_uri))
    results = dbpedia_sparql(query)
    print('Generated SPARQL to get relationships between two entities:')
    print(colorize_sparql(query))
    return [r for r in flatten(results) if not r == 'p']


def entity_results_to_relationship_links(uris):
    uris = [('<' + uri + '>') for uri in uris]
    relationship_statements = []
    for e1 in uris:
        for e2 in uris:
            if not e1 == e2:
                l1 = dbpedia_get_relationships(e1, e2)
                l2 = dbpedia_get_relationships(e2, e1)
                for x in l1:
                    relationship_statements.extend([[e1, e2, x]]) if not [e1,
                        e2, x] in relationship_statements else None
                for x in l2:
                    relationship_statements.extend([[e1, e2, x]]) if not [e1,
                        e2, x] in relationship_statements else None
    return relationship_statements
```

A listing of sparql.py that is a utility to query either the DBPedia or Wikidata SPARQL server endpoints:

```python
import requests
from .cache import fetch_result_dbpedia, save_query_results_dbpedia
wikidata_endpoint = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
dbpedia_endpoint = 'https://dbpedia.org/sparql'


def do_query_helper(endpoint, query):
    cached_results = fetch_result_dbpedia(query)
    if len(cached_results) > 0:
        print('Using cached query results')
        return cached_results # eval(cached_results)

    params = {'query': query, 'format': 'json'}
    response = requests.get(endpoint, params=params)
    json_data = response.json()
    vars = json_data['head']['vars']
    results = json_data['results']
    if 'bindings' in results:
        bindings = results['bindings']
        qr = [[[var, binding[var]['value']] for var in vars] for binding in bindings]
        save_query_results_dbpedia(query, qr)
        return qr
    return []


def wikidata_sparql(query):
    return do_query_helper(wikidata_endpoint, query)


def dbpedia_sparql(query):
    return do_query_helper(dbpedia_endpoint, query)
```

This example Python code for performing SPARQL queries differs from the previous examples that all used the **SPARQLWrapper** library. Here I used the Python **requests** library.


## Wrap Up for Semantic Web, Linked Data and Knowledge Graphs

I hope that you both enjoyed this chapter and that it has some practical use for you either in personal or professional projects. I favored the use of the open source Apache Jena/Fuseki platform. It is not open source but the free to use version of [Ontotext GraphDB](https://www.ontotext.com/products/graphdb/) has interesting graph visualization tools that you might want to experiment with. I also sometime use the commercial products [Franz AllegorGraph](https://allegrograph.com) and [Stardog](https://www.stardog.com/platform).

The Python examples in this chapter are simple examples to get you started. In real projects I build a library of low-level utilities to manipulate the JSON data returned from SPARQL endpoints. As an example, I almost always write filters for removing data that is text but not in English. This filtering is especially important for Wikidata that has most data replicated for most human written languages.