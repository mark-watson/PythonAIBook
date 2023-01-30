### OWL: The Web Ontology Language

We have already seen a few examples of using RDFS to define sub-properties in this chapter. The Web Ontology Language (OWL) extends the expressive power of RDFS. We now look at a few OWL examples and then look at parts of the Java unit test showing three SPARQL queries that use OWL reasoning. The following RDF data stores support at least some level of OWL reasoning:

- ProtegeOwlApis - compatible with the Protege Ontology editor.
- Pellet - DL reasoner.
- Owlim - OWL DL reasoner compatible with some versions of Sesame.
- Jena - General purpose library (we use Apache Jena Fuseki in this book).
- OWLAPI - a simpler API using many other libraries.
- Stardog - a commercial OWL and RDF reasoning system and datastore.
- Allegrograph - a commercial RDF+ and RDF reasoning system and datastore.

OWL is more expressive than RDFS in that it supports cardinality, richer class relationships, and Descriptive Logic (DL) reasoning. OWL treats the idea of classes very differently than object oriented programming languages like Java and Smalltalk, but similar to the way PowerLoom (see chapter on Reasoning) uses concepts (PowerLoom’s rough equivalent to a class). In OWL, instances of a class are referred to as individuals and class membership is determined by a set of properties that allow a DL reasoner to infer class membership of an individual (this is called entailment.)

We have been using the RDF file news.n3 in previous examples and we will layer new examples by adding new triples that represent RDF, RDFS, and OWL. We saw in news.n3 the definition of three triples using rdfs:subPropertyOf properties to create a more general kb:containsPlace property:

```sparql
kb:containsCity rdfs:subPropertyOf kb:containsPlace .
kb:containsCountry rdfs:subPropertyOf kb:containsPlace .
kb:containsState rdfs:subPropertyOf kb:containsPlace .

kb:containsPlace rdf:type owl:transitiveProperty .

kbplace:UnitedStates kb:containsState kbplace:Illinois .
kbplace:Illinois kb:containsCity kbplace:Chicago .
```

We can also infer that:

    kbplace:UnitedStates kb:containsPlace kbplace:Chicago .

We can also model inverse properties in OWL. For example, here we add an inverse property kb:containedIn, adding it to the example in the last listing:

    kb:containedIn owl:inverseOf kb:containsPlace .

Given an RDF container that supported extended OWL DL SPARQL queries, we can now execute SPARQL queries matching the property kb:containedIn and “match” triples in the RDF triple store that have never been asserted but are inferred by the OWL reasoner.

OWL DL is a very large subset of full OWL. From reading the chapter on Reasoning and the very light coverage of OWL in this section, you should understand the concept of class membership not by explicitly stating that an object (or individual) is a member of a class, but rather because an individual has properties that can be used to infer class membership.

The World Wide Web Consortium has defined three versions of the OWL language that are in increasing order of complexity: OWL Lite, OWL DL, and OWL Full. OWL DL (supports Description Logic) is the most widely used (and recommended) version of OWL. OWL Full is not computationally decidable since it supports full logic, multiple class inheritance, and other things that probably make it computationally intractable for all but smaller problems.


## A Hybrid Deep Learning and RDF/SPARQL Application for Question Answering

We will skip ahead a little and use two deep learning models (spaCy NLP and Transformer) with SPARQL queries to answer natural language questions. I wrote this example in January 2001 and it generated quite a lot of interest on social media and I later noticed several projects that picked up my basic idea of:

- Using spaCy to identify proper nouns in text (e.g., human names, locations, corporations, etc.).
- Use SPARQL queries to collect text describing all identified proper nouns.
- Use a question answering Transformer model with two inputs: the user's question and the text collected by the SPARQL queries.

**Note:** this example is now somewhat obsolete since GPT-3 (that we use later in the book) and ChatGPT models can answer questions directly. Still, the example we use here is very simple and "hackable" and I hope you enjoy it.

You can access this example on Google Colab [Colab DBPedia Sparql Question Answering Demo](https://colab.research.google.com/drive/1FX-0eizj2vayXsqfSB2ONuJYG8BaYpGO?usp=sharing).

![](DQA1.png)

TBD


![](DQA2.png)


TBD


![](DQA3.png)

TBD - describe Jupyter notebook example


## Knowledge Graph Creator: Convert Text Files to RDF Data Input Data for Fuseki


I published my **kgcreator** command line Python app to PyPy: [https://pypi.org/project/kgcreator/](https://pypi.org/project/kgcreator/). The GitHub repository is [https://github.com/mark-watson/kgcreator](https://github.com/mark-watson/kgcreator).

TBD: add code to also generate Neo4J input data

TBD

## Old Technology: The OpenCyc Knowledge Base (Optional Material)

You will see something new in this section. After loading the OpenCyc data as RDF into Apache Fuseki and exploring the data we will use the SPARQL SERVICE operator to combine data from our local server with the public DBPedia Knowledge Graph.

The OpenCyc Knowledge Base is no longer supported by the Cyc corporation (they sell commercial versions). I still find this knowledge base useful and here we use a version that has been converted to RDF data.

Adam Sanchez has a [GitHub repository that contains the OpenCyc OWL/RDF files](https://github.com/asanchez75/opencyc). While I try to make this section self-contained and interesting to read through if you want to experiment with the latest OpenCyc 4.0 OWL/RDF dataset then [download this file](https://www.amazon.com/clouddrive/share/urtlDhQbmeMz24TUNED3KiyzrqOlMYZ5gdLpTTSdcFR).

I wrote a blog article in 2014 [Using OpenCyc RDF/OWL data in StarDog 
](https://mark-watson.blogspot.com/2014/07/using-opencyc-rdfowl-data-in-stardog.html) that showed how to import the OpenCyc OWL/RDF files into the commercial RDF datastore Stardog. Here I do much the same thing using Apache Jena/Fuseki but we will dive in deeper.

If you have downloaded the latest OpenCyc OWL file then it can be loaded by:

```
./fuseki-server --file /Users/markw/OpenCyc_owl_rdf/opencyc-latest.owl /opencyc
```

As I did in my blog article, we start by using a SPARQL query that contains "Clinton" as the object in a triple and we get 207 triples from this query:

```sparql
SELECT ?s ?p ?o WHERE { ?s ?p ?o FILTER(REGEX(?o, "Clinton")) } LIMIT 500
```

This triple identifies the OpenCyc subject for Hilliary Clinton:

```rdf
<http://sw.opencyc.org/concept/Mx4rvV7SqpwpEbGdrcN5Y29ycA>
    <http://sw.cyc.com/CycAnnotations_v1#label>
    "HillaryClinton"@en .
```

```sparql
SELECT ?p ?o
WHERE {
  <http://sw.opencyc.org/concept/Mx4rvV7SqpwpEbGdrcN5Y29ycA>
  ?s ?p
} LIMIT 500
```

![SPARQL query results exported as TSV](opencychc.png)

Of particular use is the matched result:

```tsv
?p	?o
<http://www.w3.org/2002/07/owl#sameAs>
  <http://dbpedia.org/resource/Hillary_Rodham_Clinton>
```

This lets us combine data from OpenCyc with DBPedia (limit to just 2500 results). This is not a good example since we are in no way tying togeter data from OpenCyc to DBPedia (we will combine the results later), rather we are just doing two separate queries. :

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
## Test client for Apache Jena Fuselki server on localhost

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

The output is:

```
$ python opencyc_example_1.py
{'dbpedia_uri': {'type': 'uri',
                 'value': 'http://dbpedia.org/resource/Hillary_Rodham_Clinton'},
 'label': {'type': 'literal', 'value': 'Hillary Clinton', 'xml:lang': 'en'}}
```

Let's now link local SPARQL results from OpenCyc with information from DBPedia. We replace the **queryString** variable value in the last code listing with:

```sparql
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbo: <http://dbpedia.org/ontology/>
SELECT *
WHERE {
    <http://sw.opencyc.org/concept/Mx4rvV7SqpwpEbGdrcN5Y29ycA>  rdfs:label ?label
    FILTER (lang(?label) = 'en') .
    <http://sw.opencyc.org/concept/Mx4rvV7SqpwpEbGdrcN5Y29ycA>
      <http://www.w3.org/2002/07/owl#sameAs> ?dbpedia_uri
      filter(strstarts(str(?dbpedia_uri), "http://dbpedia.org/resource")) .
  SERVICE <http://dbpedia.org/sparql?timeout=30000> {
      ?dbpedia_uri  ?dbpedia_property ?dbpedia_object  .
  }
}
LIMIT 4
```

The output is:

```json
$ python opencyc_example_2.py
{'dbpedia_object': {'type': 'literal',
                    'value': 'Hillary Rodham Clinton',
                    'xml:lang': 'en'},
 'dbpedia_property': {'type': 'uri',
                      'value': 'http://www.w3.org/2000/01/rdf-schema#label'},
 'dbpedia_uri': {'type': 'uri',
                 'value': 'http://dbpedia.org/resource/Hillary_Rodham_Clinton'},
 'label': {'type': 'literal', 'value': 'Hillary Clinton', 'xml:lang': 'en'}}
{'dbpedia_object': {'type': 'literal',
                    'value': 'Hillary Clinton',
                    'xml:lang': 'ca'},
 'dbpedia_property': {'type': 'uri',
                      'value': 'http://www.w3.org/2000/01/rdf-schema#label'},
 'dbpedia_uri': {'type': 'uri',
                 'value': 'http://dbpedia.org/resource/Hillary_Rodham_Clinton'},
 'label': {'type': 'literal', 'value': 'Hillary Clinton', 'xml:lang': 'en'}}
{'dbpedia_object': {'type': 'literal',
                    'value': 'Hillary Clintonová',
                    'xml:lang': 'cs'},
 'dbpedia_property': {'type': 'uri',
                      'value': 'http://www.w3.org/2000/01/rdf-schema#label'},
 'dbpedia_uri': {'type': 'uri',
                 'value': 'http://dbpedia.org/resource/Hillary_Rodham_Clinton'},
 'label': {'type': 'literal', 'value': 'Hillary Clinton', 'xml:lang': 'en'}}
```

We can use the SPARQL OPTIONAL operator to match data patterns that may or may not exist. OPTIONAL is a binary operator that combines two graph patterns:

```sparql
SELECT *
WHERE {
    <http://sw.opencyc.org/concept/Mx4rvV7SqpwpEbGdrcN5Y29ycA>  rdfs:label ?label
    FILTER (lang(?label) = 'en') .
    <http://sw.opencyc.org/concept/Mx4rvV7SqpwpEbGdrcN5Y29ycA>
      <http://www.w3.org/2002/07/owl#sameAs> ?dbpedia_uri
      filter(strstarts(str(?dbpedia_uri), "http://dbpedia.org/resource")) .
  SERVICE <http://dbpedia.org/sparql?timeout=15000> {
    ?dbpedia_uri  rdfs:label ?dbpedia_label
      FILTER (lang(?dbpedia_label) = 'en') .
    OPTIONAL {
       ?dbpedia_uri  rdfs:comment ?dbpedia_comment
       FILTER (lang(?dbpedia_comment) = 'en')
    } .
  }
}
```

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