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
