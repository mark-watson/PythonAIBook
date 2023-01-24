### Using Swi-Prolog for the Semantic Web, Fetching Web Pages, and Handling JSON

In several ways reading the historic Scientific American Article from 2001 [The Semantic Web - A new form of Web content that is meaningful to computers will unleash a revolution of new possibilities](https://www-sop.inria.fr/acacia/cours/essi2006/Scientific%20American_%20Feature%20Article_%20The%20Semantic%20Web_%20May%202001.pdf) by Tim Berners-Lee, James Hendler, and Ora Lassila changed my life. I spent a lot of time experimenting with the Swi-Prolog **semweb** library that I found to be the easiest way to experiment with RDF. We will cover the open source Apache Jena/Fuseki RDF data store and query engine in a later chapter. The Common Lisp and Semantic Web tools company [Franz Inc.](https://franz.com)very kindly subsidized my work writing two Semantic Web books that cover Common Lisp, Java, Clojure, and Scala (available as downloadable PDFs on my [web site](https://markwatson.com)). Writing these books lead directly to being invited to work at Google as a consultant using their Knowledge Graph.

Here I mention how to load the **semweb** library and refer you to the [library documentation](https://www.swi-prolog.org/pldoc/doc_for?object=section(%27packages/semweb.html%27)):

```prolog
?- use_module(library(semweb/rdf_db)).
?- use_module(library(semweb/sparql_client)).
?- sparql_query('select * where { ?s ?p "Amsterdam" }',
                Row,
                host('dbpedia.org'), path('/sparql/')]).
```

The output is:

```
Row = row('http://dbpedia.org/resource/Anabaptist_riot',
          'http://dbpedia.org/ontology/combatant') ;
Row = row('http://dbpedia.org/resource/Thirteen_Years\'_War_(1454–1466)',
          'http://dbpedia.org/ontology/combatant') ;
Row = row('http://dbpedia.org/resource/Women\'s_Euro_Hockey_League',
          'http://dbpedia.org/ontology/participant').
```
