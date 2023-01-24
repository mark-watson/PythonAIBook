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

~~~
Row = row('http://dbpedia.org/resource/Anabaptist_riot',
          'http://dbpedia.org/ontology/combatant') ;)
Row = row('http://dbpedia.org/resource/Thirteen_Years_War_(1454-1466)',
          'http://dbpedia.org/ontology/combatant') ;)
Row = row('http://dbpedia.org/resource/Womens_Euro_Hockey_League',
          'http://dbpedia.org/ontology/participant') ;)
~~~

Prolog is a flexible and general purpose language that is used to write compilers, handle text processing, etc. One common thing that I need no matter what programming language I use is fetching content from the web. Here is a simple example you can try to perform a HTTP GET operation and echo the fetched data to standard output:

```prolog
use_module(library(http/http_open)).
http_open('https://markwatson.com', In, []),
   copy_stream_data(In, user_output),
   close(In).
```

Similarly, handling JSON data is a common task so here is an example for doing that:

```prolog
?- use_module(library(http/json)). % to enable json_read_dict/2
?- FPath = 'test.json', open(FPath, read, Stream), json_read_dict(Stream, Dicty).
```

### Python and Swi-Prolog Interop

You need to install the Python bridge library that is supported by the Swi-Prolog developers:

     pip install swiplserver

I copied a Prolog program from the [Swi-Prolog documentation](https://www.swi-prolog.org/pldoc/man?section=clpfd-n-queens) to calculate how to eight queens on a chess board in such a way that no queen can capture another queen. Here I get three reults by entering the semicolon key to get another answer or the period key to stop:

```prolog
$ swipl

?- use_module(library(clpfd)). /* constraint satisfaction library */
true.

?- [n_queens]. /* load the file n_queens.pl */
true.

?- n_queens(8, Qs), label(Qs).
Qs = [1, 5, 8, 6, 3, 7, 2, 4] ;
Qs = [1, 6, 8, 3, 7, 4, 2, 5] ;
Qs = [1, 7, 4, 6, 8, 2, 5, 3] .
```

The source code to **n_queens.pl** is included in the examples directory **swi-prolog** for this book. This example was copied from the Swi-Prolog documentation. Here is Python code to use this Prolog example:

```python
from swiplserver import PrologMQI
from pprint import pprint

with PrologMQI() as mqi:
    with mqi.create_thread() as prolog_thread:
        prolog_thread.query("use_module(library(clpfd)).")
        prolog_thread.query("[n_queens].")
        result = prolog_thread.query("n_queens(8, Qs), label(Qs).")
        pprint(result)
        print(len(result))
```

We can run this example to see all 92 possible answers:
