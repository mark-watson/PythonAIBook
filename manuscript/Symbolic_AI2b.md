### Using Swi-Prolog for the Semantic Web, Fetching Web Pages, and Handling JSON

In several ways reading the historic Scientific American Article from 2001 [The Semantic Web - A new form of Web content that is meaningful to computers will unleash a revolution of new possibilities](https://www-sop.inria.fr/acacia/cours/essi2006/Scientific%20American_%20Feature%20Article_%20The%20Semantic%20Web_%20May%202001.pdf) by Tim Berners-Lee, James Hendler, and Ora Lassila changed my life. I spent a lot of time experimenting with the Swi-Prolog **semweb** library that I found to be the easiest way to experiment with RDF. We will cover the open source Apache Jena/Fuseki RDF data store and query engine in a later chapter. The Common Lisp and Semantic Web tools company [Franz Inc.](https://franz.com)very kindly subsidized my work writing two Semantic Web books that cover Common Lisp, Java, Clojure, and Scala (available as downloadable PDFs on my [web site](https://markwatson.com)). Writing these books lead directly to being invited to work at Google as a consultant using their Knowledge Graph.

Here I mention how to load the **semweb** library and refer you to the [library documentation](https://www.swi-prolog.org/pldoc/doc_for?object=section(%27packages/semweb.html%27)):

```prolog
?- use_module(library(semweb/rdf_db)).
?- use_module(library(semweb/sparql_client)).
?- sparql_query('select * where { ?s ?p "Amsterdam" }', Row,                                  host('dbpedia.org'), path('/sparql/')]).
Row = row('http://dbpedia.org/resource/Anabaptist_riot', 'http://dbpedia.org/ontology/combatant') ;
Row = row('http://dbpedia.org/resource/Thirteen_Years\'_War_(1454–1466)', 'http://dbpedia.org/ontology/combatant') ;
Row = row('http://dbpedia.org/resource/Women\'s_Euro_Hockey_League', 'http://dbpedia.org/ontology/participant').

?- 
```

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

```prolog
 $ p n_queens.py
[{'Qs': [1, 5, 8, 6, 3, 7, 2, 4]},
 {'Qs': [1, 6, 8, 3, 7, 4, 2, 5]},
 {'Qs': [1, 7, 4, 6, 8, 2, 5, 3]},
 {'Qs': [1, 7, 5, 8, 2, 4, 6, 3]},
 {'Qs': [2, 4, 6, 8, 3, 1, 7, 5]},
 ...
 ]
 92
 ```

Here I call the Swi-Prolog system synchronously; that is, each call to **prolog_thread.query** waits until the answers are ready. If you can also run long running queries asynchronously then please read the [instructions online](https://www.swi-prolog.org/pldoc/doc_for?object=section(%27packages/mqi.html%27)).

In the last example we simply ran an existing Prolog program from Python. Now let's look at an example for asserting facts and Prolog rules from a Python script. First we look at a simple example of Prolog rules, asserting facts, and applying rules to facts. We will use the Prolog source file **family.pl**:

```prolog
parent(X, Y) :- mother(X, Y).
parent(X, Y) :- father(X, Y).
grandparent(X, Z) :-
  parent(X, Y),
  parent(Y, Z).
```

Before using a Python script, let's run an example in the Swi-Prolog REPL:

```bash
$ swipl
?- [family].
true.

?- assertz(mother(irene, ken)).
true.

?- assertz(father(ken, ron)).
true.

?- grandparent(A,B).
A = irene,
B = ron ;
false.
```

Now we can write a Python script **family.py** that loads the Prolog rules file **family.pl**, asserts facts, run Prolog queries, and get the results back to the Python script:

```python
from swiplserver import PrologMQI
from pprint import pprint

with PrologMQI() as mqi:
    with mqi.create_thread() as prolog_thread:
        prolog_thread.query("[family].")
        print("Assert a few initial facts:")
        prolog_thread.query("assertz(mother(irene, ken)).")
        prolog_thread.query("assertz(father(ken, ron)).")
        result = prolog_thread.query("grandparent(A, B).")
        pprint(result)
        print(len(result))
        print("Assert another test fact:")
        prolog_thread.query("assertz(father(ken, sam)).")
        result = prolog_thread.query("grandparent(A, B).")
        pprint(result)
        print(len(result))
```

The output looks like:

```bash
$ python family.py
Assert a few initial facts:
[{'A': 'irene', 'B': 'ron'}]
1
Assert another test fact:
[{'A': 'irene', 'B': 'ron'}, {'A': 'irene', 'B': 'sam'}]
2
```

Swi-Prolog is still under active development (the project was started in 1985) and used for new projects. If the declarative nature of Prolog programming appeals to you then I urge you to take the time to integrate Swi-Prolog into one of your Python-based projects.

## Swi-Prolog and Python Deep Learning Interop

Good usecases for Python and Prolog applications involve using Pyhton code to fetch and process data that is imported to Prolog. Applications can then use Prolog's reasoning and other capabilities.

Here we look at a simple example that:

- Uses the Firebase Hacker News APIs to fetch most recent stories.
- Uses the spaCy library deep learning based NLP model to identify organization and peoples names from articles.
- Assert as Prolog facts terms like **organization(Name, URI)***.

We will cover the spaCy library in depth later. For the purposes of this example, please consider spaCy as a "black box."

The following listing shows the Python script **hackernews.py**:

```python
from urllib.request import Request, urlopen
import json
from bs4 import BeautifulSoup
from pprint import pprint

from swiplserver import PrologMQI

import spacy
try:
    spacy_model = spacy.load("en_core_web_sm")
except:
  from os import system
  system("python -m spacy download en_core_web_sm")
  spacy_model = spacy.load('en_core_web_sm')

LEN = 100 # larger amount of text is more expensive for OpenAI APIs

def get_new_stories(anAgent={'User-Agent': 'PythonAiBook/1.0'}):
    req = Request("https://hacker-news.firebaseio.com/v0/newstories.json",
                  headers=anAgent)
    httpResponse = urlopen(req)
    data = httpResponse.read()
    #print(data)
    ids = json.loads(data)
    #print(ids)
    # just return the most recent 3 stories:
    return ids[0:3]

ids = get_new_stories()

def get_story_data(id, anAgent={'User-Agent': 'PythonAiBook/1.0'}):
    req = Request(f"https://hacker-news.firebaseio.com/v0/item/{id}.json",
                  headers=anAgent)
    httpResponse = urlopen(req)
    return json.loads(httpResponse.read())

def get_story_text(a_uri, anAgent={'User-Agent': 'PythonAiBook/1.0'}):
    req = Request(a_uri, headers=anAgent)
    httpResponse = urlopen(req)
    soup = BeautifulSoup(httpResponse.read(), "html.parser")
    return soup.get_text()

def find_entities_in_text(some_text):
    def clean(s):
        return s.replace('\n', ' ').strip()
    doc = spacy_model(some_text)
    return map(list, [[clean(entity.text), entity.label_] for entity in doc.ents])

import re
regex = re.compile('[^a-z_]')

def safe_prolog_text(s):
    s = s.lower().replace(' ','_').replace('&','and').replace('-','_')
    return regex.sub('', s)

for id in ids:
    story_json_data = get_story_data(id)
    #pprint(story_json_data)
    if story_json_data != None and 'url' in story_json_data:
        print(f"Processing {story_json_data['url']}\n")
        story_text = get_story_text(story_json_data['url'])
        entities = list(find_entities_in_text(story_text))
        #print(entities)
        organizations =
            set([safe_prolog_text(name)
                 for [name, entity_type] in entities if entity_type == "ORG"])
        people =
            set([safe_prolog_text(name)
                 for [name, entity_type] in entities if entity_type == "PERSON"])
        with PrologMQI() as mqi:
            with mqi.create_thread() as prolog_thread:
                for person in people:
                    s = f"assertz(person({person},
                                  '{story_json_data['url']}'))."
                    #print(s)
                    try:
                        prolog_thread.query(s)
                    except:
                        print(f"Error with term: {s}")
                for organization in organizations:
                    s = f"assertz(organization({organization}, '{story_json_data['url']}'))."
                    #print(s)
                    try:
                        prolog_thread.query(s)
                    except:
                        print(f"Error with term: {s}")
                try:
                    result = prolog_thread.query("organization(Organization, URI).")
                    pprint(result)
                except:
                    print("No results for organizations.")
                try:
                    result = prolog_thread.query("person(Person, URI).")
                    pprint(result)
                except:
                    print("No results for people.")
```

Here is some example output:

```json
 {'Organization': 'bath_and_beyond_inc',
  'URI': 'https://bedbathandbeyond.gcs-web.com/news-releases/news-release-details/bed-bath-beyond-inc-provides-business-update'},
 {'Person': 'ryan_holiday',
  'URI': 'https://www.parttimetech.io/p/what-do-you-really-want'},
{'Organization': 'nasa_satellite',
 {'Organization': 'nih',
  'URI': 'https://nap.nationalacademies.org/catalog/26424/measuring-sex-gender-identity-and-sexual-orientation'},
  'URI': 'https://landsat.gsfc.nasa.gov/article/now-then-landsat-u-s-mosaics/'},
{'Organization': 'the_national_academies',
'URI': 'https://nap.nationalacademies.org/catalog/26424/measuring-sex-gender-identity-and-sexual-orientation'},
{'Organization': 'microsoft',
  'URI': 'https://invention.si.edu/susan-kare-iconic-designer'},
{'Person': 'susan_kare',
  'URI': 'https://invention.si.edu/susan-kare-iconic-designer'},
```

Later we will use deep learning models to summarize text and other NLP tasks. This example could be extended for defining Prolog terms in the form ***summary(URI, "...")***. Other application ideas might be to use Python scripts for:

- Collect stock market data for a rule-based Prolog reasoner for stock purchase selections.
- A customer service chatbot that is mostly written in Python could be extended by using a Prolog based reasoning system.
