# Symbolic AI (Optional Material)

When I started my paid career as an AI practitioner in 1982 my company bought me a Xerox 1108 Lisp Machine and I spent every spare moment I had working through two books by Patrick Winston that I had purchased a few years earlier: "Lisp" and "Artificial Intelligence." This material was mostly what is now called symbolic AI or good old fashioned AI (GOFAI). The material in this chapter is optional for modern AI developers but I recently wrote the Python examples listed below when I was thinking of how different knowledge representation is today compared to 40 years ago. Except for the material using Python + Swi-Prolog, and Python + the MiniZinc constraint satisfaction system there is nothing in this chapter that I would consider using today for work but you might enjoy the examples anyway. After this short chapter we will bear down on deep learning, information organization using RDF and property graph data stores.

I do not implement three examples in this chapter in "pure Python," rather, I use the Python bindings for three well known tools that are implemented in C/C++:

- Swi-Prolog is a Prolog system that has many available libraries for a wide variety of tasks.
- Soar Cognitive Architecture is a flexible and general purpose reasoning and knowledge management system for building intelligent software agents.
- MiniZinc is a powerful Constraint Satisfaction System.

The material in this chapter is optional for the modern AI practitioner but I hope you find it interesting.

We will start with one "pure Python" example in the next section.

## Implementing Frame Data Structures in Python

Most of my learning experiments and AI projects in the early 1980s were built from scratch in Common Lisp and nested frame data structures were a common building block. Here we allow three types of data to be stored in frames:

- Numbers
- Strings
- Other frames

We write a general Python class **Frame** that supports creating frames and converting a frame, including deeply nested frames, into a string representation. We also write a simple Python class **BookShelf** as a container for frames that supports searching for any frames containing a string value.

```python
# Implement Lisp-like frames in Python

class Frame():
    frame_counter = 0
    def __init__(self, name = ''):
        Frame.frame_counter += 1
        self.objects = []
        self.depth = 0
        if (len(name)) == 0:
            self.name = f"Frame:{Frame.frame_counter}"
        else:
            self.name = f'"{name}"'

    def add_subframe(self, a_frame):
        a_frame.depth = self.depth + 1
        self.objects.append(a_frame)

    def add_number(self, a_number):
        self.objects.append(a_number)

    def add_string(self, a_string):
        self.objects.append(a_string)

    def __str__(self):
        indent = " " * self.depth * 2
        ret = indent + f"<Frame {self.name}>\n"
        for frm in self.objects:
            if isinstance(frm, (int, float)):
                ret = ret + indent + '  ' + f"<Number {frm}>\n"
            if isinstance(frm, str):
                ret = ret + indent + '  ' + f'<String "{frm}">\n'
            if isinstance(frm, Frame):
                ret = ret + frm.__str__()
        return ret

f1 = Frame()
f2 = Frame("a sub-frame")
f1.add_subframe(f2)
f1.add_number(3.14)
f2.add_string("a string")
print(f1)
f2.add_subframe(Frame('a sub-sub-frame'))
print(f1)

class BookShelf():

    def __init__(self, name = ''):
        self.frames = []
    
    def add_frame(self, a_frame):
        self.frames.append(a_frame)
    
    def search_text(self, search_string):
        ret = []
        for frm in self.frames:
            if frm.__str__().index(search_string):
                ret.append(frm)
        return ret
    
bookshelf = BookShelf()
bookshelf.add_frame(f1)
search_results = bookshelf.search_text('sub')
print("Search results: all frames containing 'sub':")
for rs in search_results:
    print(rs)
```

Here is some output:

```bash
$ python
>>> from frame import Frame Bookshelf
>>> f1 = Frame()
>>> f2 = Frame("a sub-frame")
>>> f1.add_subframe(f2)
>>> f1.add_number(3.14)
>>> f2.add_subframe(Frame('a sub-sub-frame'))
>>> print(f1)
<Frame Frame:4>
  <Frame "a sub-frame">
    <Frame "a sub-sub-frame">
  <Number 3.14>
>>> bookshelf = BookShelf()
>>> bookshelf.add_frame(f1)
>>> search_results = bookshelf.search_text('sub')
>>> for rs in search_results:
...   print(rs)
... 
<Frame Frame:4>
  <Frame "a sub-frame">
    <Frame "a sub-sub-frame">
  <Number 3.14>
```

I would start with implementing a simple frame library and extend it for the two types of applications that I worked on: Natural Language Processing (NLP) and planning systems.

I no longer use frames preferring the use off the shelf graph databases that we will cover in a later chapter. Graphs can represent a wider range of data representations because frames represent tree structured data and graphs are more general purpose than trees.

## Use Predicate Logic by Calling Swi-Prolog

Please skip this section if you either don't know how to program in Prolog or if you have no interest in learning Prolog. I have a writing project for a book titled Prolog for AI applications that is a work in progress. When that book is released I will add a link here. Before my Python book is released Sheila McIlraith has a [Swi-Prolog tutorial](https://www.cs.toronto.edu/~sheila/324/f05/tuts/swi.pdf) written for her students that is a good starting point and you can use the official [Swi-Prolog manual](https://www.swi-prolog.org/pldoc/doc_for?object=manual) for specific information. I will make this section self-contained if you just want to read the material without writing any Python + Prolog applications.

You can start by reading the documentation for [setting up Swi-Prolog so it can be called from Python](https://www.swi-prolog.org/pldoc/man?section=mqi-python-installation). 

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

```
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


## Soar Cognitive Architecture

[Soar](https://soar.eecs.umich.edu/) is a flexible and general purpose reasoning and knowledge management system for building intelligent software agents. The Soar project is a classic AI tool and has the advantage of being kept up to date. As I write this the [Soar GitHub repositoryt](https://github.com/SoarGroup/Soar) was just updated a few days ago.

I am writing this material many years after my previous use of Soar. My primary reference for preparing the following material is the short paper [Introduction to the Soar Cognitive Architecture](https://arxiv.org/pdf/2205.03854.pdf) by John E. Laird. For self-study you can start at the [Soar Tutorial Page](https://soar.eecs.umich.edu/articles/downloads/soar-suite/228-soar-tutorial-9-6-0) that provides a download for an eight part tutorial in separate PDF files, binary Soar executables for Linux, Windows, and macOS, and all of the code for the tutorials.

I consider Soar to be important because it proposes and implements a general purpose cognitive architecture. A warning to the reader: Soar has a steep learning curve and there are simpler frameworks for solving practical problems. Later we will look at an example from the Soar Tutorial for the classic "blocks world" problem of moving blocks on a table subject to constraints like not being alowed to move a block if it has another object on top of it. Solving this fairly simple problem requires about 400 lines of Soar source code.

### Background Theory

The design goals for the Soar Cognitive Architecture (which I will usually refer to as Soar) is to provide ["fixed structures, mechanisms, and representations"](https://soar.eecs.umich.edu/workshop/30/laird2.pdf) to develop human level behavior across a wide range of tasks. There is a commercial company [Soar.com](https://try.soar.com) that uses Soar for commercial and government projects.

We will cover [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (which I will usually refer to as RL) in a later chapter but there is similar infrastructure supported by both Soar and RL: a simulated environment, data representing the state of the environment, and possible actions that can be performed in the environment that change the state.

There are two main components of the Soar architecture:

- Working Memory - this is the data that specifies the current state of the environment. Actions in the environment change the data in working memory, either by modifiaction, addition, or deletion. At the risk of over-anthropomorphism, consider this like human short term meory.
- Production Memory - this data is a form of production rules where the left-hand side of rules consist of patterns that if matched against worign memory, the the actions on the right-hand side of a rule are executed. Consider these production rules as long-term memory.

Both Soar working memory and production memory are symbolic data. As a contrast, data in RL is numeric, mostly tensors. This symbolic data comprises goals (G), problem spaces (PS), states (S) and operators (O).

![Soar Operator transitioning from one state to another](Soararchitecture-transitions.png)

### Setup Python and Soar Development Environment

It will take you a few minutes to install Soar on your system and create the Python bindings. Start by cloning the GitHub repository and run the install script from the top directory:

```bash
python scons/scons.py sml_python
```

If you want all language bindings replace **sml_python** with **all**. Change directory to the **out** subdirectory and note the directory path. On my system:

```bash
$ pwd
/Users/markw/SOAR/Soar/out
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/markw/SOAR/Soar/out
$ export PYTHONPATH=$PYTHONPATH:/Users/markw/SOAR/Soar/out
```

I will present here a simple example and explain a subset of the capabilities of Soar. When we are done here you can reference a recent paper by Neha Rajan and 2Sunderrajan Srinivasan [Exploring Learning Capability of an Agent in SOAR: Using 8- Queens Problem](https://thescipub.com/pdf/jcssp.2020.642.650.pdf) for a complete example using Soar for cognitive modeling and a more complex example.

### A minimal Soar Tutorial

I am presenting a minimal introduction to Soar and we will later provide an example of Python and Soar interop for the purpose of introducing you to Soar. If this material looks interesting then I encourage you to work through the [Soar Tutorial Page](https://soar.eecs.umich.edu/articles/downloads/soar-suite/228-soar-tutorial-9-6-0).

Soar supports a rule language that uses the highly efficient Rete algorithm (optimized for huge numbers of rules, less optimized for large working memories). Let's look at a sample rule from Chapter 1 (first PDF file) of the Soar tutorial:

```python
sp {hello-world
   (state <s> ^type state)
-->
   (write |Hello World|)
   (halt)
}
```

The token **sp** on line 1 stands for Soar Production. Rules are enclosed in **{** and **}**. The name of this rule is the symbol **hello-world**. In the tutorial you will usually see rule names partitioned using the characters **\*** and **-**. Rules have a "left side" and a "right side", separated by **-->**. If all of the left side patterns match working memory elments then the right-hand side actions are executed.

The following figure is from the Soar tutoral and shows two blocks stacked on top of each other. The bottom block rests on a table:

![From the Soar Tutorial: two stacked blocks sitting on a table](Soarblocks.png)

This figure represents state **s1** that is a root of the graph also containing blocks named **b1** and **b2** as well as the table named **t1**. The blocks and table all have attributes **^color**, **^name**, and **^type**. The blocks also have the optional attribute **^ontop**.

Rule right-hand side actions can modify, delete, or add working memory data. For example, a left-hand side matching the attribute values for block **b1** could modify its **^ontop** attribute from the value **b2** to the table named **t1**.

### Example Soar System With Python Interop

We will use the simplest blocks world example in the Soar Tutorial in our Python interop example. In the examples directories in the Soar Tutorial, this example is spread through eight source files. I have copied them to a single file **Soar/blocks-world/bw.soar** in the GitHub repository for this book.

```python
import Python_sml_ClientInterface as sml

def callback_debug(mid, user_data, agent, message):
    print(message)

if __name__ == "__main__":
    soar_kernel = sml.Kernel.CreateKernelInCurrentThread()
    soar_agent = soar_kernel.CreateAgent("agent")
    soar_agent.RegisterForPrintEvent(sml.smlEVENT_PRINT, callback_debug, None) # no user data
    soar_agent.ExecuteCommandLine("source bw.soar")
    run_result=soar_agent.RunSelf(50)
    soar_kernel.DestroyAgent(soar_agent)
    soar_kernel.Shutdown()
```

Run this example:

```bash

$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/markw/SOAR/Soar/out
$ export PYTHONPATH=$PYTHONPATH:/Users/markw/SOAR/Soar/out
$ python bw.py                                     

     1:    O: O1 (initialize-blocks-world)
Five Blocks World - just move blocks.
The goal is to get EDBCA.
AC
DEB

     2:    O: O8 (move-block)
 Apply O8: move-block(C,table)P10*apply*move-block*internal
A
DEB
C

     3:    O: O7 (move-block)
 Apply O7: move-block(B,table)P10*apply*move-block*internal
DE
B
C
A

     4:    O: O17 (move-block)
 Apply O17: move-block(E,table)P10*apply*move-block*internal
D
B
C
A
E

     5:    O: O25 (move-block)
 Apply O25: move-block(D,E)P10*apply*move-block*internal
B
C
A
ED

     6:    O: O27 (move-block)
 Apply O27: move-block(C,D)P10*apply*move-block*internal
B
EDC
A

     7:    O: O10 (move-block)
 Apply O10: move-block(B,C)P10*apply*move-block*internal
EDCB
A

     8:    O: O11 (move-block)
 Apply O11: move-block(A,B)P10*apply*move-block*internal
EDCBA
Goal Achieved (five blocks).
System halted.
Interrupt received.This Agent halted.
```

TBD


## Constraint Programming with MiniZinc and Python

As with Soar, our excursion into constraint programming will be brief, hopefully enough to introduce you to a new style of programming though a few examples.

You may want to use the [MiniZinc Python](https://minizinc-python.readthedocs.io/en/latest/getting_started.html) documentation as a reference for the Python interface and [The MiniZink Handbook](https://www.minizinc.org/doc-2.6.4/en/index.html) as a reference to the MiniZinc language and its use.

### Installation and Setup for MiniZinc and Python

You need to first install the MiniZinc system. For macOS this can be done with **brew install minizinc** or can be [installed from source code on macOs and Linux](https://www.minizinc.org/doc-2.5.5/en/installation_detailed_linux.html). The Python interface can be installed with **pip install minizinc**.

The following figure shows the MiniZincIDE with simple constraint satisfaction problem:

![MiniZincIDE with simple constraint satisfaction problem](MiniZincIDE.png)

When I installed **minizinc** on macOS with **brew**, the solver **coinbc** was installed automatically so that is what we use here. Here is the MiniZinc source file **test1.mzn**:

```basic
int: n;
int: m;
var 1..n: x;
var 1..n: y;
constraint x+y = n;
constraint x*y = m;
```

There are several possible solvers to use with MiniZinc. When I install on macOS using *brew* the solver "coinbc" is available. When I install **sudo apt install minizinc** on Ubuntu Linux, the solver "gecode" is available.

Notice that we don't set values for the constants **n** and **m** as we did when using MiniZincIDE. We instead set them in Python code before calling the solver:

```python
from minizinc import Instance, Model, Solver

coinbc = Solver.lookup("coinbc")

test1 = Model("./test1.mzn")
instance = Instance(coinbc, test1)
instance["n"] = 30
instance["m"] = 200

result = instance.solve()
print(result)
print(result["x"])
print(result["y"])
```

The result is:

```bash
$ python test1.py
Solution(x=20, y=10, _checker='')
20
10
```

Let's look at a more complex example: on the map of the USA, the states neighboring each other are colored differently than their adjoining states. We use integers to represent colors and the mapping of numbers to colors is unimportant. Here is a partial listing of us_states.mzn:

```
int: nc = 3; %% needs to be 4 to solve this problem

var 1..nc: alabama;
var 1..nc: alaska;
var 1..nc: arizona;
var 1..nc: arkansas;
var 1..nc: california;
 ...
constraint alabama != florida;
constraint alabama != georgia;
constraint alabama != mississippi;
constraint alabama != tennessee;
constraint arizona != california;
constraint arizona != colorado;
constraint arizona != nevada;
constraint arizona != new_mexico;
constraint arizona != utah;
 ...
solve satisfy;
```

The output is:

```bash
 $ minizinc --solver coinbc us_states.mzn
=====UNSATISFIABLE=====
```

So we need more than three colors. Let's try **int: nc = 4;**:

```
$ minizinc --solver coinbc us_states.mzn
alabama = 2;
alaska = 1;
arizona = 3;
arkansas = 4;
california = 4;
colorado = 4;
connecticut = 2;
delaware = 4;
 ...
```

Here is a Python script **us_states.py** that uses this model and picks out the assigned color indices from the solution object:

``python
from minizinc import Instance, Model, Solver

coinbc = Solver.lookup("coinbc")

model = Model("./us_states.mzn")
instance = Instance(coinbc, model)
instance["nc"] = 4 # solve for a maximum of 4 colors

result = instance.solve()
print(result)
all_states = list(result.__dict__['solution'].__dict__.keys())
all_states.remove('_checker')
print(all_states)
for state in all_states:
    print(f" {state} \t: \t{result[state]}")
```

Here is some of the output:

```bash
$ python us_states.py
Solution(alabama=2, alaska=1, arizona=3, arkansas=4, california=4, colorado=4, connecticut=2, delaware=4, florida=1, georgia=4, hawaii=1, idaho=4, ... ]
 alabama 	: 	2
 alaska 	: 	1
 arizona 	: 	3
 arkansas 	: 	4
 ...
 wisconsin 	: 	1
 wyoming 	: 	3
```

## Good Old Fashioned Symbolic AI Wrapup

As a practical matter almost all of my work in the last ten years used either deep learning or was comprised of semantic web and linked data projects. While the material in this chapter is optional for the modern AI practitioner, I still find using MiniZinc for constraint programming and Prolog to be useful. I included the material for the Soar cognitive architecture because I both find it interesting and I believe the any future development of "real AI" (or AGI) will involve hybrid approaches and there are many good ideas in the Soar implementation.