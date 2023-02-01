# Getting Setup To Use Graph and Relational Databases

I use several types of data stores in my work but for the purposes of this book generally and for this chapter specifically we can explore interesting ideas and lay a foundation for examples later in this book for using graph and relational databases using just three platforms:

- [Apache Jena Fuseki](https://jena.apache.org/documentation/fuseki2/) for RDF data storage, [SPARQL queries](https://jena.apache.org/tutorials/sparql.html), and Fuseki's web interface to explore datasets.
- [Neo4j Community Edition](https://neo4j.com/download-center/#community) for a transactional disk-based graph database, the [Cypher query language](https://neo4j.com/docs/cypher-manual/current/), and the web interface to explore datasets. If you prefer you can alternatively use [Memgraph](https://memgraph.com) that is fairly compatible with Neo4j.
- [SQLite](https://www.sqlite.org/index.html) for transactional relational data storage and the [SQL query language](https://en.wikipedia.org/wiki/SQL).

The next chapter covers RDF and the SPARQL query language in some detail.

In technical terms, knowledge representation using graph and relational databases involves the use of graph structures and relational data models to represent and organize knowledge in a structured, computationally efficient, and easily accessible way.

A graph structure is a collection of nodes (also known as vertices) and edges (also known as arcs) that connect the nodes. Each node and edge in a graph can have properties, such as labels and attributes which provide information about the entities they represent. Graphs can be used to represent knowledge in a variety of ways, such as through semantic networks and using ontologies to define terms, classes, types, etc.

Relational databases, on the other hand, use a tabular data model to represent knowledge. The basic building block of a relational database is the table, which is a collection of rows (also known as tuples) and columns (also known as attributes). Each row represents an instance of an entity, and the columns provide information about the properties of that entity. Relationships between entities can also be represented by foreign keys, which link one table to another.

Combining these two technologies, knowledge can be represented as a graph of interconnected entities, where each entity is stored in a relational database table and connected to other entities through relationships represented by edges in the graph. This allows for efficient querying and manipulation of knowledge, as well as the ability to integrate and reason over large amounts of information.

## The Apache Jena Fuseki RDF Datastore and SPARQL Query Server

I use several RDF datastores in my work (Franz AllegroGraph, Stardog, GraphDB, and Blazegraph) but I particularly like Apache Jena Fuseki (that I will often just call Fuseki) because it is open source, it follows new technology like RDF* and SPARQL*, and has a simple and easy to use web interface). Most of our experiments will use Python SPARQL client libraries but you will also be spending quality time using the web interface to run SPARQL queries.

RDF data is in the form of triples: subject, property (or predicate), and object. There are several serialization formats for RDF including XML, Turtle, N1, etc. I refer you to the chapter [Background Material for the Semantic Web and Knowledge Graphs in my book Practical Artificial Intelligence Programming in Clojure](https://markwatson.com/books/clojureai-site/#semantic-web) for more details (link for online reading). Here we use client libraries and web services that can return RDF data as JSON.

This chapter is about getting tools ready to use. In the next chapter we will get more into RDF and SPARQL.

I have a GitHub repository containing Fuseki, sample data, and directions for getting setup and running in a minute or two: [https://github.com/mark-watson/fuseki-semantic-web-dev-setup](https://github.com/mark-watson/fuseki-semantic-web-dev-setup). You can clone this repository and follow along on your laptop or just read the following text if you are not yet convinced that you will want to use semantic web technologies in your own projects.

We will be using the SPARQL query language here for a few examples and then jump more deeply into the use of SPARQL and other semantic web technologies in the next chapter.

Listing of **test_fuseki_client.py** (in the directory **semantic-web**):

```python
## Test client for Apache Jena Fuselki server on localhost
##
## Do git clone https://github.com/mark-watson/fuseki-semantic-web-dev-setup
## and run ./fuseki-server --file RDF/fromdbpedia.ttl /news
## in the cloned directory before running this example.

import rdflib
from SPARQLWrapper import SPARQLWrapper, JSON
from pprint import pprint

queryString = """
SELECT *
WHERE {
    ?s ?p ?o
}
LIMIT 5000
"""

sparql = SPARQLWrapper("http://localhost:3030/news")
sparql.setQuery(queryString)
sparql.setReturnFormat(JSON)
sparql.setMethod('POST')
ret = sparql.queryAndConvert()
for r in ret["results"]["bindings"]:
    pprint(r)
```

This generates output (edited for brevity):

```bash
$ python test_fuseki_client.py 
{'o': {'datatype': 'http://www.w3.org/2001/XMLSchema#decimal',
       'type': 'literal',
       'value': '56.5'},
 'p': {'type': 'uri', 'value': 'http://dbpedia.org/property/janHighF'},
 's': {'type': 'uri', 'value': 'http://dbpedia.org/resource/Sedona,_Arizona'}}
{'o': {'datatype': 'http://www.w3.org/2001/XMLSchema#integer',
       'type': 'literal',
       'value': '0'},
 'p': {'type': 'uri', 'value': 'http://dbpedia.org/property/yearRecordLowF'},
 's': {'type': 'uri', 'value': 'http://dbpedia.org/resource/Sedona,_Arizona'}}
{'o': {'datatype': 'http://www.w3.org/2001/XMLSchema#decimal',
       'type': 'literal',
       'value': '5.7000000000000001776'},
 'p': {'type': 'uri',
       'value': 'http://dbpedia.org/property/sepPrecipitationDays'},
 's': {'type': 'uri', 'value': 'http://dbpedia.org/resource/Sedona,_Arizona'}}

```

When I use RDF data from public SPARQL endpoints like DBPedia or Wikidata in applications I start by using the web based SPARQL clients for these services, find useful entities, manually look to see what properties (or predicates) and defined for these entities, and then write custom SPARQL queries to fetch the data I need for any specific application. In a later chapter we will build a Python command line tool to partially automate this process.

The following example **fuseki_cities_and_coldest_temperatures.py** finds 20 DBPedia URIs of the type "city" (the tiny part of DBPedia that we have loaded in our local Fuseki instance the only city is home town of Sedona Arizona) and collects data for a few properties for each city URI. Later we will make the same query against the DBPedia public SPARQL endpoint.

```python
import rdflib
from SPARQLWrapper import SPARQLWrapper, JSON
from pprint import pprint

queryString = """
SELECT *
WHERE {
    ?city_uri
        <http://dbpedia.org/ontology/type>
        <http://dbpedia.org/resource/City> .
    ?city_uri
        <http://dbpedia.org/property/yearRecordLowF>
        ?record_year_low_temperature .
    ?city_uri
        <http://dbpedia.org/property/populationEst>
        ?population .
    ?city_uri
         <http://www.w3.org/2000/01/rdf-schema#label>
         ?dbpedia_label FILTER (lang(?dbpedia_label) = 'en') .        
}
LIMIT 20
"""

sparql = SPARQLWrapper("http://localhost:3030/news")
sparql.setQuery(queryString)
sparql.setReturnFormat(JSON)
sparql.setMethod('POST')
ret = sparql.queryAndConvert()
for r in ret["results"]["bindings"]:
    pprint(r)
```

The output is:

```bash
$ python fuseki_cites_and_coldest_temperatures.py 
{'city_uri': {'type': 'uri',
              'value': 'http://dbpedia.org/resource/Sedona,_Arizona'},
 'dbpedia_label': {'type': 'literal',
                   'value': 'Sedona, Arizona',
                   'xml:lang': 'en'},
 'population': {'datatype': 'http://www.w3.org/2001/XMLSchema#integer',
                'type': 'literal',
                'value': '10281'},
 'record_year_low_temperature': {'datatype':
                                 'http://www.w3.org/2001/XMLSchema#integer',
                                 'type': 'literal',
                                 'value': '0'}}
```

We can make a slight change to this example to access the public DBPedia SPARQL endpoint instead of our local Fuseki instance. I copied the above Python script, renamed it to **dbpedia_cities_and_coldest_temperatures.py** changing only the URI of the SPARQL endpoint and commented setting the HTTP method to POST:

```python
sparql = SPARQLWrapper("http://dbpedia.org/sparql")
#sparql.setMethod('POST')
```

The output when querying the public DBPedia SPARQL endpoint is (output edited for brevity, showing only two cities out of the thousands in DBPedia):

```bash
$ python dbpedia_cities_and_coldest_temperatures.py 
{'city_uri': {'type': 'uri',
              'value': 'http://dbpedia.org/resource/Allardt,_Tennessee'},
 'dbpedia_label': {'type': 'literal',
                   'value': 'Allardt, Tennessee',
                   'xml:lang': 'en'},
 'population': {'datatype': 'http://www.w3.org/2001/XMLSchema#integer',
                'type': 'typed-literal',
                'value': '627'},
 'record_year_low_temperature': {'datatype':
                                 'http://www.w3.org/2001/XMLSchema#integer',
                                 'type': 'typed-literal',
                                 'value': '-27'}}
{'city_uri': {'type': 'uri',
              'value': 'http://dbpedia.org/resource/Dayton,_Ohio'},
 'dbpedia_label': {'type': 'literal',
                   'value': 'Dayton, Ohio',
                   'xml:lang': 'en'},
 'population': {'datatype': 'http://www.w3.org/2001/XMLSchema#integer',
                'type': 'typed-literal',
                'value': '137644'},
 'record_year_low_temperature': {'datatype':
                                 'http://www.w3.org/2001/XMLSchema#integer',
                                 'type': 'typed-literal',
                                 'value': '-28'}}
 ...
```

We will use more SPARQL queries in the next chapter.

## The Neo4j Community Edition and Cypher Query Server and the Memgraph Graph Database

The examples here use Neo4j. You can either [install Neo4J using these instructions](https://neo4j.com/docs/operations-manual/current/installation/) and follow along or just read the sample code and the sample output if you are not sure will be using property graph databases in your applications.

Note: You can alternatively use the [Memgraph Graph Database](https://memgraph.com) that is largely compatible with Neo4j. However, we are using a sample graph database that is included with free community edition of Neo4J so you will need to do extra work following this material using Memgraph.

We will use a Python client to access the Neo4J sample Movie Data graph database.

### Admin Tricks and Examples for Using Neo4j

While reading this section please open [the web page for the Neo4J Cypher query language tutorial](https://neo4j.com/developer/cypher/guide-cypher-basics/) for reference since I will not duplicate that information here.

The Python source code for this section can be found in the directory **neo4j**.

When writing Python scripts to create data in Neo4J it is useful to remove all data from a graph and to verify removal:

```
MATCH (n) DETACH DELETE n
MATCH (n) RETURN n
```

We will use the interactive Movie database tutorial data that is built into the Community Edition of Neo4j. I assume that you have the tutorial running on a local copy of Neo4J or have the Cypher tutorial open. The following Cypher snippet creates a movie graph node with properties *title*, *released* year, and *tagline* for the movie The Matrix. Two nodes are created for actors Keanu Reeves and Carrie-Anne Moss that have properties *name* and *born* (for their birth year). Finally we create two links indicating the both actors stared in the move The Matrix.

```
CREATE (TheMatrix:Movie
         {title:'The Matrix', released:1999,
          tagline:'Welcome to the Real World'})
CREATE (Keanu:Person {name:'Keanu Reeves', born:1964})
CREATE (Carrie:Person {name:'Carrie-Anne Moss', born:1967})
CREATE
      (Keanu)-[:ACTED_IN {roles:['Neo']}]->(TheMatrix),
      (Carrie)-[:ACTED_IN {roles:['Trinity']}]->(TheMatrix))
```

Of particular use and interest is the ability to also define properties for links between nodes. If you use a property graph in your application you will start by:

- Document the types of nodes and links you will require for your application and what properties will be attached to the types of nodes and links.
- Write a Python load script to convert your data sources to Cypher **CREATE** statements and populate your Neo4J database.
- Write Python utilities for searching, modifying, and removing data.
- Write your Python application.

Let's continue with the Cypher tutorial material by adding data constraints to ensure that all movie and actor node names are unique:

```
CREATE CONSTRAINT FOR (n:Movie) REQUIRE (n.title) IS UNIQUE
CREATE CONSTRAINT FOR (n:Person) REQUIRE (n.name) IS UNIQUE
```

Index all movie nodes on the key of movie release date:

```
CREATE INDEX FOR (m:Movie) ON (m.released)
```

This is not strictly necessary but just as we index columns in a relational database, indexing nodes can drastically speed up queries. Here is a test query from the Neo4J tutorial:

```
MATCH (tom:Person {name: "Tom Hanks"})-[:ACTED_IN]->(tomHanksMovies)
RETURN tom,tomHanksMovies
```

![Display of movies staring Tom Hanks](neo4l01.png)

By default, not all links are shown. If we double click on the node "Cast Away" in the upper left corner then all links from that node are shown in the display graph:

![Display all links from the node "Cast Away"](neo4j02.png)

We can query for movies created during a specific time period:

```
MATCH (nineties:Movie) WHERE nineties.released >= 1990 AND
                             nineties.released < 2000
RETURN nineties.title
```

The following query returns the names of all actors who co-starred with Tom Hanks in any movie:

```
MATCH (tom:Person
        {name:"Tom Hanks"})-[:ACTED_IN]->(m)<-[:ACTED_IN]-(coActors)
RETURN DISTINCT coActors.name
```

Here we use a built in function to find the shortest path in the movie data graph between Kevin Bacon and Meg Ryan. The Cypher graph expression **-[*]-** represnts any path between the "Kevin Bacon" and "Meg Ryan" nodes in the graph:

```
MATCH p=shortestPath(
    (bacon:Person {name:"Kevin Bacon"})-[*]-(meg:Person {name:"Meg Ryan"})
)
RETURN p
```

Here is the shortest path:

![Display the shortest path between Actor nodes "Kevin Bacon" and "Meg Ryan"](neo4j03.png)

### Python client code for the Neo4J Movie graph database example

The following listing shows an example from the Neo4J documentation that I modified:

```python
import logging, sys, os

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

USER = 'neo4j'
PASSWORD = os.environ.get('NEO4J_AURADB_PASSWORD')

class MovieDbExample:
    "Boilerplate code copied from Neo4J Python client documentation"

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    @staticmethod
    def enable_log(level, output_stream):
        handler = logging.StreamHandler(output_stream)
        handler.setLevel(level)
        logging.getLogger("neo4j").addHandler(handler)
        logging.getLogger("neo4j").setLevel(level)

    def find_movies(self, actor_name):
        with self.driver.session() as session:
            result = session.execute_read(self._movies_actor_is_in, actor_name)
            for row in result:
                print("Movie: {row}".format(row=row))

    @staticmethod
    def _movies_actor_is_in(tx, actor_name):
        query = (
            "MATCH (actor:Person {name: $actor_name})-[:ACTED_IN]->(movies) "
            "RETURN movies.title as title"
        )
        result = tx.run(query, actor_name=actor_name)
        return [row["title"] for row in result]

if __name__ == "__main__":
    bolt_url = "neo4j://localhost:7687"
    MovieDbExample.enable_log(logging.INFO, sys.stdout)
    MovieDbExample = MovieDbExample(bolt_url, USER, PASSWORD)
    MovieDbExample.find_movies("Tom Hanks")
    MovieDbExample.close()
```

For practical applications, you will write many helper functions for executing required Cypher queries. Before you write one line of Python code I suggest that you always experiment in the Neo4J web app with test graph database data and interactively write the Cypher queries you need. Once you have working queries then write the Python client code based on both the example we just looked at and the Neo4J Python documentation.

This ends our brief tour of property graphs. In my own work I use semantic web RDF graph databases for most of my work so we will later take a much deeper dive into RDF and the SPARQL query language, but not further use property graphs in this book except for support both RDF and Cypher data generation in a later example Knowledge Graph Creator. I personally use RDF for about 90% of my work with graph data and property graphs like Neo4J about 10% of the time.

I wanted to introduce you to property graphs because I know developers who have an easier time using property graphs. I believe that it is worth your time, dear reader, to experiment a bit with both approaches and then choose a favorite


## The SQLite Relational Database

The SQLite database is now included in the standard Python distribution so SQLite is my default persistent datastore. I tend to use RDF and SPARQL (or occasionally Neo4J) specifically when a graph database fits the requirements an application. The example code for this section can be found in the directory **/misc/datastores** that also includes examples for Postgres that I don't cover in the book text. We will also use SQLite in a later chapter in the Knowledge Graph Navigator example to cache SPARQL queries to DBPedia.

We start with writing a simple reusable library for SQLite using the standard library **sqlite3** that is defined in the file **sqlite_lib.py**:

```python
from sqlite3 import connect, version

def create_db(db_file_path):
    "create database"
    conn = connect(db_file_path)
    print(version)
    return conn.close()

def connection(db_file_path):
    "create database connection"
    return connect(db_file_path)

def query(conn, sql, variable_bindings=None):
    "run a test query"
    cur = conn.cursor()
    status = cur.execute(sql, variable_bindings) if variable_bindings else cur.execute(sql)
    print(f"query status: {status}")
    return cur.fetchall()
```

Here is a test program **sqlite_example.py**:

```python
from sqlite_lib import create_db, connection, query

def test_sqlite_lib():
    "test library"
    dbpath = ':memory:'
    create_db(dbpath)
    conn = connection(':memory:')
    query(conn, 'CREATE TABLE people (name TEXT, email TEXT);')
    print(query(conn,
        "INSERT INTO people VALUES ('Mark', 'mark@markwatson.com')"))
    print(query(conn,
        "INSERT INTO people VALUES ('Kiddo', 'kiddo@markwatson.com')"))
    print(query(conn, 'SELECT * FROM people'))
    print(query(conn, 'UPDATE people SET name = ? WHERE email = ?', [
        'Mark Watson', 'mark@markwatson.com']))
    print(query(conn, 'SELECT * FROM people'))
    print(query(conn, 'DELETE FROM people  WHERE name=?', ['Kiddo']))
    print(query(conn, 'SELECT * FROM people'))
    return conn.close()

test_sqlite_lib()
```

We will combine the use of SQLite, RDF and SPARQL, and deep learning Natural Language Processing (NLP) libraries later in the book.

