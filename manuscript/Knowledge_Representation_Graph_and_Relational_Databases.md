# Getting Setup To Use Graph and Relational Databases

I use several types of data stores in my work but for the purposes of this book we can explore knowledge representation using two key platforms:

- [RDF data](https://www.w3.org/RDF/) via Python's **rdflib** library and public SPARQL endpoints like [Wikidata](https://query.wikidata.org/) and [DBPedia](https://dbpedia.org/sparql) for graph-based knowledge representation.
- [SQLite](https://www.sqlite.org/index.html) for relational knowledge representation using the [SQL query language](https://en.wikipedia.org/wiki/SQL).

The next chapter covers RDF and the SPARQL query language in more detail.

In technical terms, knowledge representation using graph and relational databases involves the use of graph structures and relational data models to represent and organize knowledge in a structured, computationally efficient, and easily accessible way.

A graph structure is a collection of nodes (also known as vertices) and edges (also known as arcs) that connect the nodes. Each node and edge in a graph can have properties, such as labels and attributes which provide information about the entities they represent. Graphs can be used to represent knowledge in a variety of ways, such as through semantic networks and using ontologies to define terms, classes, types, etc.

Relational databases, on the other hand, use a tabular data model to represent knowledge. The basic building block of a relational database is the table, which is a collection of rows (also known as tuples) and columns (also known as attributes). Each row represents an instance of an entity, and the columns provide information about the properties of that entity. Relationships between entities can also be represented by foreign keys, which link one table to another.

Combining these two technologies, knowledge can be represented as a graph of interconnected entities, where each entity is stored in a relational database table and connected to other entities through relationships represented by edges in the graph. This allows for efficient querying and manipulation of knowledge, as well as the ability to integrate and reason over large amounts of information.

## Querying Wikidata with SPARQL and Python

[Wikidata](https://www.wikidata.org/) is a free, open knowledge base maintained by the Wikimedia Foundation. It contains structured data about millions of entities — people, places, organizations, scientific concepts, and more — all accessible through a public SPARQL endpoint. Unlike DBPedia, which extracts structured data from Wikipedia infoboxes, Wikidata is a curated knowledge base where the data is entered and maintained directly.

The Python **SPARQLWrapper** library makes it straightforward to query any SPARQL endpoint, including Wikidata:

```bash
uv pip install sparqlwrapper
```

### Finding Information About a Person

Let's query Wikidata for information about a specific person. Wikidata uses numeric entity identifiers (like Q937 for Albert Einstein) and property identifiers (like P19 for "place of birth"):

```python
# wikidata_person.py - Query Wikidata for information about a person

from SPARQLWrapper import SPARQLWrapper, JSON
from pprint import pprint

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.addCustomHttpHeader("User-Agent", "PythonAIBook/1.0")

queryString = """
SELECT ?personLabel ?birthPlaceLabel ?birthDate ?occupationLabel
WHERE {
    ?person wdt:P31 wd:Q5 .            # instance of human
    ?person rdfs:label "Albert Einstein"@en .
    OPTIONAL { ?person wdt:P19 ?birthPlace . }
    OPTIONAL { ?person wdt:P569 ?birthDate . }
    OPTIONAL { ?person wdt:P106 ?occupation . }
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
}
LIMIT 10
"""

sparql.setQuery(queryString)
sparql.setReturnFormat(JSON)
results = sparql.queryAndConvert()

for r in results["results"]["bindings"]:
    print(f"  Name: {r['personLabel']['value']}")
    if 'birthPlaceLabel' in r:
        print(f"  Born: {r['birthPlaceLabel']['value']}")
    if 'birthDate' in r:
        print(f"  Date: {r['birthDate']['value'][:10]}")
    if 'occupationLabel' in r:
        print(f"  Occupation: {r['occupationLabel']['value']}")
    print()
```

The output shows Wikidata returning multiple results — one per occupation — for the same person:

```
$ uv run wikidata_person.py
  Name: Albert Einstein
  Born: Ulm
  Date: 1879-03-14
  Occupation: scientist

  Name: Albert Einstein
  Born: Ulm
  Date: 1879-03-14
  Occupation: physicist

  Name: Albert Einstein
  Born: Ulm
  Date: 1879-03-14
  Occupation: mathematician

  Name: Albert Einstein
  Born: Ulm
  Date: 1879-03-14
  Occupation: inventor
  ...
```

Key things to notice about Wikidata's SPARQL:

- **wdt:** properties represent direct claims (e.g., wdt:P19 is "place of birth")
- **wd:** entities are Wikidata items (e.g., wd:Q5 is "human")
- The **SERVICE wikibase:label** clause automatically resolves entity IDs to human-readable labels
- **OPTIONAL** prevents the query from failing when a property is missing

### Querying Cities and Their Properties from DBPedia

DBPedia mirrors much of Wikipedia's structured content as RDF triples. It uses different ontology conventions than Wikidata but is equally useful for knowledge representation tasks. Here we query DBPedia's public SPARQL endpoint for cities and their populations:

```python
# dbpedia_cities.py - Query DBPedia for city data

from SPARQLWrapper import SPARQLWrapper, JSON
from pprint import pprint

queryString = """
SELECT ?city_uri ?dbpedia_label ?population ?country_label
WHERE {
    ?city_uri
        <http://dbpedia.org/ontology/type>
        <http://dbpedia.org/resource/City> .
    ?city_uri
        <http://dbpedia.org/property/populationEst>
        ?population .
    ?city_uri
         <http://www.w3.org/2000/01/rdf-schema#label>
         ?dbpedia_label FILTER (lang(?dbpedia_label) = 'en') .
    OPTIONAL {
        ?city_uri <http://dbpedia.org/ontology/country> ?country .
        ?country <http://www.w3.org/2000/01/rdf-schema#label>
                 ?country_label FILTER (lang(?country_label) = 'en') .
    }
}
ORDER BY DESC(?population)
LIMIT 10
"""

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setQuery(queryString)
sparql.setReturnFormat(JSON)
results = sparql.queryAndConvert()

for r in results["results"]["bindings"]:
    city = r['dbpedia_label']['value']
    pop = int(r['population']['value'])
    country = r.get('country_label', {}).get('value', 'unknown')
    print(f"  {city} ({country}): population {pop:,}")
```

The output (results may vary as DBPedia data is updated):

```
$ uv run dbpedia_cities.py
  Fort Worth, Texas (United States): population 1,008,106
  Charlotte, North Carolina (unknown): population 911,311
  Detroit (unknown): population 645,705
  Gombe, Nigeria (Nigeria): population 446,800
  Ilesa (Nigeria): population 416,000
  Pittsburgh (unknown): population 307,668
  Durham, North Carolina (United States): population 296,186
  Toledo, Ohio (United States): population 265,638
  Winston-Salem, North Carolina (United States): population 252,975
  Huntsville, Alabama (unknown): population 249,102
```

When I use RDF data from public SPARQL endpoints like DBPedia or Wikidata in applications, I start by using the web-based SPARQL clients for these services, find useful entities, manually look to see what properties are defined for those entities, and then write custom SPARQL queries to fetch the data I need. The web-based query editors at [query.wikidata.org](https://query.wikidata.org/) and [dbpedia.org/sparql](http://dbpedia.org/sparql) are invaluable for this exploratory process.

We will use more SPARQL queries in the next chapter.


## The SQLite Relational Database for Knowledge Representation

The SQLite database is included in the standard Python distribution, making it the zero-setup option for persistent data storage. While graph databases naturally express relationships between entities, relational databases can also serve as effective knowledge representations when the schema is designed to capture entity types, attributes, and relationships.

### A Reusable SQLite Library

We start with a simple reusable library for SQLite using the standard library **sqlite3**:

```python
# sqlite_lib.py - Reusable SQLite helper functions

from sqlite3 import connect, version

def create_db(db_file_path):
    """Create a database and return the connection."""
    conn = connect(db_file_path)
    return conn

def connection(db_file_path):
    """Create and return a database connection."""
    return connect(db_file_path)

def query(conn, sql, variable_bindings=None):
    """Execute a SQL query and return all results."""
    cur = conn.cursor()
    if variable_bindings:
        cur.execute(sql, variable_bindings)
    else:
        cur.execute(sql)
    conn.commit()
    return cur.fetchall()
```

### Modeling a Knowledge Graph in SQLite

Relational databases become knowledge representation tools when we design tables to capture entities, their types, their attributes, and the relationships between them. Here is an example that builds a simple knowledge base about scientists, their fields, and their discoveries:

```python
# sqlite_knowledge.py - Knowledge representation with SQLite

import sqlite3

def build_knowledge_base():
    """Build a relational knowledge base about scientists and their work."""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    # Entity tables: each table represents a type of entity
    cur.execute("""
        CREATE TABLE scientists (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            birth_year INTEGER,
            nationality TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE fields (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE discoveries (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            year INTEGER,
            description TEXT
        )
    """)

    # Relationship tables: capture how entities are connected
    cur.execute("""
        CREATE TABLE scientist_field (
            scientist_id INTEGER REFERENCES scientists(id),
            field_id INTEGER REFERENCES fields(id),
            PRIMARY KEY (scientist_id, field_id)
        )
    """)

    cur.execute("""
        CREATE TABLE scientist_discovery (
            scientist_id INTEGER REFERENCES scientists(id),
            discovery_id INTEGER REFERENCES discoveries(id),
            PRIMARY KEY (scientist_id, discovery_id)
        )
    """)

    # Populate with knowledge
    cur.executemany("INSERT INTO scientists VALUES (?, ?, ?, ?)", [
        (1, "Albert Einstein", 1879, "German"),
        (2, "Marie Curie", 1867, "Polish"),
        (3, "Richard Feynman", 1918, "American"),
    ])

    cur.executemany("INSERT INTO fields VALUES (?, ?, ?)", [
        (1, "Physics", "Study of matter, energy, and their interactions"),
        (2, "Chemistry", "Study of the composition and properties of matter"),
        (3, "Quantum Mechanics", "Physics of atomic and subatomic systems"),
    ])

    cur.executemany("INSERT INTO discoveries VALUES (?, ?, ?, ?)", [
        (1, "Special Relativity", 1905, "Time and space are relative"),
        (2, "Radioactivity", 1898, "Discovery of radium and polonium"),
        (3, "Quantum Electrodynamics", 1948, "Quantum theory of light and matter"),
    ])

    cur.executemany("INSERT INTO scientist_field VALUES (?, ?)", [
        (1, 1), (1, 3),  # Einstein: Physics, Quantum Mechanics
        (2, 1), (2, 2),  # Curie: Physics, Chemistry
        (3, 1), (3, 3),  # Feynman: Physics, Quantum Mechanics
    ])

    cur.executemany("INSERT INTO scientist_discovery VALUES (?, ?)", [
        (1, 1),  # Einstein -> Special Relativity
        (2, 2),  # Curie -> Radioactivity
        (3, 3),  # Feynman -> QED
    ])

    conn.commit()
    return conn


def query_knowledge_base(conn):
    """Demonstrate knowledge queries against the relational schema."""
    cur = conn.cursor()

    # Query 1: Who works in Quantum Mechanics?
    print("Scientists in Quantum Mechanics:")
    cur.execute("""
        SELECT s.name, s.nationality
        FROM scientists s
        JOIN scientist_field sf ON s.id = sf.scientist_id
        JOIN fields f ON sf.field_id = f.id
        WHERE f.name = 'Quantum Mechanics'
    """)
    for row in cur.fetchall():
        print(f"  {row[0]} ({row[1]})")

    # Query 2: What did each scientist discover?
    print("\nDiscoveries by scientist:")
    cur.execute("""
        SELECT s.name, d.name, d.year, d.description
        FROM scientists s
        JOIN scientist_discovery sd ON s.id = sd.scientist_id
        JOIN discoveries d ON sd.discovery_id = d.id
        ORDER BY d.year
    """)
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]} ({row[2]}) — {row[3]}")

    # Query 3: Which fields overlap between scientists?
    print("\nScientists who share a field:")
    cur.execute("""
        SELECT s1.name, s2.name, f.name
        FROM scientist_field sf1
        JOIN scientist_field sf2 ON sf1.field_id = sf2.field_id
                                AND sf1.scientist_id < sf2.scientist_id
        JOIN scientists s1 ON sf1.scientist_id = s1.id
        JOIN scientists s2 ON sf2.scientist_id = s2.id
        JOIN fields f ON sf1.field_id = f.id
    """)
    for row in cur.fetchall():
        print(f"  {row[0]} & {row[1]}: {row[2]}")


conn = build_knowledge_base()
query_knowledge_base(conn)
conn.close()
```

The output shows how SQL JOIN queries traverse the relationships between entities, much like following edges in a graph:

```
Scientists in Quantum Mechanics:
  Albert Einstein (German)
  Richard Feynman (American)

Discoveries by scientist:
  Marie Curie: Radioactivity (1898) — Discovery of radium and polonium
  Albert Einstein: Special Relativity (1905) — Time and space are relative
  Richard Feynman: Quantum Electrodynamics (1948) — Quantum theory of light and matter

Scientists who share a field:
  Albert Einstein & Marie Curie: Physics
  Albert Einstein & Richard Feynman: Physics
  Albert Einstein & Richard Feynman: Quantum Mechanics
  Marie Curie & Richard Feynman: Physics
```

The key insight is that the **relationship tables** (scientist_field, scientist_discovery) transform a flat relational database into a knowledge representation. Each relationship table captures a specific type of connection between entity types, and SQL JOINs let you traverse these connections to answer knowledge queries. While not as natural as a graph database for highly connected data, this pattern works well for structured knowledge with well-defined entity types and relationships.

We will combine the use of SQLite, RDF, SPARQL, and deep learning Natural Language Processing (NLP) libraries later in the book.

If you want to deepen your understanding of the standards behind the SPARQL queries we used in this chapter, the next chapter provides optional reference material on RDF data formats, RDFS sub-property hierarchies, the SPARQL query language in detail, and OWL reasoning. That background will help you write more sophisticated queries against Wikidata, DBPedia, and any other SPARQL endpoint.
