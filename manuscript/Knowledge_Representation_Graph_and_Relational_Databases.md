# Getting Setup To Use Graph and Relational Databases

I use several types of data stores in my work but for the purposes of this book we can explore knowledge representation using two key platforms:

- [RDF data](https://www.w3.org/RDF/) via Python's **rdflib** library and public SPARQL endpoints like [Wikidata](https://query.wikidata.org/) and [DBPedia](https://dbpedia.org/sparql) for graph-based knowledge representation.
- [SQLite](https://www.sqlite.org/index.html) for relational knowledge representation using the [SQL query language](https://en.wikipedia.org/wiki/SQL).

The next chapter covers RDF and the SPARQL query language in more detail.

The examples for this chapter are in the directory **source-code/knowledge_representation**.

In technical terms, knowledge representation using graph and relational databases involves the use of graph structures and relational data models to represent and organize knowledge in a structured, computationally efficient, and easily accessible way.

A graph structure is a collection of nodes (also known as vertices) and edges (also known as arcs) that connect the nodes. Each node and edge in a graph can have properties, such as labels and attributes which provide information about the entities they represent. Graphs can be used to represent knowledge in a variety of ways, such as through semantic networks and using ontologies to define terms, classes, types, etc.

Relational databases, on the other hand, use a tabular data model to represent knowledge. The basic building block of a relational database is the table, which is a collection of rows (also known as tuples) and columns (also known as attributes). Each row represents an instance of an entity, and the columns provide information about the properties of that entity. Relationships between entities can also be represented by foreign keys, which link one table to another.

Combining these two technologies, knowledge can be represented as a graph of interconnected entities, where each entity is stored in a relational database table and connected to other entities through relationships represented by edges in the graph. This allows for efficient querying and manipulation of knowledge, as well as the ability to integrate and reason over large amounts of information.

{width: "80%"}
![Architecture diagram for the Knowledge Representation example](FIG_knowledge_representation.jpg)

## Querying Wikidata with SPARQL and Python

[Wikidata](https://www.wikidata.org/) is a free, open knowledge base maintained by the Wikimedia Foundation. It contains structured data about millions of entities (people, places, organizations, scientific concepts, and more), all accessible through a public SPARQL endpoint. Unlike DBPedia, which extracts structured data from Wikipedia infoboxes, Wikidata is a curated knowledge base where the data is entered and maintained directly.

The Python **SPARQLWrapper** library makes it straightforward to query any SPARQL endpoint, including Wikidata:

```bash
uv pip install sparqlwrapper
```

### Finding Information About a Person

Let's query Wikidata for information about a specific person. Wikidata uses numeric entity identifiers (like Q937 for Albert Einstein) and property identifiers (like P19 for "place of birth"):

```python
# wikidata_person.py - Query Wikidata for information about a person

import sys
from SPARQLWrapper import SPARQLWrapper, JSON

QUERY_TEMPLATE = """
SELECT ?personLabel ?birthPlaceLabel ?birthDate
       (GROUP_CONCAT(DISTINCT ?occupationLabel; SEPARATOR=", ") AS ?occupations)
WHERE {{
    ?person wdt:P31 wd:Q5 .
    ?person rdfs:label "{name}"@en .
    OPTIONAL {{ ?person wdt:P19 ?birthPlace . }}
    OPTIONAL {{ ?person wdt:P569 ?birthDate . }}
    OPTIONAL {{ ?person wdt:P106 ?occupation . }}
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
GROUP BY ?personLabel ?birthPlaceLabel ?birthDate
LIMIT 5
"""


def fetch_person(name: str) -> list[dict[str, str]]:
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.addCustomHttpHeader("User-Agent", "PythonAIBook/1.0")
    sparql.setQuery(QUERY_TEMPLATE.format(name=name))
    sparql.setReturnFormat(JSON)
    results = sparql.queryAndConvert()

    bindings = results.get("results", {}).get("bindings", [])
    people = []
    for r in bindings:
        people.append({
            "name": r.get("personLabel", {}).get("value", "unknown"),
            "birth_place": r.get("birthPlaceLabel", {}).get("value", ""),
            "birth_date": r.get("birthDate", {}).get("value", "")[:10],
            "occupations": r.get("occupations", {}).get("value", ""),
        })
    return people


if __name__ == "__main__":
    person_name = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Albert Einstein"
    try:
        people = fetch_person(person_name)
        if not people:
            print(f"No results found for '{person_name}'.")
        else:
            for p in people:
                print(f"  Name: {p['name']}")
                if p["birth_place"]:
                    print(f"  Born: {p['birth_place']}")
                if p["birth_date"]:
                    print(f"  Date: {p['birth_date']}")
                if p["occupations"]:
                    print(f"  Occupations: {p['occupations']}")
                print()
    except Exception as e:
        print(f"Error querying Wikidata: {e}")
```

The output returns a single clean row per person, with all occupations aggregated into one field:

```
$ uv run wikidata_person.py
  Name: Albert Einstein
  Born: Ulm
  Date: 1879-03-14
  Occupations: scientist, physicist, mathematician, inventor
```

Key things to notice about Wikidata's SPARQL:

- **wdt:** properties represent direct claims (e.g., wdt:P19 is "place of birth")
- **wd:** entities are Wikidata items (e.g., wd:Q5 is "human")
- The **SERVICE wikibase:label** clause automatically resolves entity IDs to human-readable labels
- **OPTIONAL** prevents the query from failing when a property is missing

### Querying Cities and Their Properties from DBPedia

DBPedia mirrors much of Wikipedia's structured content as RDF triples. It uses different ontology conventions than Wikidata but is equally useful for knowledge representation tasks. Here we query DBPedia's public SPARQL endpoint (using HTTPS for reliability) for cities and their populations:

```python
# dbpedia_cities.py - Query DBPedia for city data

from SPARQLWrapper import SPARQLWrapper, JSON

QUERY_STRING = """
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


def fetch_cities() -> list[dict[str, str]]:
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.addCustomHttpHeader("User-Agent", "PythonAIBook/1.0")
    sparql.setQuery(QUERY_STRING)
    sparql.setReturnFormat(JSON)
    results = sparql.queryAndConvert()

    bindings = results.get("results", {}).get("bindings", [])
    cities = []
    for r in bindings:
        cities.append({
            "city": r.get("dbpedia_label", {}).get("value", "unknown"),
            "population": int(r.get("population", {}).get("value", 0)),
            "country": r.get("country_label", {}).get("value", "unknown"),
        })
    return cities


if __name__ == "__main__":
    try:
        cities = fetch_cities()
        if not cities:
            print("No results returned from DBpedia.")
        else:
            for c in cities:
                print(f"  {c['city']} ({c['country']}): population {c['population']:,}")
    except Exception as e:
        print(f"Error querying DBpedia: {e}")
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

When I use RDF data from public SPARQL endpoints like DBPedia or Wikidata in applications, I start by using the web-based SPARQL clients for these services, find useful entities, manually look to see what properties are defined for those entities, and then write custom SPARQL queries to fetch the data I need. The web-based query editors at [query.wikidata.org](https://query.wikidata.org/) and [dbpedia.org/sparql](https://dbpedia.org/sparql) are invaluable for this exploratory process.

We will use more SPARQL queries in the next chapter.


## The SQLite Relational Database for Knowledge Representation

The SQLite database is included in the standard Python distribution, making it the zero-setup option for persistent data storage. While graph databases naturally express relationships between entities, relational databases can also serve as effective knowledge representations when the schema is designed to capture entity types, attributes, and relationships.

### A Reusable SQLite Library

We start with a simple reusable library for SQLite using the standard library **sqlite3**:

```python
# sqlite_lib.py - Reusable SQLite helper functions

import sqlite3
from typing import Any


def connection(db_file_path: str) -> sqlite3.Connection:
    """Create and return a database connection."""
    return sqlite3.connect(db_file_path)


def query(
    conn: sqlite3.Connection, sql: str, variable_bindings: tuple[Any, ...] | None = None
) -> list[sqlite3.Row]:
    """Execute a SQL query, commit if it modifies data, and return all results."""
    cur = conn.cursor()
    try:
        if variable_bindings:
            cur.execute(sql, variable_bindings)
        else:
            cur.execute(sql)
    except sqlite3.Error as e:
        conn.rollback()
        raise
    else:
        if sql.strip().upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE")):
            conn.commit()
        return cur.fetchall()
```

### Modeling a Knowledge Graph in SQLite

Relational databases become knowledge representation tools when we design tables to capture entities, their types, their attributes, and the relationships between them. Here is an example that builds a simple knowledge base about scientists, their fields, and their discoveries:

```python
# sqlite_knowledge.py - Knowledge representation with SQLite

import sqlite3


def build_knowledge_base() -> sqlite3.Connection:
    """Build a relational knowledge base about scientists and their work."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

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

    cur.execute(
        "INSERT INTO scientists (name, birth_year, nationality) VALUES (?, ?, ?)",
        ("Albert Einstein", 1879, "German"),
    )
    einstein_id = cur.lastrowid

    cur.execute(
        "INSERT INTO scientists (name, birth_year, nationality) VALUES (?, ?, ?)",
        ("Marie Curie", 1867, "Polish"),
    )
    curie_id = cur.lastrowid

    cur.execute(
        "INSERT INTO scientists (name, birth_year, nationality) VALUES (?, ?, ?)",
        ("Richard Feynman", 1918, "American"),
    )
    feynman_id = cur.lastrowid

    cur.execute(
        "INSERT INTO fields (name, description) VALUES (?, ?)",
        ("Physics", "Study of matter, energy, and their interactions"),
    )
    physics_id = cur.lastrowid

    cur.execute(
        "INSERT INTO fields (name, description) VALUES (?, ?)",
        ("Chemistry", "Study of the composition and properties of matter"),
    )
    chemistry_id = cur.lastrowid

    cur.execute(
        "INSERT INTO fields (name, description) VALUES (?, ?)",
        ("Quantum Mechanics", "Physics of atomic and subatomic systems"),
    )
    qm_id = cur.lastrowid

    cur.execute(
        "INSERT INTO discoveries (name, year, description) VALUES (?, ?, ?)",
        ("Special Relativity", 1905, "Time and space are relative"),
    )
    sr_id = cur.lastrowid

    cur.execute(
        "INSERT INTO discoveries (name, year, description) VALUES (?, ?, ?)",
        ("Radioactivity", 1898, "Discovery of radium and polonium"),
    )
    rad_id = cur.lastrowid

    cur.execute(
        "INSERT INTO discoveries (name, year, description) VALUES (?, ?, ?)",
        ("Quantum Electrodynamics", 1948, "Quantum theory of light and matter"),
    )
    qed_id = cur.lastrowid

    cur.executemany("INSERT INTO scientist_field VALUES (?, ?)", [
        (einstein_id, physics_id), (einstein_id, qm_id),
        (curie_id, physics_id), (curie_id, chemistry_id),
        (feynman_id, physics_id), (feynman_id, qm_id),
    ])

    cur.executemany("INSERT INTO scientist_discovery VALUES (?, ?)", [
        (einstein_id, sr_id),
        (curie_id, rad_id),
        (feynman_id, qed_id),
    ])

    conn.commit()
    return conn


def query_knowledge_base(conn: sqlite3.Connection) -> None:
    """Demonstrate knowledge queries against the relational schema."""
    cur = conn.cursor()

    print("Scientists in Quantum Mechanics:")
    cur.execute("""
        SELECT s.name, s.nationality
        FROM scientists s
        JOIN scientist_field sf ON s.id = sf.scientist_id
        JOIN fields f ON sf.field_id = f.id
        WHERE f.name = 'Quantum Mechanics'
    """)
    for row in cur.fetchall():
        print(f"  {row['name']} ({row['nationality']})")

    print("\nDiscoveries by scientist:")
    cur.execute("""
        SELECT s.name AS scientist, d.name AS discovery,
               d.year, d.description
        FROM scientists s
        JOIN scientist_discovery sd ON s.id = sd.scientist_id
        JOIN discoveries d ON sd.discovery_id = d.id
        ORDER BY d.year
    """)
    for row in cur.fetchall():
        print(f"  {row['scientist']}: {row['discovery']} "
              f"({row['year']}) — {row['description']}")

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


if __name__ == "__main__":
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

## Optional Practice Problems

To help solidify your understanding of knowledge representation using graph and relational systems, try implementing the following practice problems. They build directly on the code examples provided in this chapter.

### 1. Extended Wikidata Querying (Easy)
Modify the [wikidata_person.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/knowledge_representation/wikidata_person.py) script to fetch additional metadata for a queried individual.
- **Goal**: Add the person's **date of death** (`wdt:P570`) and **awards received** (`wdt:P166`).
- **Details**:
  - Update the SPARQL query template to retrieve the date of death and awards as optional values.
  - Since a person can receive multiple awards, use `GROUP_CONCAT` in your SPARQL query (similar to how occupations are handled in the existing query) to join the awards into a comma-separated string: `(GROUP_CONCAT(DISTINCT ?awardLabel; SEPARATOR=", ") AS ?awards)`.
  - Update the [fetch_person](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/knowledge_representation/wikidata_person.py#L30) function to parse these new fields.
  - Format the console output to print the date of death and the list of awards if they are present.
  - Test your script with "Marie Curie" or "Stephen Hawking".

### 2. Relational Knowledge Base Expansion (Medium)
Extend the SQLite knowledge representation in [sqlite_knowledge.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/knowledge_representation/sqlite_knowledge.py) to model the academic and research institutions associated with the scientists.
- **Goal**: Track where scientists performed their research.
- **Details**:
  - Create a new `institutions` table: `id INTEGER PRIMARY KEY`, `name TEXT NOT NULL`, `location TEXT`.
  - Create a many-to-many relationship table `scientist_institution` to link scientists to institutions. Include fields for `scientist_id` (foreign key to `scientists`), `institution_id` (foreign key to `institutions`), and optionally `start_year` and `end_year` to track their tenure.
  - Add sample data for the existing scientists. For example:
    - Albert Einstein at the *Institute for Advanced Study* (Princeton, USA).
    - Marie Curie at the *University of Paris* (Paris, France).
    - Richard Feynman at the *California Institute of Technology* (Pasadena, USA).
  - Modify the [query_knowledge_base](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/knowledge_representation/sqlite_knowledge.py#L134) function to display a list of scientists along with their institutions and locations.

### 3. DBpedia Query Customization and Pagination (Medium)
Expand the query capability in [dbpedia_cities.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/knowledge_representation/dbpedia_cities.py) to allow more flexible searching.
- **Goal**: Allow users to query cities in a specific country and paginate the results.
- **Details**:
  - Modify the [fetch_cities](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/knowledge_representation/dbpedia_cities.py#L35) function to accept optional parameters for a country filter, a `limit` parameter, and an `offset` parameter.
  - In the SPARQL query, dynamically inject a country filter (e.g., using `FILTER(CONTAINS(LCASE(?country_label), LCASE("{country}")))` if a country is provided).
  - Use SPARQL `LIMIT` and `OFFSET` clauses dynamically based on the passed parameters to support pagination.
  - Update the CLI entry point to accept command-line arguments for country, limit, and offset, demonstrating how users can page through large result sets.

### 4. Hybrid Knowledge Loader: API to SQL (Hard)
Write a new Python script that bridges SPARQL query APIs and local relational stores by dynamically loading Wikidata details into a persistent SQLite database.
- **Goal**: Populate SQLite tables dynamically from public SPARQL query results.
- **Details**:
  - Create a new script `wikidata_to_sqlite.py` in [source-code/knowledge_representation](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/knowledge_representation) that connects to a local, persistent SQLite file (e.g., `knowledge.db`) using the schema structure from [sqlite_knowledge.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/knowledge_representation/sqlite_knowledge.py).
  - Accept a scientist's name from the command line, query Wikidata using [fetch_person](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/knowledge_representation/wikidata_person.py#L30) (or an extended version of it), and retrieve their birth year, nationality, occupations, and discoveries.
  - Implement duplicate checking: use SQL queries to verify if the scientist already exists in the `scientists` table to avoid duplicates.
  - Programmatically insert any new fields or discoveries into the `fields` and `discoveries` tables, retrieve their generated database IDs, and construct the relationship links in `scientist_field` and `scientist_discovery`.
