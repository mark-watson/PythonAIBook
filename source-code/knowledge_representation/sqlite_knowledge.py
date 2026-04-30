# sqlite_knowledge.py - Knowledge representation with SQLite
#
# Demonstrates using a relational database as a knowledge representation.
# Entity tables capture types (scientists, fields, discoveries) and
# relationship tables (scientist_field, scientist_discovery) capture
# how entities are connected. SQL JOINs traverse these relationships
# to answer knowledge queries.
#
# No external dependencies — uses Python's built-in sqlite3.
# Run: uv run sqlite_knowledge.py

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
