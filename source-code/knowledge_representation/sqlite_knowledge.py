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
        SELECT s.name AS scientist, d.name AS discovery, d.year, d.description
        FROM scientists s
        JOIN scientist_discovery sd ON s.id = sd.scientist_id
        JOIN discoveries d ON sd.discovery_id = d.id
        ORDER BY d.year
    """)
    for row in cur.fetchall():
        print(f"  {row['scientist']}: {row['discovery']} ({row['year']}) — {row['description']}")

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
