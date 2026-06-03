# sqlite_lib.py - Reusable SQLite helper functions
#
# A thin wrapper around Python's built-in sqlite3 module providing
# simple connect and query functions.

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
