# sqlite_lib.py - Reusable SQLite helper functions
#
# A thin wrapper around Python's built-in sqlite3 module providing
# simple create, connect, and query functions.

from sqlite3 import connect

def create_db(db_file_path):
    """Create a database and return the connection."""
    conn = connect(db_file_path)
    return conn

def connection(db_file_path):
    """Create and return a database connection."""
    return connect(db_file_path)

def query(conn, sql, variable_bindings=None):
    """Execute a SQL query, commit, and return all results."""
    cur = conn.cursor()
    if variable_bindings:
        cur.execute(sql, variable_bindings)
    else:
        cur.execute(sql)
    conn.commit()
    return cur.fetchall()
