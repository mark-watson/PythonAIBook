# n_queens.py - Solve the 8-queens problem using Python + Swi-Prolog
#
# Demonstrates calling Prolog from Python via swiplserver. The Prolog
# program n_queens.pl uses the clpfd (Constraint Logic Programming over
# Finite Domains) library to find all 92 solutions for placing 8 queens
# on a chessboard so that no two queens threaten each other.
#
# Requirements: brew install swi-prolog && uv pip install swiplserver

from swiplserver import PrologMQI
from pprint import pprint

with PrologMQI() as mqi:
    with mqi.create_thread() as prolog_thread:
        # Load the constraint library and the n_queens rules
        prolog_thread.query("use_module(library(clpfd)).")
        prolog_thread.query("[n_queens].")

        # Solve: find all placements of 8 queens.
        # label(Qs) triggers the constraint solver to enumerate solutions.
        # Each solution is a list of row positions for queens in columns 1-8.
        result = prolog_thread.query("n_queens(8, Qs), label(Qs).")
        pprint(result)
        print(len(result))  # should print 92
