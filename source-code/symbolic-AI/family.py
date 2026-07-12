# family.py - Prolog-based family relationship reasoning from Python
#
# Demonstrates loading Prolog rules (family.pl), asserting facts at runtime,
# and querying derived relationships. The Prolog rules define parent/2 and
# grandparent/2; we assert mother/father facts from Python and let Prolog's
# inference engine deduce grandparent relationships.
#
# Requirements: brew install swi-prolog && uv pip install swiplserver

from swiplserver import PrologMQI
from pprint import pprint

with PrologMQI() as mqi:
    with mqi.create_thread() as prolog_thread:
        # Load the family rules (parent/2 and grandparent/2)
        prolog_thread.query("[family].")

        # Assert initial family facts: Irene is Ken's mother, Ken is Ron's father
        print("Assert a few initial facts:")
        prolog_thread.query("assertz(mother(irene, ken)).")
        prolog_thread.query("assertz(father(ken, ron)).")

        # Query: who are grandparents of whom?
        # Prolog chains: mother(irene, ken) + father(ken, ron) => grandparent(irene, ron)
        result = prolog_thread.query("grandparent(A, B).")
        pprint(result)
        print(len(result))

        # Add another child for Ken and re-query
        print("Assert another test fact:")
        prolog_thread.query("assertz(father(ken, sam)).")
        result = prolog_thread.query("grandparent(A, B).")
        pprint(
            result
        )  # now includes both grandparent(irene, ron) and grandparent(irene, sam)
        print(len(result))
