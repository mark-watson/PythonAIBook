%% family.pl - Prolog rules for family relationship reasoning
%%
%% Defines parent/2 and grandparent/2 as derived relations.
%% Facts (mother/2, father/2) are asserted at runtime from Python.
%%
%% Usage with family.py:
%%   The Python script loads this file, asserts mother/father facts,
%%   and queries grandparent relationships.

%% A parent is either a mother or a father.
parent(X, Y) :- mother(X, Y).
parent(X, Y) :- father(X, Y).

%% X is a grandparent of Z if X is a parent of Y and Y is a parent of Z.
grandparent(X, Z) :-
  parent(X, Y),
  parent(Y, Z).
