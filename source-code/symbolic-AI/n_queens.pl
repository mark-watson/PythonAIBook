%% n_queens.pl - N-Queens constraint satisfaction program
%%
%% Uses Swi-Prolog's clpfd (Constraint Logic Programming over Finite Domains)
%% library to solve the N-Queens problem. Given N, finds all ways to place N
%% queens on an NxN board such that no two queens share a row, column, or
%% diagonal.
%%
%% Adapted from the Swi-Prolog documentation:
%%   https://www.swi-prolog.org/pldoc/man?section=clpfd-n-queens
%%
%% Usage from Prolog REPL:
%%   ?- use_module(library(clpfd)).
%%   ?- [n_queens].
%%   ?- n_queens(8, Qs), label(Qs).

:- use_module(library(clpfd)).

%% n_queens(+N, -Qs): Qs is a list of N row positions (one per column).
n_queens(N, Qs) :-
    length(Qs, N),
    Qs ins 1..N,        % each queen's row is in the range 1..N
    safe_queens(Qs).

%% safe_queens(+Qs): all queens in the list are mutually non-attacking.
safe_queens([]).
safe_queens([Q|Qs]) :-
    safe_queens(Qs, Q, 1),  % check Q against every queen to its right
    safe_queens(Qs).         % recursively check remaining queens

%% safe_queens(+Qs, +Q0, +D0): queen Q0 does not attack any queen in Qs,
%% where D0 is the column distance between Q0 and the head of Qs.
safe_queens([], _, _).
safe_queens([Q|Qs], Q0, D0) :-
    Q0 #\= Q,               % not on the same row
    abs(Q0 - Q) #\= D0,     % not on the same diagonal
    D1 #= D0 + 1,
    safe_queens(Qs, Q0, D1).
