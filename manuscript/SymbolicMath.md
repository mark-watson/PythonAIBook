# Symbolic Math in Pure Python

Dear reader, I have experimented with symbolic math problems since the early 1980s when I discovered the Reduce math system installed on my Xerox 1108 Lisp Machine. A few months ago I wrote a symbolic math chapter for my Common Lisp book. Here in this chapter, I implement useful symbolic math functionality in Python.

## 1. Why Symbolic Computation Matters

A calculator can tell you that the derivative of `x^2` at the point `x = 0.8`
is *approximately* `1.6`. But it cannot tell you *why*, and it cannot hand you
back the general answer `2·x`. That is the difference between **numeric**
mathematics and **symbolic** mathematics.

- *Numeric* computation operates on concrete values. It is fast and universal,
  but every answer is approximate, valid only at a single point, and carries
  floating-point rounding error.
- *Symbolic* computation operates on *expressions themselves*. It applies the
  rules of algebra (the product rule, the chain rule, the power rule) as
  transformations on expression trees, producing exact results such as
  `d/dx (x²) = 2x` and `∫ x² dx = x³/3`.

Symbolic systems (computer algebra systems, or CAS) are the engine behind
Mathematica, SymPy, and the equation solvers inside graphing calculators. At
their heart is a single, powerful idea: **represent a mathematical expression
as a data structure, and implement the rules of calculus as functions over
that data structure.**

This chapter builds a miniature CAS from nothing but the Python standard
library. The result differentiates and integrates a useful subset of
elementary functions, simplifies the results algebraically, and crucially
*verifies its own answers* by comparing them against numeric finite-difference
approximations. No `sympy`, no `numpy`, nothing to install.

The whole system is four ingredients:

1. A **representation** consists of expressions encoded as nested tuples.
2. A **simplifier** object consists of algebraic identities applied bottom-up.
3. A **differentiator** and an **integrator** are recursive rule tables.
4. A **verifier** object is a numeric evaluation checked against finite differences.

### The central design choice: expressions as data

Just as a compiler turns source text into an abstract syntax tree before
generating code, a CAS turns a formula into an *expression tree*. In Python,
the cheapest honest tree is a **nested tuple**, and every node is tagged with a
string naming the operation. Here is the entire vocabulary:

```
('num', value)              constant (value is a fractions.Fraction)
('var', name)               variable, e.g. ('var', 'x')
('add', a, b)               a + b
('mul', a, b)               a * b
('pow', base, exp)          base ** exp
('sin'|'cos'|'exp'|'log', arg)
```

A concrete formula is *data you can print, inspect, and recurse over*:

```
x^2 + 3*x   becomes
('add',
    ('pow', ('var', 'x'), ('num', 2)),
    ('mul', ('num', 3), ('var', 'x')))
```

Compare this to the numeric approach, where `x^2 + 3*x` would be compiled to a
function that only accepts a numeric `x`. In the symbolic approach the tree
itself is the object of study; for example you can ask "what is the derivative?" because
the structure of the expression is right there in the tuple, ready to be
pattern-matched.

One deliberately un-Pythonic choice will seem odd at first: we do **not**
overload `+` and `*` on a `Symbol` class. Instead we construct expressions
with small builder functions (`add`, `mul`, `pow`). This keeps every node a
plain tuple, which makes the code self-contained, trivially serializable, and
dead simple to read: three virtues that outweigh the loss of operator
sugar in a teaching implementation.

### Why exact arithmetic via `fractions.Fraction`

If we stored `1/3` as the float `0.3333333333333333`, then simplifying
`∫ x² dx = (1/3)·x³` would quietly round the leading coefficient. After a few
dozen rules fired, error would accumulate and the self-checks in this chapter
would start failing. The fix is to store every numeric constant as a
`fractions.Fraction`, an exact rational type in the standard library.
`Fraction(1, 3)` is genuinely one third, not an approximation, so the
coefficient `1/3` emerges from integration *exactly*.

This is the first lesson of the chapter in miniature: **choose the
representation to preserve the properties your algorithms need.** Differentiation
and integration are exact operations, so constants must be exact too.

### A tour of the algorithm

With the tree defined, everything else is a recursive function that walks it:

- `simplify(e)` rewrites a tree bottom-up: folds constant arithmetic, removes
  `x + 0` and `1·x` noise, and merges `x^a · x^b` into `x^(a+b)`.
- `deriv(e, x)` applies the calculus rules (sum, product, power, chain rule)
  and chains into `simplify` to keep output tidy.
- `integrate(e, x)` recognizes the limited family it can handle (polynomials,
  powers and trig/exp/log of linear arguments) and raises
  `NotImplementedError` rather than guessing.
- `evaluate(e, env)` turns a tree back into a number, which lets the program
  *numerically check* that its own symbolic answers are correct.

We now descend into each component, seeing the code in full before and after
explaining it.

---

## 2. The Expression Vocabulary

The file begins with a docstring that is also a grammar: a complete list of
the six node shapes the whole program will ever construct. In a CAS this is
the analogue of a language's grammar where every function that follows is simply a
case analysis over these tags.

```python
#!/usr/bin/env python3
"""Symbolic differentiation and integration (no third-party libraries).

Expressions are nested tuples:
    ('num', value)              constant (value is a Fraction)
    ('var', name)               variable
    ('add', a, b)               a + b
    ('mul', a, b)               a * b
    ('pow', base, exp)          base ** exp
    ('sin'|'cos'|'exp'|'log', x)
"""

from fractions import Fraction
```

Only one import is needed. `Fraction` is the exact-rational workhorse; `math`
is imported lazily later inside the numeric evaluator so that the symbolic
core never touches floats.

### Builder functions: the mini "DSL"

Because raw tuples are verbose, the program provides one small function per
node type. These are the *only* constructors in the codebase, which means the
invariant "every `num` holds a `Fraction`" can be enforced in exactly one
place.

```python
# ---------------------------------------------------------------- builders

def num(n):
    return ('num', n if isinstance(n, Fraction) else Fraction(n))


def var(name):
    return ('var', name)


def add(a, b):
    return ('add', a, b)


def sub(a, b):
    return ('add', a, ('mul', num(-1), b))


def mul(a, b):
    return ('mul', a, b)


def div(a, b):
    return ('mul', a, ('pow', b, num(-1)))


def pow(base, exp):
    return ('pow', base, exp)


def neg(a):
    return ('mul', num(-1), a)


def sin(a):
    return ('sin', a)


def cos(a):
    return ('cos', a)


def exp(a):
    return ('exp', a)


def log(a):
    return ('log', a)


ZERO = num(0)
ONE = num(1)


def is_num(e):
    return e[0] == 'num'


def contains_var(e, x):
    tag = e[0]
    if tag == 'var':
        return e[1] == x
    if tag == 'num':
        return False
    return any(contains_var(c, x) for c in e[1:])
```

**Why define `sub`, `div`, and `neg` at all?** Because they are *derived*
forms, not new node types. `sub(a, b)` becomes `a + (-1)·b`, `div(a, b)`
becomes `a · b^-1`, and `neg(a)` becomes `-1·a`. Keeping only six node types
dramatically shrinks the later case analyses: the simplifier, differentiator,
and integrator never have to handle subtraction, division, or negation as
special cases — they fall out of `add`, `mul`, and `pow` for free. This is a
recurring theme in compilers and CAS alike: **express features by desugaring
(translating them into simpler forms) rather than by adding new cases.**

Notice that every builder is a pure function with no state. A `num(3)` built
anywhere is identical to every other `num(3)`, which is why `ZERO` and `ONE`
are precomputed as module-level singletons and compared with `==` throughout
the code.

The two predicates round out the vocabulary. `is_num(e)` answers "is this a
constant?" `contains_var(e, x)` answers "does the variable `x` occur anywhere
in this tree?" — a structural question that drives the integration logic
later, where deciding whether a multiplier is a constant *depends on whether it
mentions the variable of integration*. The recursion is generic: for any
non-leaf node, check the node's children (`e[1:]`). This pattern — a recursive
walk that terminates at leaves — is the backbone of every function in the file.

---

## 3. Simplification: the Algebraic Engine

Raw differentiation produces correct but *ugly* trees. Differentiating `x^3`
by the rules below yields, without cleanup, something like

```
1 · 3 · x^2 + 0
```

A human would never write that. Simplification exists to apply the identities
everyone learns in school until the expression is in normal form: constants
folded, additive and multiplicative identities removed, numeric factors
normalized, and equal powers collected.

The workhorse is `simplify`, a single recursive dispatch over tags. Two small
helpers precede it.

### Helper: factoring a term into `(coefficient, rest)`

```python
# ---------------------------------------------------------------- simplify

def _decomp(t):
    """Return (coeff, rest) such that t == coeff * rest."""
    if is_num(t):
        return (t, ONE)
    if t[0] == 'mul' and is_num(t[1]):
        return (t[1], t[2])
    return (ONE, t)


def _merge_add(a, b):
    ca, ra = _decomp(a)
    cb, rb = _decomp(b)
    if ra == rb:
        return simplify(mul(simplify(add(ca, cb)), ra))
    return None
```

`_decomp` splits a term into a numeric coefficient and "the rest". For `3*x`
it returns `(3, x)`; for plain `x` it returns `(1, x)`; for a constant like
`4` it returns `(4, 1)`. This matters because of *collecting like terms*:
`3*x + 5*x` should simplify to `8*x`. `_merge_add` tries exactly that: it pulls
both terms apart, and if the "rest" is identical — `x` and `x` — it adds the
coefficients and rebuilds. If the rests differ (e.g. `3*x + 5*y`), it returns
`None`, signalling "no merge possible".

### The main simplifier

```python
def simplify(e):
    if not isinstance(e, tuple):
        raise TypeError(f"bad expr: {e!r}")
    tag = e[0]

    if tag in ('num', 'var'):
        return e

    if tag == 'add':
        a = simplify(e[1])
        b = simplify(e[2])
        if is_num(a) and is_num(b):
            return num(a[1] + b[1])
        if a == ZERO:
            return b
        if b == ZERO:
            return a
        merged = _merge_add(a, b)
        return merged if merged is not None else ('add', a, b)

    if tag == 'mul':
        a = simplify(e[1])
        b = simplify(e[2])
        if is_num(a) and is_num(b):
            return num(a[1] * b[1])
        if a == ZERO or b == ZERO:
            return ZERO
        if a == ONE:
            return b
        if b == ONE:
            return a
        # flatten numeric factors: c1 * (c2 * x)
        if is_num(a) and b[0] == 'mul' and is_num(b[1]):
            return simplify(mul(num(a[1] * b[1][1]), b[2]))
        if a[0] == 'mul' and is_num(a[1]) and is_num(b):
            return simplify(mul(num(a[1][1] * b[1]), a[2]))
        # normalize numeric factor to the left
        if is_num(b) and not is_num(a):
            return ('mul', b, a)
        # combine equal bases
        if a[0] == 'pow' and b[0] == 'pow' and a[1] == b[1]:
            return simplify(pow(a[1], add(a[2], b[2])))
        if a[0] == 'var' and b[0] == 'var' and a[1] == b[1]:
            return pow(a, num(2))
        if a[0] == 'var' and b[0] == 'pow' and b[1] == ('var', a[1]):
            return simplify(pow(a, add(ONE, b[2])))
        if b[0] == 'var' and a[0] == 'pow' and a[1] == ('var', b[1]):
            return simplify(pow(b, add(ONE, a[2])))
        return ('mul', a, b)

    if tag == 'pow':
        base = simplify(e[1])
        expo = simplify(e[2])
        if is_num(base) and is_num(expo) and expo[1].denominator == 1:
            return num(base[1] ** int(expo[1]))
        if expo == ZERO:
            return ONE
        if expo == ONE:
            return base
        return ('pow', base, expo)

    if tag in ('sin', 'cos', 'exp', 'log'):
        return (tag, simplify(e[1]))

    raise TypeError(f"unknown tag: {tag!r}")
```

Every case follows the same shape: first *recursively simplify the children*,
then *apply local rewrite rules*. This bottom-up discipline guarantees that by
the time a rule examines a child, that child is already in normal form.

Read each case as a small ordered list of identities:

- **`add`**: fold `2 + 3` into `5`; drop `0` (both sides); otherwise try to
  collect like terms via `_merge_add`.
- **`mul`**: fold `2 · 3`; annihilate with `0`; drop `·1`; then a set of
  structural rewrites that flatten nested numeric factors (`2·(3·x)` becomes
  `6·x`), move any numeric factor to the *left* (`x·3` becomes `3·x`), and
  combine equal powers (`x^a · x^b` becomes `x^(a+b)`). The last four
  `pow`/`var` cases spell out the combinations `x·x`, `x·x^b`, and `x^a·x`
  explicitly.
- **`pow`**: evaluate `2^3` into `8` *only* when the exponent is a whole
  number (`denominator == 1`) — otherwise `1/2` would be rounded; then apply
  `e^0 = 1` and `e^1 = e`.
- **`sin`/`cos`/`exp`/`log`**: recurse into the argument only.

Two things are worth lingering on. First, the constant-folding guard
`expo[1].denominator == 1` is the entire reason exactness survives: it refuses
to turn `x^(1/2)` into anything numeric, and only evaluates *constant* powers
whose exponent is an integer. Second, the simplification rules do **not** need
to be complete. The system does not attempt trigonometric identities
(`sin² + cos² = 1`) or full polynomial canonicalization. A simplifier in a
teaching CAS only has to be *good enough* that downstream output is readable
and that the self-checks can compare trees reliably. Over-simplifying is a
separate research project.

---

## 4. Pretty-Printing: Trees Back into Math

A CAS is judged as much by what it *prints* as by what it computes. The
printer's job is to render a tree as a human-readable linear string —
`2*x^3 - 5*x + 4`, `1/3*log(3*x + 2)` — with the right parentheses and no
redundant `+ -`, `1*`, or `*1`.

```python
# ---------------------------------------------------------------- printing

def _frac_str(n):
    return str(n.numerator) if n.denominator == 1 else f"{n.numerator}/{n.denominator}"


def _paren(e):
    s = to_str(e)
    return f"({s})" if e[0] in ('add', 'mul') else s


def _flatten_add(e, out):
    if e[0] == 'add':
        _flatten_add(e[1], out)
        _flatten_add(e[2], out)
    else:
        out.append(e)


def _exp_str(e):
    if e[0] == 'num':
        n = e[1]
        return str(n.numerator) if n.denominator == 1 else f"({_frac_str(n)})"
    if e[0] == 'var':
        return e[1]
    return f"({to_str(e)})"


def _is_neg(e):
    if is_num(e):
        return e[1] < 0
    return e[0] == 'mul' and is_num(e[1]) and e[1][1] < 0


def _neg_str(e):
    if is_num(e):
        return to_str(num(-e[1]))
    return to_str(mul(num(-e[1][1]), e[2]))


def to_str(e):
    tag = e[0]
    if tag == 'num':
        return _frac_str(e[1])
    if tag == 'var':
        return e[1]
    if tag == 'add':
        terms = []
        _flatten_add(e, terms)
        s = to_str(terms[0])
        for t in terms[1:]:
            if _is_neg(t):
                s += f" - {_neg_str(t)}"
            else:
                s += f" + {to_str(t)}"
        return s
    if tag == 'mul':
        a, b = e[1], e[2]
        if is_num(a):
            c = a[1]
            if c == -1:
                return f"-{_paren(b)}"
            if c == 1:
                return _paren(b)
            return f"{_frac_str(c)}*{_paren(b)}"
        return f"{_paren(a)}*{_paren(b)}"
    if tag == 'pow':
        return f"{_paren(e[1])}^{_exp_str(e[2])}"
    if tag in ('sin', 'cos', 'exp', 'log'):
        return f"{tag}({to_str(e[1])})"
    raise TypeError(f"unknown tag: {tag!r}")
```

The interesting decisions are all about *notation*, and each maps to a helper:

- `_frac_str` renders a `Fraction` as `3` when it is whole and `3/2` otherwise,
  so the output never shows `3.0` or `1_000_000/500_000`.
- `_flatten_add` converts the left-leaning binary tree
  `('add', ('add', a, b), c)` into the flat list `[a, b, c]`. Since `simplify`
  builds sums as left-leaning binary trees, flattening gives the printer a
  clean term list for free, and subtraction is printed as `a - b` whenever a
  term is recognized as negative by `_is_neg`.
- `_neg_str` then renders a negative term by flipping its sign. `_is_neg`
  returns true for a negative constant or for a product whose left factor is a
  negative number. This is what produces `2*x^3 - 5*x + 4` (with an infix minus)
  rather than the correct-but-ugly `2*x^3 + -5*x + 4`.
- `_paren` and `_exp_str` implement minimal, conservative parenthesization:
  wrap a `+`/`*` child in parentheses when it appears as a factor or exponent,
  and wrap a fractional exponent such as `1/2` to avoid the ambiguous `x^1/2`.
- The `mul` case drops the `1` coefficient entirely (`1*x` prints as `x`) and
  renders a coefficient of `-1` as a bare leading minus (`-x` instead of
  `-1*x`).

Because every rule lives in one function, the printer can never disagree with
the simplifier about what a "number" is — they both check `is_num` and the
`Fraction`. The result is compact output that a reader could hand back to a
calculator and verify.

---

## 5. Differentiation: Rules as Recursion

Differentiation is the cleanest algorithm in the book, because calculus gives
us a *complete, compositional* set of rules. If you know the derivative of the
parts, you know the derivative of the whole. The implementation is therefore a
one-to-one transcription of the table:

| Form        | Rule                                    |
|-------------|-----------------------------------------|
| constant    | `0`                                     |
| `x`         | `1` (or `0` if a *different* variable)  |
| `a + b`     | `a' + b'`                               |
| `a · b`     | `a'·b + a·b'`  (product rule)           |
| `f(x)^c`    | `c · f(x)^(c-1) · f'(x)` (power rule)   |
| `c^f(x)`    | `c^f(x) · log(c) · f'(x)`               |
| `sin u`     | `cos u · u'` (chain rule)               |
| `cos u`     | `-sin u · u'`                           |
| `exp u`     | `exp u · u'`                            |
| `log u`     | `u' / u`                                |

```python
# ---------------------------------------------------------------- derivative

def deriv(e, x):
    tag = e[0]
    if tag == 'num':
        return ZERO
    if tag == 'var':
        return ONE if e[1] == x else ZERO
    if tag == 'add':
        return simplify(add(deriv(e[1], x), deriv(e[2], x)))
    if tag == 'mul':
        a, b = e[1], e[2]
        return simplify(add(mul(deriv(a, x), b), mul(a, deriv(b, x))))
    if tag == 'pow':
        base, expo = e[1], e[2]
        if not contains_var(expo, x):
            return simplify(mul(mul(expo, pow(base, sub(expo, ONE))), deriv(base, x)))
        if not contains_var(base, x):
            return simplify(mul(mul(pow(base, expo), log(base)), deriv(expo, x)))
        raise NotImplementedError(f"cannot differentiate {to_str(e)}")
    inner = deriv(e[1], x)
    if tag == 'sin':
        return simplify(mul(cos(e[1]), inner))
    if tag == 'cos':
        return simplify(mul(neg(sin(e[1])), inner))
    if tag == 'exp':
        return simplify(mul(exp(e[1]), inner))
    if tag == 'log':
        return simplify(mul(div(ONE, e[1]), inner))
    raise TypeError(f"unknown tag: {tag!r}")
```

Two details deserve emphasis, because they carry the whole design.

**The power rule has two branches, distinguished by `contains_var`.** When the
exponent is free of `x` (`x^2`, `(2x+1)^3`, `√x`), we use the familiar power
rule `c · base^(c-1) · base'`. When instead the *base* is free of `x`
(`2^x`, `a^sin(x)`), we use the exponential rule `base^expo · log(base) ·
expo'`. When *both* base and exponent mention `x` (`x^x`), differentiation is
genuinely harder and the honest answer is `NotImplementedError` — a decision
we will see echoed in the integrator. The guard
`expo[1].denominator == 1` inside `simplify` guarantees that `2^3` (an integer
exponent) collapses numerically, while `2^x` stays symbolic so the `log(2)`
branch can fire.

**Every rule wraps its answer in `simplify`.** Differentiation *constructs*
trees; simplification *normalizes* them. Without the wrapper, the derivative
of `x^3` would be `3 · x^2 · 1 + ...` (the `·1` and `+0` terms from the
product and power rules). With it, the output is `3*x^2`. This pairing —
generate, then normalize — is the standard idiom of symbolic computing, and it
is why the simplifier was built first.

The chain rule appears entirely inside the four leaf cases: each computes the
derivative of its *argument* (`inner`) exactly once and multiplies it onto the
outer derivative. Because `deriv` is recursive, nesting works automatically —
differentiating `sin(3*x + 1)` recurses through `sin` into `add` into `mul`
and assembles `3·cos(3x+1)` on the way back out.

---

## 6. Integration: Recognizing Patterns

Integration is where the chapter's honesty shows. Unlike differentiation,
there is **no simple compositional algorithm** for antiderivatives in general;
a complete method (Risch-style) is the subject of graduate study. So this
implementation integrates only a *pragmatic subset* — and, crucially, it
*detects* the boundary of that subset and refuses politely rather than
returning a wrong answer.

The subset is built on one recurring idea: **linear arguments**. Many
elementary antiderivatives are known in the form with `x` alone; the constant
`∫ f(a·x + b) dx` is handled by a change of variable that divides by `a`. The
function `linear_arg` is the test that recognizes the pattern `a·x + b`:

```python
# ---------------------------------------------------------------- integrate

def linear_arg(e, x):
    """Return (a, b) if e == a*x + b with a, b independent of x, else None."""
    e = simplify(e)
    if not contains_var(e, x):
        return (ZERO, e)
    if e[0] == 'var' and e[1] == x:
        return (ONE, ZERO)
    if e[0] == 'add':
        la = linear_arg(e[1], x)
        lb = linear_arg(e[2], x)
        if la is not None and lb is not None:
            return (simplify(add(la[0], lb[0])), simplify(add(la[1], lb[1])))
        return None
    if e[0] == 'mul':
        a, b = e[1], e[2]
        if not contains_var(a, x):
            rb = linear_arg(b, x)
            if rb is not None:
                return (simplify(mul(a, rb[0])), simplify(mul(a, rb[1])))
        if not contains_var(b, x):
            ra = linear_arg(a, x)
            if ra is not None:
                return (simplify(mul(b, ra[0])), simplify(mul(b, ra[1])))
    return None
```

`linear_arg` returns the tuple `(a, b)` meaning "this expression equals
`a·x + b`", or `None` if it does not. A constant is `0·x + c`; the variable
`x` is `1·x + 0`; a sum is linear if both sides are, with coefficients added;
a product `c·(a·x+b)` is linear if the non-`x` factor is a constant and the
other side is linear, with `c` folded into both coefficients. This is exactly
the kind of "does this expression match this shape?" predicate that a CAS
centralizes, so that several integration rules can share it.

### Integrating powers: `f(x)^c` and `c^(a·x+b)`

```python
def integrate_pow(e, x):
    base = simplify(e[1])
    expo = simplify(e[2])

    if not contains_var(expo, x):          # f(x)^c
        la = linear_arg(base, x)
        if la is None:
            raise NotImplementedError(f"cannot integrate {to_str(e)}")
        a, _ = la
        if a == ZERO:
            return simplify(mul(e, var(x)))
        if expo == num(-1):
            return simplify(mul(div(ONE, a), log(base)))
        return simplify(mul(div(ONE, mul(a, add(expo, ONE))), pow(base, add(expo, ONE))))

    if not contains_var(base, x):          # c^(a*x+b)
        lb = linear_arg(expo, x)
        if lb is None:
            raise NotImplementedError(f"cannot integrate {to_str(e)}")
        a, _ = lb
        if a == ZERO:
            return simplify(mul(e, var(x)))
        return simplify(div(e, mul(a, log(base))))

    raise NotImplementedError(f"cannot integrate {to_str(e)}")
```

The two branches mirror the two branches of `deriv`'s power case. In the first
(`f(x)^c`), if the base is linear, a change of variables gives
`∫ (a·x+b)^c dx = (a·x+b)^(c+1) / (a·(c+1))`, with the special case `c = -1`
handled separately as `log(base)/a`. The `a == ZERO` guard catches a base that
is actually constant (e.g. `3^x` reduced to a base free of `x` — that can't
happen here, but the guard keeps the code safe). In the second branch
(`c^(a·x+b)`), `∫ c^(a·x+b) dx = c^(a·x+b) / (a·log(c))`.

### Integrating `sin`/`cos`/`exp`/`log` of linear arguments

```python
def integrate_linear_fn(e, x):
    tag = e[0]
    inner = e[1]
    la = linear_arg(inner, x)
    if la is None:
        return None
    a, _ = la
    if a == ZERO:
        return simplify(mul(e, var(x)))
    inv = div(ONE, a)
    if tag == 'sin':
        return simplify(mul(neg(inv), cos(inner)))
    if tag == 'cos':
        return simplify(mul(inv, sin(inner)))
    if tag == 'exp':
        return simplify(mul(inv, exp(inner)))
    if tag == 'log':
        return simplify(mul(inv, sub(mul(inner, log(inner)), inner)))
    return None
```

Each row of this table is a single calculus fact scaled by `1/a` for the
linear argument:

`∫ sin(a·x+b) dx = -cos(a·x+b)/a`, `∫ cos(a·x+b) dx = sin(a·x+b)/a`,
`∫ exp(a·x+b) dx = exp(a·x+b)/a`, and
`∫ log(u) dx = (u·log(u) - u)/a` where `u = a·x+b`.

The `log` case is worth pausing on: the antiderivative of `log(u)` is
`u·log(u) - u` (verify by the product rule and chain rule). This is the one
rule in the file whose result is not just "scale the outer function" — it
genuinely restructures the expression — and the self-check in the next section
will confirm it numerically.

### The top-level dispatcher

```python
def integrate(e, x):
    e = simplify(e)
    tag = e[0]

    if tag == 'num':
        return simplify(mul(e, var(x)))
    if tag == 'var':
        if e[1] == x:
            return simplify(mul(num(Fraction(1, 2)), pow(e, num(2))))
        return simplify(mul(e, var(x)))
    if tag == 'add':
        return simplify(add(integrate(e[1], x), integrate(e[2], x)))
    if tag == 'mul':
        a, b = e[1], e[2]
        if not contains_var(a, x):
            return simplify(mul(a, integrate(b, x)))
        if not contains_var(b, x):
            return simplify(mul(b, integrate(a, x)))
        raise NotImplementedError(f"cannot integrate {to_str(e)}")
    if tag == 'pow':
        return integrate_pow(e, x)
    if tag in ('sin', 'cos', 'exp', 'log'):
        res = integrate_linear_fn(e, x)
        if res is None:
            raise NotImplementedError(f"cannot integrate {to_str(e)}")
        return res

    raise TypeError(f"unknown tag: {tag!r}")
```

The dispatcher has the same shape as `deriv`, with one crucial behavioral
difference: **its `mul` case is only partially recursive.** If one factor is a
constant (free of `x`), it pulls the constant out — `∫ c·f(x) dx = c·∫f(x) dx`.
But if *both* factors mention `x` (as in `x·exp(x)`), it raises
`NotImplementedError`. That restriction is the honest boundary of the subset:
integrating `x·exp(x)` needs integration by parts, which is out of scope.

This "recognize or refuse" discipline is the most important idea in the
chapter after the representation itself. A symbolic integrator that *guesses*
is far worse than one that says no — a wrong "simplified" answer silently
corrupts everything downstream. The `NotImplementedError` carrying the
expression (`cannot integrate x*exp(x)`) makes the system fail loudly and
explain itself.

---

## 7. Numeric Evaluation: Closing the Loop

`evaluate` is the mirror image of the earlier functions: instead of building
trees, it destroys them into floats. It is the bridge between the symbolic
world and the numeric world, and it exists for one purpose — **verification**.

```python
# ---------------------------------------------------------------- numeric eval

def evaluate(e, env):
    e = simplify(e)
    tag = e[0]
    if tag == 'num':
        return float(e[1])
    if tag == 'var':
        return env[e[1]]
    if tag == 'add':
        return evaluate(e[1], env) + evaluate(e[2], env)
    if tag == 'mul':
        return evaluate(e[1], env) * evaluate(e[2], env)
    if tag == 'pow':
        return evaluate(e[1], env) ** evaluate(e[2], env)
    if tag == 'sin':
        import math
        return math.sin(evaluate(e[1], env))
    if tag == 'cos':
        import math
        return math.cos(evaluate(e[1], env))
    if tag == 'exp':
        import math
        return math.exp(evaluate(e[1], env))
    if tag == 'log':
        import math
        return math.log(evaluate(e[1], env))
    raise TypeError(f"unknown tag: {tag!r}")
```

The `env` argument is an ordinary `dict` mapping variable names to values,
e.g. `{'x': 0.8, 'y': 1.3}`. A `var` node simply looks itself up; a `num`
becomes a `float` (this is the one place exactness is deliberately sacrificed,
and only for the numeric check). The `math` functions are imported *inside*
the leaf cases so the symbolic core never imports `math` at all — a small
encapsulation that keeps the "pure" part of the library free of floats.

This function is deliberately trivial because its correctness is assumed and
then *used to test* the more interesting functions. That inversion — using
simple code to check clever code — is the theme of the next section.

---

## 8. The Self-Verifying Demo

The `main()` function is not just a demo; it is a *test suite written as a
narrative*. It builds thirty-one expressions, and for each one:

1. prints the function and its symbolic derivative,
2. checks the derivative against a central finite-difference estimate,
3. integrates, prints the antiderivative,
4. differentiates it back, and checks that `d/dx(∫f) = f` numerically.

```python
# ---------------------------------------------------------------- main

def main():
    x = var('x')
    y = var('y')

    examples = [
        ("constant", num(7)),
        ("x", x),
        ("3*x", mul(num(3), x)),
        ("x^2", pow(x, num(2))),
        ("x^3", pow(x, num(3))),
        ("x^2 + 3*x", add(pow(x, num(2)), mul(num(3), x))),
        ("2*x^3 - 5*x + 4",
         add(mul(num(2), pow(x, num(3))), add(mul(num(-5), x), num(4)))),
        ("1/x", pow(x, num(-1))),
        ("1/x^2", pow(x, num(-2))),
        ("sqrt(x)", pow(x, num(Fraction(1, 2)))),
        ("x^(3/2)", pow(x, num(Fraction(3, 2)))),
        ("sin(x)", sin(x)),
        ("cos(x)", cos(x)),
        ("exp(x)", exp(x)),
        ("log(x)", log(x)),
        ("sin(2*x)", sin(mul(num(2), x))),
        ("cos(3*x + 1)", cos(add(mul(num(3), x), num(1)))),
        ("exp(-2*x)", exp(mul(num(-2), x))),
        ("log(2*x + 3)", log(add(mul(num(2), x), num(3)))),
        ("2*sin(x)", mul(num(2), sin(x))),
        ("3*x^2", mul(num(3), pow(x, num(2)))),
        ("x^2/2", mul(num(Fraction(1, 2)), pow(x, num(2)))),
        ("(2*x + 1)^3", pow(add(mul(num(2), x), num(1)), num(3))),
        ("1/(3*x + 2)", pow(add(mul(num(3), x), num(2)), num(-1))),
        ("2^x", pow(num(2), x)),
        ("y", y),
        ("y*x", mul(y, x)),
        ("exp(x) + sin(x)", add(exp(x), sin(x))),
        ("x*exp(x)", mul(x, exp(x))),          # needs integration by parts
        ("x*sin(x)", mul(x, sin(x))),          # needs integration by parts
        ("x*cos(x) + sin(x)", add(mul(x, cos(x)), sin(x))),  # d(x sin x)
    ]

    x0, tol = 0.8, 1e-6
    env = {'x': x0, 'y': 1.3}

    for name, f in examples:
        f = simplify(f)
        print(f"f = {name!r}  :  {to_str(f)}")

        d = simplify(deriv(f, 'x'))
        print(f"  d/dx = {to_str(d)}")

        # numeric check of derivative (central finite difference)
        h = 1e-6
        fd = (evaluate(f, {'x': x0 + h, 'y': 1.3})
              - evaluate(f, {'x': x0 - h, 'y': 1.3})) / (2 * h)
        ok = abs(evaluate(d, env) - fd) <= tol * max(1.0, abs(fd))
        print(f"  derivative check: {'OK' if ok else 'FAIL'}")

        try:
            i = integrate(f, 'x')
            print(f"  ∫ dx = {to_str(i)}")

            back = simplify(deriv(i, 'x'))
            print(f"  d/dx(∫) = {to_str(back)}")
            ok2 = abs(evaluate(back, env) - evaluate(f, env)) <= tol * max(1.0, abs(evaluate(f, env)))
            print(f"  integral check: {'OK' if ok2 else 'FAIL'}")
        except NotImplementedError as err:
            print(f"  integrate: not supported ({err})")
        print()


if __name__ == '__main__':
    main()
```

Three engineering details here are the difference between "prints some output"
and "actually proves correctness".

**Central finite differences.** The derivative check estimates
`f'(x₀)` as `(f(x₀+h) - f(x₀-h)) / (2h)` with `h = 1e-6`. The *central*
form (as opposed to the one-sided `(f(x₀+h)-f(x₀))/h`) is second-order
accurate, meaning its error shrinks as `h²`. At `h = 1e-6` the error is around
`1e-12` in the function values, comfortably inside the `1e-6` tolerance —
*provided* `h` is not so small that floating-point cancellation in the
subtraction round the difference to zero, which is why `h = 1e-6` rather than
`1e-30`.

**A relative tolerance.** The check
`abs(symbolic - numeric) <= tol * max(1.0, abs(numeric))` scales the tolerance
to the magnitude of the compared value. For large values (`2^x` at `x = 0.8`
times `log 2`, say) an absolute `1e-6` bound would be too strict relative to
floating-point roundoff; the `max(1.0, ...)` keeps the test meaningful for both
tiny and huge values.

**Differentiating the antiderivative back.** The integral is verified by the
fundamental theorem itself: compute `∫f`, differentiate it, and confirm the
result equals `f` numerically. This is a beautifully *self-contained* check —
it never requires a reference CAS to compare against, only the two operations
the program already implements. If either the integrator or the differentiator
were broken in a way that affected these expressions, the `OK`/`FAIL` lines
would expose it.

---

## 9. Running the Code

Save the listing above as `sym-math.py` (the filename contains a hyphen, which
is why the README shows how to load it with `importlib` rather than a plain
`import`). Run it from the directory:

```bash
python3 sym-math.py
```

There is nothing to install; the program uses only `fractions` and `math`.
The output is long — thirty-one examples, each a small block — so below we
reproduce it in full as the program actually prints it:

```
f = 'constant'  :  7
  d/dx = 0
  derivative check: OK
  ∫ dx = 7*x
  d/dx(∫) = 7
  integral check: OK

f = 'x'  :  x
  d/dx = 1
  derivative check: OK
  ∫ dx = 1/2*x^2
  d/dx(∫) = x
  integral check: OK

f = '3*x'  :  3*x
  d/dx = 3
  derivative check: OK
  ∫ dx = 3/2*x^2
  d/dx(∫) = 3*x
  integral check: OK

f = 'x^2'  :  x^2
  d/dx = 2*x
  derivative check: OK
  ∫ dx = 1/3*x^3
  d/dx(∫) = x^2
  integral check: OK

f = 'x^3'  :  x^3
  d/dx = 3*x^2
  derivative check: OK
  ∫ dx = 1/4*x^4
  d/dx(∫) = x^3
  integral check: OK

f = 'x^2 + 3*x'  :  x^2 + 3*x
  d/dx = 2*x + 3
  derivative check: OK
  ∫ dx = 1/3*x^3 + 3/2*x^2
  d/dx(∫) = x^2 + 3*x
  integral check: OK

f = '2*x^3 - 5*x + 4'  :  2*x^3 - 5*x + 4
  d/dx = 6*x^2 - 5
  derivative check: OK
  ∫ dx = 1/2*x^4 - 5/2*x^2 + 4*x
  d/dx(∫) = 2*x^3 - 5*x + 4
  integral check: OK

f = '1/x'  :  x^-1
  d/dx = -x^-2
  derivative check: OK
  ∫ dx = log(x)
  d/dx(∫) = x^-1
  integral check: OK

f = '1/x^2'  :  x^-2
  d/dx = -2*x^-3
  derivative check: OK
  ∫ dx = -x^-1
  d/dx(∫) = x^-2
  integral check: OK

f = 'sqrt(x)'  :  x^(1/2)
  d/dx = 1/2*x^(-1/2)
  derivative check: OK
  ∫ dx = 2/3*x^(3/2)
  d/dx(∫) = x^(1/2)
  integral check: OK

f = 'x^(3/2)'  :  x^(3/2)
  d/dx = 3/2*x^(1/2)
  derivative check: OK
  ∫ dx = 2/5*x^(5/2)
  d/dx(∫) = x^(3/2)
  integral check: OK

f = 'sin(x)'  :  sin(x)
  d/dx = cos(x)
  derivative check: OK
  ∫ dx = -cos(x)
  d/dx(∫) = sin(x)
  integral check: OK

f = 'cos(x)'  :  cos(x)
  d/dx = -sin(x)
  derivative check: OK
  ∫ dx = sin(x)
  d/dx(∫) = cos(x)
  integral check: OK

f = 'exp(x)'  :  exp(x)
  d/dx = exp(x)
  derivative check: OK
  ∫ dx = exp(x)
  d/dx(∫) = exp(x)
  integral check: OK

f = 'log(x)'  :  log(x)
  d/dx = x^-1
  derivative check: OK
  ∫ dx = x*log(x) - x
  d/dx(∫) = log(x) + 1 - 1
  integral check: OK

f = 'sin(2*x)'  :  sin(2*x)
  d/dx = 2*cos(2*x)
  derivative check: OK
  ∫ dx = -1/2*cos(2*x)
  d/dx(∫) = sin(2*x)
  integral check: OK

f = 'cos(3*x + 1)'  :  cos(3*x + 1)
  d/dx = -3*sin(3*x + 1)
  derivative check: OK
  ∫ dx = 1/3*sin(3*x + 1)
  d/dx(∫) = cos(3*x + 1)
  integral check: OK

f = 'exp(-2*x)'  :  exp(-2*x)
  d/dx = -2*exp(-2*x)
  derivative check: OK
  ∫ dx = -1/2*exp(-2*x)
  d/dx(∫) = exp(-2*x)
  integral check: OK

f = 'log(2*x + 3)'  :  log(2*x + 3)
  d/dx = 2*(2*x + 3)^-1
  derivative check: OK
  ∫ dx = 1/2*((2*x + 3)*log(2*x + 3) - (2*x + 3))
  d/dx(∫) = 1/2*(2*log(2*x + 3) + (2*x + 3)*(2*(2*x + 3)^-1) - 2)
  integral check: OK

f = '2*sin(x)'  :  2*sin(x)
  d/dx = 2*cos(x)
  derivative check: OK
  ∫ dx = -2*cos(x)
  d/dx(∫) = 2*sin(x)
  integral check: OK

f = '3*x^2'  :  3*x^2
  d/dx = 6*x
  derivative check: OK
  ∫ dx = x^3
  d/dx(∫) = 3*x^2
  integral check: OK

f = 'x^2/2'  :  1/2*x^2
  d/dx = x
  derivative check: OK
  ∫ dx = 1/6*x^3
  d/dx(∫) = 1/2*x^2
  integral check: OK

f = '(2*x + 1)^3'  :  (2*x + 1)^3
  d/dx = 6*(2*x + 1)^2
  derivative check: OK
  ∫ dx = 1/8*(2*x + 1)^4
  d/dx(∫) = (2*x + 1)^3
  integral check: OK

f = '1/(3*x + 2)'  :  (3*x + 2)^-1
  d/dx = -3*(3*x + 2)^-2
  derivative check: OK
  ∫ dx = 1/3*log(3*x + 2)
  d/dx(∫) = (3*x + 2)^-1
  integral check: OK

f = '2^x'  :  2^x
  d/dx = 2^x*log(2)
  derivative check: OK
  ∫ dx = 2^x*log(2)^-1
  d/dx(∫) = (2^x*log(2))*log(2)^-1
  integral check: OK

f = 'y'  :  y
  d/dx = 0
  derivative check: OK
  ∫ dx = y*x
  d/dx(∫) = y
  integral check: OK

f = 'y*x'  :  y*x
  d/dx = y
  derivative check: OK
  ∫ dx = y*(1/2*x^2)
  d/dx(∫) = y*x
  integral check: OK

f = 'exp(x) + sin(x)'  :  exp(x) + sin(x)
  d/dx = exp(x) + cos(x)
  derivative check: OK
  ∫ dx = exp(x) - cos(x)
  d/dx(∫) = exp(x) + sin(x)
  integral check: OK

f = 'x*exp(x)'  :  x*exp(x)
  d/dx = exp(x) + x*exp(x)
  derivative check: OK
  integrate: not supported (cannot integrate x*exp(x))

f = 'x*sin(x)'  :  x*sin(x)
  d/dx = sin(x) + x*cos(x)
  derivative check: OK
  integrate: not supported (cannot integrate x*sin(x))

f = 'x*cos(x) + sin(x)'  :  x*cos(x) + sin(x)
  d/dx = cos(x) + x*(-sin(x)) + cos(x)
  derivative check: OK
  integrate: not supported (cannot integrate x*cos(x))
```

Every example ends with `OK` on both checks, and the final three examples
report `integrate: not supported` — exactly as designed.

---

## 10. Interpreting the Results

The output a
sequence of small mathematical arguments, each independently verified:

**The `derivative check: OK` lines prove the symbolic differentiator agrees
with numeric reality.** Every derivative — constant, power, product, chain
rule, the two power-rule branches, and the four transcendental functions — is
compared against a central finite-difference estimate at `x = 0.8`. If any
rule were implemented with a wrong sign or a missing chain factor, that single
numeric comparison would catch it. The fact that all thirty-one pass is
strong evidence that the rule table is correct, not just that the code parses.

**The `integral check: OK` lines prove the integrator inverts the
differentiator, the deepest property of the whole system.** The program does
not rely on a reference table of integrals. It computes `∫f`, then differentiates
the answer, then confirms `d/dx(∫f) = f` at the same `x`. This is the
Fundamental Theorem of Calculus used as a *runtime assertion*. Notice that
this check exercises the two hardest rules jointly: for `log(2*x + 3)` the
printed antiderivative is `1/2*((2*x + 3)*log(2*x + 3) - (2*x + 3))`, and its
rederivative — `1/2*(2*log(2*x + 3) + (2*x + 3)*(2*(2*x + 3)^-1) - 2)` — is
an ugly expression that only *simplifies numerically* to the original. The
`OK` on that line is genuine, non-trivial verification of the `log` integration
rule.

**The output also demonstrates the two design properties we set out to
achieve.** First, *exactness*: coefficients appear as `1/3`, `2/5`, `3/2` —
rational numbers, not `0.333333…`. This is the `Fraction`-based representation
paying off; nothing was ever rounded. Second, *honesty at the boundary*: the
last three expressions (`x·exp(x)`, `x·sin(x)`, `x·cos(x) + sin(x)`) each get a
correct derivative (`x·exp(x)` → `exp(x) + x·exp(x)`) but then print
`integrate: not supported` instead of a guessed answer. Those integrals require
integration by parts — deliberately out of scope — and the system says so in
plain text. A production CAS would integrate them; a teaching CAS is *more*
correct for refusing than for silently returning a wrong result.

**The `2^x` case quietly demonstrates both power-rule branches working in
tandem.** Its derivative is `2^x*log(2)` (the `c^f(x)` branch of `deriv`); its
integral is `2^x*log(2)^-1` (the `c^(a·x+b)` branch of `integrate_pow`). The
inverse relationship between the two is exactly the `1/log(2)` factor flipping
between numerator and denominator — a one-line illustration that
differentiation and integration are true inverses here, confirmed numerically.

**Two cosmetic oddities in the output are worth understanding, not
"fixing".** First, `1/x` prints as `x^-1` and its integral prints as `log(x)`
— negative exponents are kept as powers (they fall out of the power rule), so
the printer never writes them as fractions. Second, in the `log(x)` example the
line `d/dx(∫) = log(x) + 1 - 1` shows an expression that is numerically `log(x)`
but not fully simplified, because the simplifier does not cancel the `+1 - 1`
that arises from the product rule applied to `x·log(x)`. The numeric check
still passes — `1 - 1` cancels exactly in floating point — but it is a visible
reminder that the simplifier is a *subset* of a real CAS, not a complete one.

---

## 11. What to Build Next

The architecture makes each enhancement a local change, which is the real test
of a clean symbolic design:

- **More transcendental functions** — `tan`, `asin`, `sqrt` as a distinct node:
  add one builder, one derivative case, one integral case, and one printer case.
- **Integration by parts** — recognize the `mul`-of-two-`x`-dependent-factors
  pattern and apply `∫u dv = uv - ∫v du`; this would turn the three
  `not supported` lines into answers.
- **A full simplifier** — normalize polynomials into sorted monomial form,
  cancel `+1 - 1`, and combine `log` factors; this removes the cosmetic
  `log(x) + 1 - 1` artifact.
- **A parser** — turn the *string* `"x^2 + 3*x"` into the tree; this inverts
  `to_str` and rounds out the builder DSL into a true little language.

Each suggestion extends the same loop: **represent → simplify → differentiate
→ integrate → verify**. That loop, not any single rule, is the enduring
takeaway. A computer algebra system is not magic; it is a data structure, a
handful of recursive rewrite rules, and the discipline to refuse what it does
not know.
