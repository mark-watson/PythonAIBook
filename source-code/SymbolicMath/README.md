# Symbolic Math in Pure Python

A self-contained symbolic differentiation and integration library written in
plain Python 3. It uses **no third-party dependencies** — only the standard
library (`fractions.Fraction` and `math`).

The system models mathematical expressions as nested tuples, applies
algebraic simplification, computes exact derivatives and antiderivatives
symbolically, and can evaluate expressions numerically to verify its own
results.

Part of the *PythonAIBook* source code collection.

## Features

- **Symbolic expression tree** built from nested tuples:
  `('num', value)`, `('var', name)`, `('add', a, b)`, `('mul', a, b)`,
  `('pow', base, exp)`, `('sin'|'cos'|'exp'|'log', arg)`
- **Exact arithmetic** with `fractions.Fraction` (no floating point drift
  during simplification; e.g. `1/3*x^3` stays exact)
- **Algebraic simplification**: constant folding, `x + 0 = x`, `1*x = x`,
  `0*x = 0`, combining powers (`x^a * x^b = x^(a+b)`), factor flattening and
  normalization
- **Symbolic differentiation** (`deriv`) with the chain rule, product rule,
  power rule, and derivatives of `sin`, `cos`, `exp`, `log`
- **Symbolic integration** (`integrate`) for a pragmatic subset: polynomials,
  powers with linear arguments (`(a·x + b)^c`), `a^x`, and `sin`/`cos`/`exp`/
  `log` of linear arguments (`a·x + b`), plus products of a constant with any
  integrable expression
- **Numeric evaluation** (`evaluate`) via `math` functions
- **Self-verifying demo**: the `main()` driver checks every derivative against
  a central finite-difference approximation and every integral by
  differentiating the result back and comparing numerically

## Supported Expression Form

| Form          | Meaning              | Derivative        | Integral (`∫ f dx`)          |
|---------------|----------------------|-------------------|------------------------------|
| `c`           | constant             | `0`               | `c·x`                        |
| `x^n`         | power of variable    | `n·x^(n-1)`       | `x^(n+1)/(n+1)` (n ≠ −1)     |
| `1/x`         | reciprocal           | `-x^-2`           | `log(x)`                     |
| `a^x`         | exponential          | `a^x·log(a)`      | `a^x / log(a)`               |
| `sin(kx+b)`   | sine                 | `k·cos(kx+b)`     | `-cos(kx+b)/k`               |
| `cos(kx+b)`   | cosine               | `-k·sin(kx+b)`    | `sin(kx+b)/k`                |
| `exp(kx+b)`   | exponential          | `k·exp(kx+b)`     | `exp(kx+b)/k`                |
| `log(kx+b)`   | natural log          | `k/(kx+b)`        | `(u·log(u) − u)/k`, u=kx+b   |

Integration is **not** a full Risch-style algorithm: expressions that would
need integration by parts (e.g. `x·exp(x)`, `x·sin(x)`) are reported as
"not supported" rather than silently returning a wrong answer.

## Files

| File          | Description                                                        |
|---------------|--------------------------------------------------------------------|
| `sym-math.py` | The complete library plus a self-testing `main()` demo (one file)  |
| `README.md`   | This document                                                      |

## Requirements

- Python 3.8+ (uses only the standard library)
- No `pip install` needed

## Running the Code

From this directory:

```bash
python3 sym-math.py
```

or make it executable and run directly:

```bash
chmod +x sym-math.py
./sym-math.py
```

Each example prints the function, its derivative, a finite-difference check
of the derivative, the antiderivative, and an integral check
(`d/dx ∫f − f ≈ 0`):

```
f = 'x^2'  :  x^2
  d/dx = 2*x
  derivative check: OK
  ∫ dx = 1/3*x^3
  d/dx(∫) = x^2
  integral check: OK
```

Expect a clean run with every check reporting `OK`, and three expressions
(`x*exp(x)`, `x*sin(x)`, `x*cos(x) + sin(x)`) whose derivatives are computed
correctly but whose integrals are reported as `not supported` (they require
integration by parts).

## Using the Library Programmatically

The filename contains a hyphen (`sym-math.py`), which is not a valid Python
identifier, so load it explicitly with `importlib` and build expressions with
the little builder DSL:

```python
import importlib.util

spec = importlib.util.spec_from_file_location('sym_math', 'sym-math.py')
sym_math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sym_math)

var, num, add, mul, pow = (sym_math.var, sym_math.num, sym_math.add,
                           sym_math.mul, sym_math.pow)
to_str, deriv, integrate, evaluate = (sym_math.to_str, sym_math.deriv,
                                      sym_math.integrate, sym_math.evaluate)

x = var('x')

f = add(pow(x, num(2)), mul(num(3), x))   # x^2 + 3*x

print(to_str(f))                          # "x^2 + 3*x"
print(to_str(deriv(f, 'x')))              # "2*x + 3"
print(to_str(integrate(f, 'x')))          # "1/3*x^3 + 3/2*x^2"
print(evaluate(f, {'x': 2.0}))            # 10.0
```

### The builder functions

| Builder     | Expression      | Example                                  |
|-------------|-----------------|------------------------------------------|
| `num(n)`    | constant        | `num(3)`, `num(Fraction(1, 2))`          |
| `var(n)`    | variable        | `var('x')`                               |
| `add(a, b)` | `a + b`         | `add(x, num(1))`                         |
| `sub(a, b)` | `a − b`         | `sub(x, num(1))`                         |
| `mul(a, b)` | `a · b`         | `mul(num(2), x)`                         |
| `div(a, b)` | `a / b`         | `div(num(1), x)`                         |
| `pow(a, b)` | `a ** b`        | `pow(x, num(2))`                         |
| `neg(a)`    | `−a`            | `neg(sin(x))`                            |
| `sin/...`   | function        | `sin(x)`, `cos(x)`, `exp(x)`, `log(x)`   |

Numerical constants supplied to `num()` are converted to `Fraction`
automatically, so `num(0.5)` and `num(Fraction(1, 2))` are equivalent.

### Key API

- `simplify(e)` — normalize an expression tree (constants folded, identities
  applied, powers combined)
- `deriv(e, x)` — symbolic derivative of `e` with respect to variable `x`
- `integrate(e, x)` — symbolic antiderivative of `e` w.r.t. `x` (raises
  `NotImplementedError` if the expression is not in the supported subset)
- `to_str(e)` — human-readable rendering, e.g. `2*x^3 - 5*x + 4`,
  `1/3*log(3*x + 2)`, `x^-1/2`
- `evaluate(e, env)` — numeric evaluation with a dict of variable values,
  e.g. `evaluate(f, {'x': 0.8, 'y': 1.3})`
- `contains_var(e, x)` — test whether expression `e` mentions variable `x`

## How It Works

1. **Representation.** Every expression is a nested tuple whose first element
   is a tag (`'num'`, `'var'`, `'add'`, `'mul'`, `'pow'`, `'sin'`, ...).
   Constants are stored as `fractions.Fraction` values so all algebra is exact.
2. **Simplification.** `simplify()` recursively rewrites the tree: folds
   constant arithmetic, drops `+0`/`·1`/`·0` identities, merges `x^a · x^b`
   into `x^(a+b)`, and normalizes numeric factors to the left of a product.
3. **Differentiation.** `deriv()` applies the standard rules recursively —
   constant, variable, sum, product, power (with both `x^c` and `c^x` forms),
   and chain rule for `sin`/`cos`/`exp`/`log` — then simplifies the result.
4. **Integration.** `integrate()` recognizes the supported forms by first
   checking whether a sub-expression is *linear in x* (`a·x + b` via
   `linear_arg()`), which lets it handle `(a·x+b)^c`, `sin(a·x+b)`,
   `log(a·x+b)`, etc. Results are checked empirically in `main()` by
   differentiating them back.
5. **Verification.** Derivatives are compared against a central
   finite-difference estimate; integrals are verified by `d/dx(∫f) = f`
   evaluated numerically at `x = 0.8`, `y = 1.3`.

## Limitations

- Integration covers the linear-argument subset only; `x·exp(x)`,
  polynomials times transcendentals, and trigonometric products raise
  `NotImplementedError`.
- `pow(x, n)` for constant non-integer base/exponent simplifies only when the
  exponent has denominator 1.
- The expression `1/x^2` prints as `x^-2` (negative exponents are kept as
  powers, not written as fractions).
- Full symbolic evaluation (e.g. trigonometric identities like
  `sin^2 + cos^2 = 1`) is out of scope.
