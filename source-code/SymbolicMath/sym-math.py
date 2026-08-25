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
