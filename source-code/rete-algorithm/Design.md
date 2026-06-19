# Design: Rete Algorithm Engine for Python

> A concise, idiomatic Python implementation of the Rete algorithm for embedded expert systems, inspired by OPS5.

---

## 1. Goals & Non-Goals

### Goals

- **Faithful Rete implementation** — alpha network (intra-condition tests), beta network (inter-condition joins), conflict resolution, and match-remove cycle.
- **Idiomatic Python API** — rules defined as decorated functions; working memory elements (WMEs) are plain Python objects (dataclasses / named tuples / dicts).
- **Embeddable** — zero external dependencies; usable as a library inside any Python application.
- **OPS5-familiar semantics** — LHS patterns with variable binding, RHS actions that modify working memory, and a recognize-act cycle.
- **Incremental matching** — only propagate deltas through the network (the core Rete advantage).

### Non-Goals

- Full CLIPS/Jess feature parity (defmodules, salience groups beyond basic priority).
- Rete-II optimizations (right-unlinking, sub-network sharing) — can be added later.
- Thread-safety in the first iteration.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    User Code                         │
│  engine = ReteEngine()                               │
│  engine.add_rule(...)   engine.assert_fact(...)      │
│  engine.run()                                        │
└────────────┬─────────────────────────┬───────────────┘
             │                         │
             ▼                         ▼
┌────────────────────┐    ┌────────────────────────────┐
│   Rule Compiler    │    │     Working Memory (WM)    │
│  (LHS → network)   │    │  fact store + α-indexing   │
└────────┬───────────┘    └────────────┬───────────────┘
         │                             │
         ▼                             ▼
┌──────────────────────────────────────────────────────┐
│                   Rete Network                       │
│                                                      │
│  ┌─────────┐    ┌──────────┐    ┌──────────────────┐ │
│  │  Root    │───▶│  Alpha   │───▶│  Alpha Memories  │ │
│  │  Node    │    │  Tests   │    │  (per-condition)  │ │
│  └─────────┘    └──────────┘    └────────┬─────────┘ │
│                                          │           │
│                                          ▼           │
│                                 ┌────────────────┐   │
│                                 │   Beta Nodes   │   │
│                                 │  (join nodes)  │   │
│                                 └────────┬───────┘   │
│                                          │           │
│                                          ▼           │
│                                 ┌────────────────┐   │
│                                 │  Production    │   │
│                                 │  (p-node)      │   │
│                                 └────────────────┘   │
└──────────────────────────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Conflict Set &  │
              │ Resolution      │
              └─────────────────┘
```

---

## 3. Core Data Structures

### 3.1 Working Memory Element (WME)

WMEs are the facts the engine reasons over. They are **immutable** after assertion (modify = retract + assert).

```python
from dataclasses import dataclass, field
from typing import Any

# Users define their own fact types as frozen dataclasses:
@dataclass(frozen=True)
class Fact:
    """Base marker. User facts inherit from this."""
    pass

# Example user facts:
@dataclass(frozen=True)
class Patient:
    name: str
    temperature: float
    symptoms: tuple[str, ...] = ()

@dataclass(frozen=True)
class Diagnosis:
    patient: str
    condition: str
    confidence: float = 1.0
```

**Design decision**: Frozen dataclasses give us hashability (needed for alpha memory sets and token hashing) and field-level attribute access. Users can also register plain dicts or named tuples via an adapter.

### 3.2 Token

A token represents a partial match as it flows through the beta network. It is a chain of bound WMEs plus a variable-binding environment.

```python
@dataclass(frozen=True)
class Token:
    """A partial match flowing through the beta network."""
    parent: Token | None          # previous token in the chain
    wme: WME                      # the WME matched at this level
    bindings: dict[str, Any]      # variable name → bound value

    def extend(self, wme: WME, new_bindings: dict[str, Any]) -> Token:
        merged = {**self.bindings, **new_bindings}
        return Token(parent=self, wme=wme, bindings=merged)

    def wmes(self) -> list[WME]:
        """Collect all WMEs in this match chain."""
        result, tok = [], self
        while tok is not None:
            result.append(tok.wme)
            tok = tok.parent
        result.reverse()
        return result
```

### 3.3 Pattern (Condition Element)

Each condition in a rule's LHS compiles into a `Pattern`:

```python
@dataclass
class Pattern:
    fact_type: type                          # e.g. Patient
    tests: list[tuple[str, Callable]]        # intra-element tests (alpha)
    bindings: dict[str, str]                 # field_name → variable_name
    negated: bool = False                    # negative condition element
```

---

## 4. Rete Network Nodes

### 4.1 Alpha Network

The alpha network filters individual WMEs against intra-element conditions.

| Node | Purpose |
|------|---------|
| **RootNode** | Entry point; dispatches WMEs by `type(wme)` to type-specific alpha branches. Uses a `dict[type, list[AlphaNode]]` for O(1) type dispatch. |
| **AlphaTestNode** | Applies a single constant/predicate test on one field (e.g., `temperature > 38.0`). Chains to the next test or to an AlphaMemory. |
| **AlphaMemory** | Stores WMEs that passed all alpha tests for one condition. Feeds into beta join nodes. Indexed as a `set[WME]` for O(1) add/remove. |

**Optimization**: Hash-based alpha indexing. For equality tests (`field == constant`), use a `dict[value, set[WME]]` inside the AlphaMemory to avoid scanning.

### 4.2 Beta Network

The beta network performs inter-element joins (variable consistency tests).

| Node | Purpose |
|------|---------|
| **BetaMemory** | Stores tokens (partial matches). Fed by the left input of a join. |
| **JoinNode** | Two inputs: *left* (BetaMemory / previous JoinNode) and *right* (AlphaMemory). Performs join tests comparing bound variables in the token against fields of the new WME. |
| **NegativeNode** | Like JoinNode but implements negated conditions (NCEs). A token passes only if *no* WME in the alpha memory satisfies the join. |
| **ProductionNode (P-Node)** | Terminal node. When a token arrives, a complete match (instantiation) is added to the conflict set. |

```python
class JoinNode:
    def __init__(self, alpha_memory: AlphaMemory,
                 tests: list[JoinTest],
                 children: list[BetaNode]):
        self.alpha_memory = alpha_memory
        self.tests = tests            # (token_field, wme_field) pairs
        self.children = children
        self.beta_memory: list[Token] = []

    def left_activate(self, token: Token):
        """A new token arrived from the left (beta side)."""
        for wme in self.alpha_memory.wmes:
            bindings = self._try_join(token, wme)
            if bindings is not None:
                new_token = token.extend(wme, bindings)
                for child in self.children:
                    child.left_activate(new_token)

    def right_activate(self, wme: WME):
        """A new WME arrived from the right (alpha side)."""
        for token in self.beta_memory:
            bindings = self._try_join(token, wme)
            if bindings is not None:
                new_token = token.extend(wme, bindings)
                for child in self.children:
                    child.left_activate(new_token)
```

### 4.3 Node Sharing

Multiple rules with identical prefixes share alpha and beta nodes. The compiler detects shared condition prefixes during rule compilation and reuses existing network branches.

---

## 5. Rule Definition API

### 5.1 Decorator-Based DSL

Rules are defined as decorated Python functions. The decorator captures the LHS patterns; the function body is the RHS action.

```python
engine = ReteEngine()

@engine.rule(
    Patient(name=Var("n"), temperature=Gt(38.0), symptoms=Contains("cough")),
    ~Diagnosis(patient=Var("n")),      # negated: no diagnosis yet
    salience=10,
)
def fever_with_cough(ctx: RuleContext, n: str):
    """If a patient has fever + cough and no diagnosis, diagnose flu."""
    ctx.assert_fact(Diagnosis(patient=n, condition="flu", confidence=0.7))
    ctx.print(f"Diagnosed {n} with flu")
```

### 5.2 Pattern Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `Var("x")` | Bind field to variable `x` | `name=Var("n")` |
| `Eq(val)` | Field equals constant | `status=Eq("active")` |
| `Gt(val)`, `Lt(val)`, `Ge(val)`, `Le(val)` | Numeric comparisons | `temperature=Gt(38.0)` |
| `In(...)` | Field value in set | `color=In("red","yellow")` |
| `Contains(val)` | Collection field contains value | `symptoms=Contains("cough")` |
| `Match(regex)` | Regex match on string field | `name=Match(r"^J.*")` |
| `Test(fn)` | Arbitrary predicate `fn(value) → bool` | `age=Test(lambda a: 18 <= a < 65)` |
| `~Pattern(...)` | Negated condition element | `~Diagnosis(patient=Var("n"))` |

### 5.3 Multi-Condition Joins

Variables with the same name across patterns create implicit join constraints:

```python
@engine.rule(
    Order(id=Var("oid"), customer=Var("cid"), total=Var("t")),
    Customer(id=Var("cid"), tier=Eq("gold")),   # join on cid
    Test(lambda t: t > 1000),                     # inter-element test on t
)
def gold_customer_large_order(ctx, oid, cid, t):
    ctx.assert_fact(Alert(order=oid, message=f"Gold customer {cid} large order: ${t}"))
```

---

## 6. Working Memory API

### 6.1 Primitives

```python
class WorkingMemory:
    def assert_fact(self, fact: Fact) -> WME:
        """Add a fact. Returns the WME handle. Triggers alpha activation."""

    def retract(self, wme: WME) -> None:
        """Remove a WME. Propagates removal through the network."""

    def modify(self, wme: WME, **changes) -> WME:
        """Retract old WME, assert a new one with updated fields.
        Sugar for: retract(wme); assert_fact(replace(wme, **changes))
        """

    def facts(self, fact_type: type | None = None) -> Iterator[Fact]:
        """Iterate over current facts, optionally filtered by type."""

    def clear(self) -> None:
        """Remove all facts and reset the network state."""
```

### 6.2 RuleContext (RHS Handle)

The `RuleContext` passed to rule RHS functions provides scoped WM access:

```python
class RuleContext:
    def assert_fact(self, fact: Fact) -> WME: ...
    def retract(self, wme: WME) -> None: ...
    def modify(self, wme: WME, **changes) -> WME: ...
    def halt(self) -> None:
        """Stop the recognize-act cycle."""
    def print(self, msg: str) -> None:
        """Output to the engine's trace log."""
```

All RHS mutations are **deferred** until the current rule finishes executing, then applied as a batch before the next cycle. This avoids re-entrant network updates and matches OPS5 semantics.

---

## 7. Recognize-Act Cycle

```
          ┌─────────────────────────┐
          │  1. Match               │
          │  (network is current)   │
          └───────────┬─────────────┘
                      │
                      ▼
          ┌─────────────────────────┐
          │  2. Conflict Resolution │
          │  Select one production  │
          │  instantiation to fire  │
          └───────────┬─────────────┘
                      │
                      ▼
          ┌─────────────────────────┐
          │  3. Act                 │
          │  Execute RHS action     │
          │  (deferred WM updates)  │
          └───────────┬─────────────┘
                      │
                      ▼
          ┌─────────────────────────┐
          │  4. Apply WM deltas     │
          │  Propagate through Rete │
          └───────────┬─────────────┘
                      │
                      ▼
               conflict set empty
               or halt() called?
              ╱                ╲
           Yes                  No ──▶ goto 2
            │
            ▼
          Done
```

### 7.1 Conflict Resolution Strategy

Default strategy (configurable):

1. **Refraction** — an instantiation fires at most once (unless its supporting WMEs change). Mandatory.
2. **Salience** — higher salience fires first.
3. **Recency** — prefer instantiations involving the most recently asserted WMEs (OPS5 LEX/MEA style). Default: LEX (most recent WME across all conditions).
4. **Specificity** — prefer rules with more conditions.

```python
engine = ReteEngine(strategy="lex")   # or "mea", "priority-only", or custom
```

Users can supply a custom `Callable[[list[Instantiation]], Instantiation]` for full control.

---

## 8. Engine API

```python
class ReteEngine:
    def __init__(self, strategy: str = "lex"):
        self.wm = WorkingMemory()
        self.network = ReteNetwork()
        self.strategy = resolve_strategy(strategy)

    # --- Rule registration ---
    def rule(self, *patterns, salience=0, name=None):
        """Decorator to register a rule."""

    def add_rule(self, name: str, patterns: list[Pattern],
                 action: Callable, salience: int = 0):
        """Programmatic rule registration."""

    # --- Working memory (delegates to self.wm) ---
    def assert_fact(self, fact: Fact) -> WME: ...
    def retract(self, wme: WME) -> None: ...
    def modify(self, wme: WME, **changes) -> WME: ...

    # --- Execution ---
    def run(self, max_cycles: int | None = None) -> int:
        """Run the recognize-act cycle. Returns number of rules fired."""

    def step(self) -> bool:
        """Execute one cycle. Returns True if a rule fired."""

    # --- Introspection ---
    def facts(self, fact_type=None) -> Iterator[Fact]: ...
    def conflict_set(self) -> list[Instantiation]: ...
    def trace(self, enabled: bool = True) -> None:
        """Enable/disable firing trace to stderr."""
```

---

## 9. Module / Package Layout

```
rete/
├── __init__.py          # Public API re-exports
├── engine.py            # ReteEngine, recognize-act cycle
├── facts.py             # Fact base class, WME wrapper
├── patterns.py          # Pattern, Var, Gt, Lt, Eq, etc.
├── network.py           # ReteNetwork builder, node sharing
├── alpha.py             # RootNode, AlphaTestNode, AlphaMemory
├── beta.py              # BetaMemory, JoinNode, NegativeNode, ProductionNode
├── tokens.py            # Token data structure
├── conflict.py          # Conflict resolution strategies
├── context.py           # RuleContext for RHS actions
└── tests/
    ├── test_alpha.py
    ├── test_beta.py
    ├── test_engine.py
    └── test_examples.py
```

---

## 10. Worked Example — Medical Diagnosis

```python
from dataclasses import dataclass
from rete import ReteEngine, Fact, Var, Gt, Contains, Eq

engine = ReteEngine()

# --- Fact types ---
@dataclass(frozen=True)
class Patient(Fact):
    name: str
    temperature: float
    symptoms: tuple[str, ...] = ()

@dataclass(frozen=True)
class Diagnosis(Fact):
    patient: str
    condition: str

@dataclass(frozen=True)
class Treatment(Fact):
    patient: str
    action: str

# --- Rules ---
@engine.rule(
    Patient(name=Var("n"), temperature=Gt(38.5), symptoms=Contains("cough")),
    ~Diagnosis(patient=Var("n"), condition=Eq("flu")),
    salience=10,
)
def diagnose_flu(ctx, n):
    ctx.assert_fact(Diagnosis(patient=n, condition="flu"))

@engine.rule(
    Diagnosis(patient=Var("n"), condition=Eq("flu")),
    ~Treatment(patient=Var("n")),
)
def treat_flu(ctx, n):
    ctx.assert_fact(Treatment(patient=n, action="prescribe oseltamivir"))
    ctx.print(f"Treatment plan for {n}: oseltamivir")

# --- Run ---
engine.assert_fact(Patient(name="Alice", temperature=39.2, symptoms=("cough", "fatigue")))
engine.assert_fact(Patient(name="Bob", temperature=37.0, symptoms=("headache",)))

fired = engine.run()
print(f"Rules fired: {fired}")
# Output:
#   Treatment plan for Alice: oseltamivir
#   Rules fired: 2

# Introspect
for f in engine.facts(Diagnosis):
    print(f)  # Diagnosis(patient='Alice', condition='flu')
```

---

## 11. Performance Considerations

| Concern | Approach |
|---------|----------|
| Alpha hashing | Equality tests use `dict[value, set[WME]]` inside AlphaMemory — avoids linear scans |
| Beta indexing | JoinNodes index the right (alpha) input on join keys for O(1) lookup during left-activation |
| Token GC | Tokens are immutable chains; retraction removes from beta memories and uses parent-pointer traversal to invalidate downstream tokens |
| Memory | Frozen dataclass facts are hashable and deduplicated in sets; token chains share structure |
| Large WM | For >10k facts, AlphaMemory uses hash partitioning on the most selective test field |

---

## 12. Extension Points (Future)

- **Truth Maintenance (TMS)** — auto-retract derived facts when justifications are removed.
- **Temporal reasoning** — fact validity windows, `cron`-style periodic re-evaluation.
- **Rete-II right-unlinking** — lazily disconnect alpha memories with no pending tokens.
- **Rule groups / modules** — OPS5-style context switching with `focus` / `pop-focus`.
- **Persistence** — serialize/deserialize WM and network state.
- **Explanation** — trace which rules and facts led to a conclusion (audit trail).

---

## 13. Dependencies & Tooling

- **Python ≥ 3.12** (for modern dataclass and type hint features).
- **Zero runtime dependencies**.
- **Dev**: `pytest`, `mypy`, `ruff`.
- **Package manager**: `uv` (per project convention).

---

## 14. Summary

This design delivers a **Rete-based production system** with:

1. **Declarative rules** via Python decorators — no external DSL files.
2. **Pythonic facts** as frozen dataclasses — full IDE support, type checking.
3. **Efficient incremental matching** — the Rete algorithm's core value proposition.
4. **Clean embedding** — instantiate `ReteEngine()`, add rules and facts, call `run()`.

The API is deliberately small. An expert system can be defined in a single Python file with no boilerplate beyond fact and rule definitions.
