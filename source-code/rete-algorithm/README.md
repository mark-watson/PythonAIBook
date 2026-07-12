# Rete Algorithm Engine for Python

A lightweight, embeddable, and idiomatic Python implementation of the Rete pattern matching algorithm. Designed for building efficient, forward-chaining expert systems directly in Python, this implementation is heavily inspired by OPS5.

---

## Features

- **Incremental Pattern Matching**: Leverages a compiled Rete network (Alpha network for type-dispatch and constant tests, Beta network for multi-condition variable joins) to evaluate facts incrementally.
- **Idiomatic Python Syntax**:
  - Define facts as frozen, type-safe Python `@dataclass` objects.
  - Define rules using standard Python functions decorated with `@engine.rule`.
  - Automatic dispatch of bound variables from the LHS pattern directly to the RHS function arguments.
- **Negated Condition Elements (NCEs)**: Express negative constraints (e.g. "if condition X is true and no fact of type Y exists") using the natural bitwise negation operator (`~Pat(...)`).
- **Deferred Mutations**: All assertions, retractions, and modifications triggered from rule actions are queued and applied atomically after each action completes, matching OPS5 semantics and preventing re-entrant corruption.
- **Flexible Conflict Resolution**: Supports multiple resolution strategies, including:
  - `lex` (Lexicographical specificity and recency, default)
  - `mea` (Focusing on the recency of the first condition)
  - `priority-only` (Salience/priority priority)
- **Refraction**: Prevents infinite loops by ensuring that any given rule instantiation (set of matching facts) fires at most once.
- **Zero External Runtime Dependencies**: Built entirely with Python's standard library.

---

## Installation

This project is configured to use [uv](https://github.com/astral-sh/uv) for dependency management and execution.

Ensure you have `uv` installed, then synchronize the environment:

```bash
uv sync
```

---

## Quick Start

Create a patient diagnosis rule system as shown in the example below:

```python
from dataclasses import dataclass
from rete import ReteEngine, Fact, Pat, Var, Gt, Eq, Contains

# 1. Define Facts (as frozen dataclasses inheriting from Fact)
@dataclass(frozen=True)
class Patient(Fact):
    name: str
    temperature: float
    symptoms: tuple[str, ...] = ()

@dataclass(frozen=True)
class Diagnosis(Fact):
    patient: str
    condition: str

# 2. Instantiate the Engine
engine = ReteEngine(strategy="lex")

# 3. Define Rules
@engine.rule(
    Pat(Patient, name=Var("n"), temperature=Gt(38.5), symptoms=Contains("cough")),
    ~Pat(Diagnosis, patient=Var("n"), condition=Eq("flu")),
    salience=10
)
def diagnose_flu(ctx, n):
    # ctx.assert_fact queues the assertion to run after the action completes
    ctx.assert_fact(Diagnosis(patient=n, condition="flu"))
    ctx.print(f"Diagnosed {n} with flu")

# 4. Assert Facts and Run
engine.assert_fact(Patient(name="Alice", temperature=39.1, symptoms=("cough", "chills")))
fired_rules = engine.run()
print(f"Rules fired: {fired_rules}")
```

To run the default medical diagnosis example:

```bash
uv run example_medical.py
```

### Complete Examples Suite

The engine includes several self-contained examples demonstrating various design patterns in Rete-based expert systems:

1. **Smart Home Automation & Sensor Fusion** ([example_smart_home.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/rete-algorithm/example_smart_home.py)):
   - *Concepts*: Multi-sensor conditions (motion, light, temp), reactive device rules, and global state controls (e.g., away mode auto-shutoff).
   - *Run*: `uv run example_smart_home.py`

2. **Network Intrusion Detection & Threat Alerting** ([example_network_security.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/rete-algorithm/example_network_security.py)):
   - *Concepts*: Ephemeral network events consumption, stateful counters, brute-force / port-scan detection, alerts, and connection blocking.
   - *Run*: `uv run example_network_security.py`

3. **E-Commerce Pricing & Discount Engine** ([example_ecom_pricing.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/rete-algorithm/example_ecom_pricing.py)):
   - *Concepts*: Multi-condition joins (customer loyalty tier + categories), accumulator loops for subtotals/discounts, and Negated Condition Elements (NCEs) as a completion barrier for invoice creation.
   - *Run*: `uv run example_ecom_pricing.py`

4. **Financial Portfolio Rebalancing & Alerting** ([example_portfolio_rebalance.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/rete-algorithm/example_portfolio_rebalance.py)):
   - *Concepts*: Aggregate asset holdings calculation, drift tolerance threshold comparisons, and rebalance trade action recommendations.
   - *Run*: `uv run example_portfolio_rebalance.py`

5. **Hospital Patient Triage & Resource Allocation** ([example_hospital_triage.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/rete-algorithm/example_hospital_triage.py)):
   - *Concepts*: Multi-resource assignments (Patient + Bed + Staff), salience-based case prioritization (critical first), and state-release workflow loops.
   - *Run*: `uv run example_hospital_triage.py`

---


## Core API Reference

### Facts & Working Memory

All facts must inherit from the `Fact` class and should be defined as frozen dataclasses to ensure hashability (which the Rete network relies on for tracking and indexing matches):

```python
from dataclasses import dataclass
from rete import Fact

@dataclass(frozen=True)
class SensorReading(Fact):
    sensor_id: str
    value: float
```

Assert, retract, or modify facts using the engine:

- `engine.assert_fact(fact)`: Inserts a new fact and propagates matches.
- `engine.retract(wme)`: Retracts an existing fact representation (`WME`) from memory.
- `engine.modify(wme, **changes)`: Conveniently retracts the old fact and asserts a modified copy with updated fields.
- `engine.facts(fact_type=None)`: Iterates over all active facts in working memory (with optional filtering by type).

### LHS Patterns & Conditions

Rules specify condition patterns inside the `@engine.rule` decorator. The patterns are built using:
- `Pat(FactClass, field_name=Operator)`: Matches a fact of the specified type matching the field constraints.
- `~Pat(FactClass, ...)`: Represents a negated pattern. The rule matches only if no fact fits this pattern in working memory.

Available field operators include:
- `Var("var_name")`: Binds the field value to a variable, checking consistency across conditions if the same variable is used multiple times.
- `Eq(value)` / Constant matching: Checks if a field matches a specific value. Regular values are automatically wrapped in `Eq` (e.g., `symptoms="cough"` behaves as `symptoms=Eq("cough")`).
- `Gt(value)` / `Lt(value)` / `Ge(value)` / `Le(value)`: Relational comparisons.
- `Contains(item)`: Checks if a collection field contains a specific item.
- `In(collection)`: Checks if a field's value exists in the specified collection.
- `Test(predicate)`: Evaluates a custom predicate function `lambda val: ...`.

### RHS Actions and the `RuleContext`

The decorated function is the action (RHS) of the rule. The first parameter is always `ctx`, a `RuleContext` handle that manages mutations safely:
- `ctx.assert_fact(fact)`: Schedules a new fact to be added.
- `ctx.retract(wme)`: Schedules an existing fact to be removed.
- `ctx.modify(wme, **changes)`: Schedules a retract + assert update on a fact.
- `ctx.halt()`: Stops execution of the current `recognize-act` cycle immediately.
- `ctx.print(msg)`: Appends messages to the engine's execution traces.

Additional parameters of the decorated function are automatically mapped to variables bound on the LHS:

```python
@engine.rule(Pat(Patient, name=Var("p_name"), temperature=Gt(38.0)))
def alert_fever(ctx, p_name):
    # p_name is automatically filled with the string bound to Var("p_name")
    print(f"Alert: {p_name} has a fever!")
```

---

## Development & Testing

Run unit tests via `pytest`:

```bash
uv run pytest
```

## Development workflow

Uses [`uv`](https://docs.astral.sh/uv/) for dependency management and [`just`](https://just.systems/) as the task runner. Install both, then:

```bash
uv sync
just check       # fmt-check + lint + typecheck + test
just fmt         # ruff format
just lint        # ruff --fix
just typecheck   # pyrefly (strict)
just test        # pytest with testmon (fast)
just test-all    # full parallel pytest run
```

Under Claude Code, `.claude/hooks/py-check.sh` runs after every edit (format + lint + per-file typecheck) and `.claude/hooks/py-stop.sh` runs the full gate before the turn ends. See `CLAUDE.md` for the workflow contract.
