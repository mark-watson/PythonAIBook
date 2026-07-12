"""Tests for the Rete engine — medical diagnosis example from Design.md."""

from dataclasses import dataclass
from typing import Any

from rete import Contains, Eq, Fact, Gt, Lt, Pat, ReteEngine, RuleContext, Var


# ---------------------------------------------------------------------------
# Fact types
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test: basic single-rule firing
# ---------------------------------------------------------------------------


def test_single_rule_fires():
    engine = ReteEngine()

    @engine.rule(Pat(Patient, name=Var("n"), temperature=Gt(38.5)), salience=1)
    def fever(ctx: RuleContext, n: Any) -> None:
        ctx.assert_fact(Diagnosis(patient=n, condition="fever"))

    engine.assert_fact(Patient(name="Alice", temperature=39.2))
    fired = engine.run()

    assert fired == 1
    diagnoses = list(engine.facts(Diagnosis))
    assert len(diagnoses) == 1
    assert diagnoses[0].patient == "Alice"
    assert diagnoses[0].condition == "fever"


# ---------------------------------------------------------------------------
# Test: rule does NOT fire when conditions are not met
# ---------------------------------------------------------------------------


def test_rule_does_not_fire_when_unmatched():
    engine = ReteEngine()

    @engine.rule(Pat(Patient, temperature=Gt(38.5)))
    def fever(ctx: RuleContext) -> None:
        ctx.assert_fact(Diagnosis(patient="?", condition="fever"))

    engine.assert_fact(Patient(name="Bob", temperature=37.0))
    fired = engine.run()

    assert fired == 0
    assert list(engine.facts(Diagnosis)) == []


# ---------------------------------------------------------------------------
# Test: multi-condition join
# ---------------------------------------------------------------------------


def test_join_on_variable():
    engine = ReteEngine()

    @engine.rule(
        Pat(Patient, name=Var("n"), temperature=Gt(38.5), symptoms=Contains("cough")),
        Pat(Diagnosis, patient=Var("n"), condition=Eq("fever")),
    )
    def fever_with_cough(ctx: RuleContext, n: Any) -> None:
        ctx.assert_fact(Treatment(patient=n, action="rest and fluids"))

    engine.assert_fact(
        Patient(name="Alice", temperature=39.2, symptoms=("cough", "fatigue"))
    )
    engine.assert_fact(Diagnosis(patient="Alice", condition="fever"))
    fired = engine.run()

    assert fired == 1
    treatments = list(engine.facts(Treatment))
    assert len(treatments) == 1
    assert treatments[0].patient == "Alice"


def test_join_no_match_different_variable():
    """Join should fail when variable values don't match."""
    engine = ReteEngine()

    @engine.rule(
        Pat(Patient, name=Var("n"), temperature=Gt(38.5)),
        Pat(Diagnosis, patient=Var("n")),
    )
    def combined(ctx: RuleContext, n: Any) -> None:
        ctx.assert_fact(Treatment(patient=n, action="treat"))

    engine.assert_fact(Patient(name="Alice", temperature=39.0))
    engine.assert_fact(Diagnosis(patient="Bob", condition="fever"))
    fired = engine.run()

    assert fired == 0  # "Alice" != "Bob" — join fails


# ---------------------------------------------------------------------------
# Test: negated condition element
# ---------------------------------------------------------------------------


def test_negated_condition():
    engine = ReteEngine()

    @engine.rule(
        Pat(Patient, name=Var("n"), temperature=Gt(38.5)),
        ~Pat(Diagnosis, patient=Var("n")),
        salience=5,
    )
    def needs_diagnosis(ctx: RuleContext, n: Any) -> None:
        ctx.assert_fact(Diagnosis(patient=n, condition="unknown"))

    engine.assert_fact(Patient(name="Alice", temperature=39.0))
    fired = engine.run()

    assert fired == 1
    d = list(engine.facts(Diagnosis))
    assert len(d) == 1
    assert d[0].patient == "Alice"


def test_negated_blocks_when_present():
    engine = ReteEngine()

    @engine.rule(
        Pat(Patient, name=Var("n")),
        ~Pat(Diagnosis, patient=Var("n")),
    )
    def needs_diag(ctx: RuleContext, n: Any) -> None:
        ctx.assert_fact(Diagnosis(patient=n, condition="pending"))

    # Assert diagnosis BEFORE patient — negation should block
    engine.assert_fact(Diagnosis(patient="Alice", condition="flu"))
    engine.assert_fact(Patient(name="Alice", temperature=37.0))
    fired = engine.run()

    assert fired == 0


# ---------------------------------------------------------------------------
# Test: chained rules (forward chaining)
# ---------------------------------------------------------------------------


def test_chained_rules():
    """Rules fire in sequence: diagnose_flu -> treat_flu."""
    engine = ReteEngine()

    @engine.rule(
        Pat(Patient, name=Var("n"), temperature=Gt(38.5), symptoms=Contains("cough")),
        ~Pat(Diagnosis, patient=Var("n"), condition=Eq("flu")),
        salience=10,
    )
    def diagnose_flu(ctx: RuleContext, n: Any) -> None:
        ctx.assert_fact(Diagnosis(patient=n, condition="flu"))

    @engine.rule(
        Pat(Diagnosis, patient=Var("n"), condition=Eq("flu")),
        ~Pat(Treatment, patient=Var("n")),
    )
    def treat_flu(ctx: RuleContext, n: Any) -> None:
        ctx.assert_fact(Treatment(patient=n, action="prescribe oseltamivir"))

    engine.assert_fact(
        Patient(name="Alice", temperature=39.2, symptoms=("cough", "fatigue"))
    )
    engine.assert_fact(Patient(name="Bob", temperature=37.0, symptoms=("headache",)))
    fired = engine.run()

    assert fired == 2
    diagnoses = list(engine.facts(Diagnosis))
    assert any(d.patient == "Alice" and d.condition == "flu" for d in diagnoses)

    treatments = list(engine.facts(Treatment))
    assert any(
        t.patient == "Alice" and t.action == "prescribe oseltamivir" for t in treatments
    )

    # Bob should NOT have been diagnosed or treated
    assert not any(d.patient == "Bob" for d in diagnoses)
    assert not any(t.patient == "Bob" for t in treatments)


# ---------------------------------------------------------------------------
# Test: refraction
# ---------------------------------------------------------------------------


def test_refraction_prevents_refire():
    engine = ReteEngine()
    fire_count = 0

    @engine.rule(Pat(Patient, name=Var("n")))
    def count_rule(ctx: RuleContext, n: Any) -> None:
        nonlocal fire_count
        fire_count += 1

    engine.assert_fact(Patient(name="Alice", temperature=37.0))
    engine.run()
    engine.run()  # second run should not re-fire

    assert fire_count == 1


# ---------------------------------------------------------------------------
# Test: salience ordering
# ---------------------------------------------------------------------------


def test_salience_ordering():
    engine = ReteEngine()
    order: list[str] = []

    @engine.rule(Pat(Patient, name=Var("n")), salience=1)
    def low_priority(ctx: RuleContext, n: Any) -> None:
        order.append("low")

    @engine.rule(Pat(Patient, name=Var("n")), salience=10)
    def high_priority(ctx: RuleContext, n: Any) -> None:
        order.append("high")

    engine.assert_fact(Patient(name="Alice", temperature=37.0))
    engine.run()

    # Both fire, but high salience first
    assert order[0] == "high"


# ---------------------------------------------------------------------------
# Test: halt
# ---------------------------------------------------------------------------


def test_halt_stops_cycle():
    engine = ReteEngine()

    @engine.rule(Pat(Patient, name=Var("n")), salience=10)
    def stop(ctx: RuleContext, n: Any) -> None:
        ctx.halt()

    @engine.rule(Pat(Patient, name=Var("n")), salience=1)
    def should_not_fire(ctx: RuleContext, n: Any) -> None:
        raise AssertionError("Should not have fired!")

    engine.assert_fact(Patient(name="Alice", temperature=37.0))
    fired = engine.run()

    assert fired == 1


# ---------------------------------------------------------------------------
# Test: max_cycles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Counter(Fact):
    value: int


def test_max_cycles():
    engine = ReteEngine()

    @engine.rule(Pat(Counter, value=Var("v")))
    def inc(ctx: RuleContext, v: Any) -> None:
        # This would loop forever without max_cycles
        pass

    engine.assert_fact(Counter(value=0))
    fired = engine.run(max_cycles=3)

    assert fired <= 3


# ---------------------------------------------------------------------------
# Test: pattern operators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Item(Fact):
    name: str
    price: float
    tags: tuple[str, ...] = ()


def test_lt_operator():
    engine = ReteEngine()
    results: list[str] = []

    @engine.rule(Pat(Item, name=Var("n"), price=Lt(10.0)))
    def cheap(ctx: RuleContext, n: Any) -> None:
        results.append(n)

    engine.assert_fact(Item(name="pen", price=5.0))
    engine.assert_fact(Item(name="laptop", price=999.0))
    engine.run()

    assert results == ["pen"]


def test_contains_operator():
    engine = ReteEngine()
    results: list[str] = []

    @engine.rule(Pat(Item, name=Var("n"), tags=Contains("sale")))
    def on_sale(ctx: RuleContext, n: Any) -> None:
        results.append(n)

    engine.assert_fact(Item(name="shirt", price=20.0, tags=("sale", "new")))
    engine.assert_fact(Item(name="pants", price=40.0, tags=("new",)))
    engine.run()

    assert results == ["shirt"]


# ---------------------------------------------------------------------------
# Test: retraction removes instantiations
# ---------------------------------------------------------------------------


def test_retraction():
    engine = ReteEngine()

    @engine.rule(Pat(Patient, name=Var("n")))
    def greet(ctx: RuleContext, n: Any) -> None:
        ctx.assert_fact(Diagnosis(patient=n, condition="greeted"))

    wme = engine.assert_fact(Patient(name="Alice", temperature=37.0))
    # Before running, retract the fact
    engine.retract(wme)
    fired = engine.run()

    assert fired == 0
