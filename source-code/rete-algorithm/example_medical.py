#!/usr/bin/env python3
"""Medical diagnosis expert system — worked example from Design.md.

Demonstrates:
  - Frozen dataclass facts (Patient, Diagnosis, Treatment)
  - Variable binding and inter-condition joins
  - Negated condition elements (NCEs)
  - Forward chaining (diagnose → treat)
  - Salience-based ordering
"""

from dataclasses import dataclass
from typing import Any

from rete import Contains, Eq, Fact, Gt, Pat, ReteEngine, RuleContext, Var


# ── Fact types ─────────────────────────────────────────────────────


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


# ── Engine & rules ─────────────────────────────────────────────────

engine = ReteEngine(strategy="lex")


@engine.rule(
    Pat(Patient, name=Var("n"), temperature=Gt(38.5), symptoms=Contains("cough")),
    ~Pat(Diagnosis, patient=Var("n"), condition=Eq("flu")),
    salience=10,
)
def diagnose_flu(ctx: RuleContext, n: Any) -> None:
    """High-fever patient with cough and no existing flu diagnosis → diagnose flu."""
    ctx.assert_fact(Diagnosis(patient=n, condition="flu"))
    ctx.print(f"  → Diagnosed {n} with flu")


@engine.rule(
    Pat(Diagnosis, patient=Var("n"), condition=Eq("flu")),
    ~Pat(Treatment, patient=Var("n")),
)
def treat_flu(ctx: RuleContext, n: Any) -> None:
    """Flu diagnosis with no treatment yet → prescribe oseltamivir."""
    ctx.assert_fact(Treatment(patient=n, action="prescribe oseltamivir"))
    ctx.print(f"  → Treatment for {n}: oseltamivir")


@engine.rule(
    Pat(Patient, name=Var("n"), temperature=Gt(38.0)),
    ~Pat(Diagnosis, patient=Var("n")),
    salience=5,
)
def diagnose_fever(ctx: RuleContext, n: Any) -> None:
    """Moderate fever with no diagnosis → generic fever diagnosis."""
    ctx.assert_fact(Diagnosis(patient=n, condition="fever"))
    ctx.print(f"  → Diagnosed {n} with fever")


# ── Run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Medical Diagnosis Expert System ===\n")

    patients = [
        Patient(name="Alice", temperature=39.2, symptoms=("cough", "fatigue")),
        Patient(name="Bob", temperature=37.0, symptoms=("headache",)),
        Patient(name="Carol", temperature=38.3, symptoms=("chills",)),
    ]

    for p in patients:
        print(f"Asserting: {p}")
        engine.assert_fact(p)

    print()
    fired = engine.run()
    print(f"\nRules fired: {fired}\n")

    print("── Working Memory ──")
    for f in engine.facts():
        print(f"  {f}")
