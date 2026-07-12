#!/usr/bin/env python3
"""Hospital Patient Triage & Resource Allocation Example.

Demonstrates:
  - Multi-resource matching (Patient + Bed + Staff).
  - Priority-based scheduling using salience (critical patients assigned first).
  - Sequential workflow progression (waiting -> assigned -> treated).
  - State-release rules (freeing beds and staff dynamically to handle next patient).
"""

from dataclasses import dataclass
from typing import Any

from rete import Eq, Fact, Gt, Lt, Pat, ReteEngine, RuleContext, Var


# ── Fact types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Patient(Fact):
    id: str
    name: str
    severity: int  # 1 (lowest) to 5 (highest, critical)


@dataclass(frozen=True)
class PatientStatus(Fact):
    patient_id: str
    status: str  # "waiting", "assigned", "treated"


@dataclass(frozen=True)
class Bed(Fact):
    id: str
    occupied: bool = False


@dataclass(frozen=True)
class Staff(Fact):
    id: str
    name: str
    role: str  # "doctor", "nurse"
    busy: bool = False


@dataclass(frozen=True)
class Assignment(Fact):
    patient_id: str
    bed_id: str
    staff_id: str


# ── Engine & Rules ─────────────────────────────────────────────────

engine = ReteEngine(strategy="lex")


@engine.rule(
    Pat(Patient, id=Var("p_id"), severity=Gt(3)),
    Pat(PatientStatus, patient_id=Var("p_id"), status=Eq("waiting")),
    Pat(Bed, id=Var("b_id"), occupied=Eq(False)),
    Pat(Staff, id=Var("s_id"), role=Eq("doctor"), busy=Eq(False)),
    salience=10,  # Assign critical cases first
)
def triage_critical_patient(ctx: RuleContext, p_id: Any, b_id: Any, s_id: Any) -> None:
    """Critical patient (severity 4 or 5) assigned to empty bed and doctor."""
    ctx.assert_fact(Assignment(patient_id=p_id, bed_id=b_id, staff_id=s_id))

    # Mark status as assigned, bed occupied, doctor busy
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, PatientStatus):
            ctx.modify(wme, status="assigned")
        elif isinstance(wme.fact, Bed):
            ctx.modify(wme, occupied=True)
        elif isinstance(wme.fact, Staff):
            ctx.modify(wme, busy=True)

    ctx.print(
        f"  [Triage - CRITICAL] Assigned critical Patient {p_id} to Bed {b_id} and Doctor {s_id}"
    )


@engine.rule(
    Pat(Patient, id=Var("p_id"), severity=Lt(4)),
    Pat(PatientStatus, patient_id=Var("p_id"), status=Eq("waiting")),
    Pat(Bed, id=Var("b_id"), occupied=Eq(False)),
    Pat(Staff, id=Var("s_id"), role=Eq("nurse"), busy=Eq(False)),
    salience=5,  # Standard cases have normal priority
)
def triage_standard_patient(ctx: RuleContext, p_id: Any, b_id: Any, s_id: Any) -> None:
    """Standard patient (severity 1, 2, or 3) assigned to empty bed and nurse."""
    ctx.assert_fact(Assignment(patient_id=p_id, bed_id=b_id, staff_id=s_id))

    # Mark status as assigned, bed occupied, nurse busy
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, PatientStatus):
            ctx.modify(wme, status="assigned")
        elif isinstance(wme.fact, Bed):
            ctx.modify(wme, occupied=True)
        elif isinstance(wme.fact, Staff):
            ctx.modify(wme, busy=True)

    ctx.print(
        f"  [Triage - Standard] Assigned Patient {p_id} to Bed {b_id} and Nurse {s_id}"
    )


@engine.rule(
    Pat(Assignment, patient_id=Var("p_id"), bed_id=Var("b_id"), staff_id=Var("s_id")),
    Pat(PatientStatus, patient_id=Var("p_id"), status=Eq("assigned")),
    salience=1,  # Treatment happens after assignments have settled
)
def treat_patient(ctx: RuleContext, p_id: Any, b_id: Any, s_id: Any) -> None:
    """Treat patient, complete workflow, and free up resources (bed + staff)."""
    # Retract Assignment and modify PatientStatus to 'treated'
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, Assignment):
            ctx.retract(wme)
        elif isinstance(wme.fact, PatientStatus):
            ctx.modify(wme, status="treated")

    # Locate and release the Bed and Staff associated with this assignment
    # (Since Bed and Staff are not in the LHS of this rule, we queue updates inside working memory)
    for wme in list(ctx._engine.wm._facts.values()):
        if isinstance(wme.fact, Bed) and wme.fact.id == b_id:
            ctx.modify(wme, occupied=False)
        elif isinstance(wme.fact, Staff) and wme.fact.id == s_id:
            ctx.modify(wme, busy=False)

    ctx.print(
        f"  [Workflow] Treated Patient {p_id}. Bed {b_id} and Staff {s_id} are now FREE."
    )


# ── Execution ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Hospital Patient Triage Expert System ===")

    # 1. Setup Staff and Beds
    print("\nInitializing hospital resources...")
    engine.assert_fact(Staff(id="Dr_House", name="Gregory House", role="doctor"))
    engine.assert_fact(Staff(id="Nurse_Jackie", name="Jackie Peyton", role="nurse"))
    engine.assert_fact(Bed(id="Bed_A"))
    engine.assert_fact(Bed(id="Bed_B"))

    # 2. Add patients arriving (Alice is standard, Bob is critical, Carol is standard)
    # Triage prioritization will place Bob first, despite Alice arriving first.
    print("\nAdmitting patients...")

    engine.assert_fact(Patient(id="P01", name="Alice", severity=2))
    engine.assert_fact(PatientStatus(patient_id="P01", status="waiting"))

    engine.assert_fact(Patient(id="P02", name="Bob", severity=5))  # CRITICAL!
    engine.assert_fact(PatientStatus(patient_id="P02", status="waiting"))

    engine.assert_fact(Patient(id="P03", name="Carol", severity=3))
    engine.assert_fact(PatientStatus(patient_id="P03", status="waiting"))

    print("\nRunning rule cycle...")
    fired = engine.run()
    print(f"\nRules fired: {fired}")

    print("\n── Final Patient Statuses ──")
    for stat in engine.facts(PatientStatus):
        print(f"  Patient: {stat.patient_id} - Status: {stat.status.upper()}")
