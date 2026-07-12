#!/usr/bin/env python3
"""Smart Home Automation & Sensor Fusion Example.

Demonstrates:
  - Multi-sensor fusion (motion, temperature, luminance).
  - Occupancy inference based on activity.
  - Reactive rule chaining (sensor -> occupancy -> climate/lighting controls).
  - Negated condition elements (avoiding sending commands if already in that state).
"""

from dataclasses import dataclass
from typing import Any

from rete import Eq, Fact, Lt, Pat, ReteEngine, RuleContext, Var


# ── Fact types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class SensorReading(Fact):
    sensor_id: str
    room: str
    type: str  # "motion", "temperature", "light"
    value: float | str  # float for temp/light, "active"/"inactive" for motion


@dataclass(frozen=True)
class Occupancy(Fact):
    room: str
    status: str  # "occupied", "vacant"


@dataclass(frozen=True)
class DeviceStatus(Fact):
    device_id: str
    room: str
    type: str  # "light", "hvac"
    state: str  # "on", "off", "heating", "cooling", "idle"


@dataclass(frozen=True)
class HouseMode(Fact):
    mode: str  # "day", "night", "away"


@dataclass(frozen=True)
class ActionLog(Fact):
    message: str


# ── Engine & Rules ─────────────────────────────────────────────────

engine = ReteEngine(strategy="lex")


@engine.rule(
    Pat(SensorReading, room=Var("r"), type=Eq("motion"), value=Eq("active")),
    ~Pat(Occupancy, room=Var("r"), status=Eq("occupied")),
)
def detect_occupancy(ctx: RuleContext, r: Any) -> None:
    """If motion is active and room is not marked occupied, mark it occupied."""
    # Retract vacancy if it exists
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, Occupancy) and wme.fact.room == r:
            ctx.retract(wme)
    ctx.assert_fact(Occupancy(room=r, status="occupied"))
    ctx.print(f"  [Occupancy] Motion detected in {r}. Setting status to occupied.")


@engine.rule(
    Pat(Occupancy, room=Var("r"), status=Eq("occupied")),
    Pat(SensorReading, room=Var("r"), type=Eq("light"), value=Lt(100.0)),
    Pat(
        DeviceStatus,
        device_id=Var("d"),
        room=Var("r"),
        type=Eq("light"),
        state=Eq("off"),
    ),
    ~Pat(HouseMode, mode=Eq("away")),
)
def turn_on_lights(ctx: RuleContext, r: Any, d: Any) -> None:
    """If room is occupied, light is low (< 100 lux), lights are off, and mode is not away, turn them on."""
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, DeviceStatus) and wme.fact.device_id == d:
            ctx.modify(wme, state="on")
    ctx.assert_fact(
        ActionLog(message=f"Turned on lights '{d}' in dark occupied room '{r}'")
    )
    ctx.print(f"  [Lights] Room {r} is dark & occupied. Turning ON light: {d}")


@engine.rule(
    Pat(Occupancy, room=Var("r"), status=Eq("occupied")),
    Pat(SensorReading, room=Var("r"), type=Eq("temperature"), value=Lt(19.0)),
    Pat(
        DeviceStatus,
        device_id=Var("h"),
        room=Var("r"),
        type=Eq("hvac"),
        state=Eq("idle"),
    ),
    ~Pat(HouseMode, mode=Eq("away")),
)
def trigger_heating(ctx: RuleContext, r: Any, h: Any) -> None:
    """If room is occupied, temperature is cold (< 19.0°C), HVAC is idle, and mode is not away, turn on heating."""
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, DeviceStatus) and wme.fact.device_id == h:
            ctx.modify(wme, state="heating")
    ctx.assert_fact(
        ActionLog(
            message=f"Activated heating for HVAC '{h}' in cold occupied room '{r}'"
        )
    )
    ctx.print(
        f"  [Climate] Temperature in occupied room {r} is cold. Turning ON HVAC heating: {h}"
    )


@engine.rule(
    Pat(HouseMode, mode=Eq("away")),
    Pat(DeviceStatus, device_id=Var("d"), state=Eq("on")),
)
def auto_shutoff_away(ctx: RuleContext, d: Any) -> None:
    """If house mode changes to 'away', turn off any active devices (lights/HVAC)."""
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, DeviceStatus) and wme.fact.device_id == d:
            ctx.modify(wme, state="off")
    ctx.assert_fact(ActionLog(message=f"Auto shutoff device '{d}' due to AWAY mode"))
    ctx.print(f"  [Away Mode] Shutting down active device '{d}' to save energy.")


# ── Execution ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Smart Home Expert System ===")

    # Initial setup
    print("\nInitializing House State...")
    engine.assert_fact(HouseMode(mode="day"))
    engine.assert_fact(
        DeviceStatus(
            device_id="living_room_light", room="living_room", type="light", state="off"
        )
    )
    engine.assert_fact(
        DeviceStatus(
            device_id="living_room_hvac", room="living_room", type="hvac", state="idle"
        )
    )
    engine.assert_fact(Occupancy(room="living_room", status="vacant"))

    # Assert sensor readings
    print("\nReceiving Sensor Events...")
    engine.assert_fact(
        SensorReading(
            sensor_id="LR_motion_1", room="living_room", type="motion", value="active"
        )
    )
    engine.assert_fact(
        SensorReading(
            sensor_id="LR_lux_1", room="living_room", type="light", value=50.0
        )
    )
    engine.assert_fact(
        SensorReading(
            sensor_id="LR_temp_1", room="living_room", type="temperature", value=18.2
        )
    )

    print("\nRunning rule cycle...")
    fired = engine.run()
    print(f"\nRules fired: {fired}")

    print("\nTriggering AWAY mode...")
    # Change mode to away (by asserting new and retracting old)
    for wme in list(engine.wm._facts.values()):
        if isinstance(wme.fact, HouseMode):
            engine.retract(wme)
    engine.assert_fact(HouseMode(mode="away"))

    fired_away = engine.run()
    print(f"AWAY rules fired: {fired_away}")

    print("\n── Final Device States ──")
    for fact in engine.facts(DeviceStatus):
        print(f"  {fact}")
    print("── Actions Logged ──")
    for fact in engine.facts(ActionLog):
        print(f"  {fact}")
