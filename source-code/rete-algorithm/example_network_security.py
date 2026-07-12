#!/usr/bin/env python3
"""Network Intrusion Detection & Threat Alerting Example.

Demonstrates:
  - Consuming event logs (asserting events, processing, and retracting them).
  - State tracking (incrementing counters, tracking collections).
  - Variable joins to match event sources with counters.
  - Multi-step reasoning (events -> counter updates -> threshold alerts -> blocking).
"""

from dataclasses import dataclass
from typing import Any

from rete import Eq, Fact, Gt, Pat, ReteEngine, RuleContext, Test, Var


# ── Fact types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class LoginAttempt(Fact):
    ip: str
    username: str
    result: str  # "success" or "fail"


@dataclass(frozen=True)
class ConnectionEvent(Fact):
    ip: str
    port: int


@dataclass(frozen=True)
class FailedLoginCounter(Fact):
    ip: str
    count: int


@dataclass(frozen=True)
class PortScanCounter(Fact):
    ip: str
    ports: tuple[int, ...] = ()


@dataclass(frozen=True)
class BlockedIP(Fact):
    ip: str


@dataclass(frozen=True)
class SecurityAlert(Fact):
    ip: str
    severity: str  # "medium", "high"
    message: str


# ── Engine & Rules ─────────────────────────────────────────────────

engine = ReteEngine(strategy="lex")


@engine.rule(
    Pat(LoginAttempt, ip=Var("ip"), result=Eq("fail")),
    ~Pat(FailedLoginCounter, ip=Var("ip")),
)
def init_failed_login_counter(ctx: RuleContext, ip: Any) -> None:
    """If a failed login occurs and no counter exists for this IP, create one."""
    ctx.assert_fact(FailedLoginCounter(ip=ip, count=1))
    # Consume the login attempt event
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, LoginAttempt):
            ctx.retract(wme)
    ctx.print(f"  [Security] Created failed login counter for {ip}")


@engine.rule(
    Pat(LoginAttempt, ip=Var("ip"), result=Eq("fail")),
    Pat(FailedLoginCounter, ip=Var("ip"), count=Var("c")),
)
def increment_failed_login_counter(ctx: RuleContext, ip: Any, c: Any) -> None:
    """If a failed login occurs and a counter exists, increment the counter."""
    # Increment counter
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, FailedLoginCounter):
            ctx.modify(wme, count=c + 1)
        elif isinstance(wme.fact, LoginAttempt):
            # Consume event
            ctx.retract(wme)
    ctx.print(f"  [Security] Incremented failed login counter for {ip} to {c + 1}")


@engine.rule(
    Pat(LoginAttempt, ip=Var("ip"), result=Eq("success")),
    Pat(FailedLoginCounter, ip=Var("ip")),
)
def reset_failed_login_counter(ctx: RuleContext, ip: Any) -> None:
    """If a successful login occurs, reset/retract any failed login counter for that IP."""
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, FailedLoginCounter):
            ctx.retract(wme)
        elif isinstance(wme.fact, LoginAttempt):
            # Consume success event
            ctx.retract(wme)
    ctx.print(
        f"  [Security] Reset failed login counter for {ip} due to successful login"
    )


@engine.rule(
    Pat(FailedLoginCounter, ip=Var("ip"), count=Gt(2)),
    ~Pat(BlockedIP, ip=Var("ip")),
)
def detect_brute_force(ctx: RuleContext, ip: Any) -> None:
    """If failed login counter exceeds 2 and IP is not blocked, block it and issue HIGH alert."""
    ctx.assert_fact(BlockedIP(ip=ip))
    ctx.assert_fact(
        SecurityAlert(
            ip=ip,
            severity="high",
            message="Multiple failed login attempts (Brute Force)",
        )
    )
    ctx.print(f"  [ALERT] HIGH: Brute force detected from {ip}! Blocking IP.")


@engine.rule(
    Pat(ConnectionEvent, ip=Var("ip"), port=Var("p")),
    ~Pat(PortScanCounter, ip=Var("ip")),
)
def init_port_scan_counter(ctx: RuleContext, ip: Any, p: Any) -> None:
    """If a connection occurs and no scan counter exists, create one with the port."""
    ctx.assert_fact(PortScanCounter(ip=ip, ports=(p,)))
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, ConnectionEvent):
            ctx.retract(wme)
    ctx.print(f"  [Security] Initialized port scan counter for {ip} on port {p}")


@engine.rule(
    Pat(ConnectionEvent, ip=Var("ip"), port=Var("p")),
    Pat(PortScanCounter, ip=Var("ip"), ports=Var("pts")),
)
def update_port_scan_counter(ctx: RuleContext, ip: Any, p: Any, pts: Any) -> None:
    """If connection occurs and port is new, append it. Otherwise consume connection."""
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, ConnectionEvent):
            ctx.retract(wme)

    if p not in pts:
        new_ports = pts + (p,)
        for wme in ctx.token_wmes:
            if isinstance(wme.fact, PortScanCounter):
                ctx.modify(wme, ports=new_ports)
        ctx.print(
            f"  [Security] Logged new port {p} scan from {ip}. Ports: {new_ports}"
        )


@engine.rule(
    Pat(PortScanCounter, ip=Var("ip"), ports=Test(lambda ports: len(ports) >= 3)),
    ~Pat(BlockedIP, ip=Var("ip")),
)
def detect_port_scan(ctx: RuleContext, ip: Any) -> None:
    """If an IP has connected to 3 or more distinct ports, block it and issue MEDIUM alert."""
    ctx.assert_fact(BlockedIP(ip=ip))
    ctx.assert_fact(
        SecurityAlert(
            ip=ip, severity="medium", message="Distinct ports scanned (Port Scan)"
        )
    )
    ctx.print(f"  [ALERT] MEDIUM: Port scan detected from {ip}! Blocking IP.")


@engine.rule(
    Pat(ConnectionEvent, ip=Var("ip")),
    Pat(BlockedIP, ip=Var("ip")),
)
def drop_blocked_ip_traffic(ctx: RuleContext, ip: Any) -> None:
    """Silently drop connections from blocked IPs."""
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, ConnectionEvent):
            ctx.retract(wme)
    ctx.print(f"  [Security] DROPPED connection attempt from blocked IP {ip}")


# ── Execution ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Network Security Monitoring Expert System ===")

    # Assert network events
    print("\nSimulating Brute-Force Attack...")
    engine.assert_fact(
        LoginAttempt(ip="192.168.1.100", username="admin", result="fail")
    )
    engine.assert_fact(
        LoginAttempt(ip="192.168.1.100", username="admin", result="fail")
    )
    engine.assert_fact(
        LoginAttempt(ip="192.168.1.100", username="admin", result="fail")
    )

    # Run the cycle to trigger block
    engine.run()

    print("\nSimulating connection from the blocked IP...")
    engine.assert_fact(ConnectionEvent(ip="192.168.1.100", port=80))
    engine.run()

    print("\nSimulating Port Scan from a new IP...")
    engine.assert_fact(ConnectionEvent(ip="10.0.0.50", port=22))
    engine.assert_fact(ConnectionEvent(ip="10.0.0.50", port=80))
    engine.assert_fact(ConnectionEvent(ip="10.0.0.50", port=443))
    engine.run()

    print("\nSimulating failed then successful login...")
    engine.assert_fact(LoginAttempt(ip="172.16.0.5", username="user1", result="fail"))
    engine.run()
    engine.assert_fact(
        LoginAttempt(ip="172.16.0.5", username="user1", result="success")
    )
    engine.run()

    print("\n── Active Alerts ──")
    for alert in engine.facts(SecurityAlert):
        print(f"  [{alert.severity.upper()}] IP: {alert.ip} - {alert.message}")

    print("\n── Blocked IPs ──")
    for block in engine.facts(BlockedIP):
        print(f"  Blocked: {block.ip}")
