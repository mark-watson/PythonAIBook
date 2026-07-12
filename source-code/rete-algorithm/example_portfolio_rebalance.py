#!/usr/bin/env python3
"""Financial Portfolio Rebalancing & Alerting Example.

Demonstrates:
  - Aggregating portfolio total value across multiple asset holdings.
  - Computing deviation (drift) from target percentages.
  - Chaining execution barrier rules (wait until all asset holdings are totaled).
  - Conditionally emitting "buy" and "sell" trades based on tolerance thresholds.
"""

from dataclasses import dataclass
from rete import Eq, Fact, Pat, ReteEngine, Var


# ── Fact types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class TargetAllocation(Fact):
    portfolio_id: str
    asset_class: str
    target_pct: float  # e.g., 0.40 for 40% target


@dataclass(frozen=True)
class AssetAllocation(Fact):
    portfolio_id: str
    asset_class: str
    current_value: float
    processed: bool = False  # Track if factored into portfolio total value


@dataclass(frozen=True)
class PortfolioSummary(Fact):
    portfolio_id: str
    total_value: float = 0.0


@dataclass(frozen=True)
class RebalanceTrade(Fact):
    portfolio_id: str
    asset_class: str
    action: str  # "buy" or "sell"
    amount: float
    deviation_pct: float


# ── Engine & Rules ─────────────────────────────────────────────────

engine = ReteEngine(strategy="lex")


# ── Step 1: Compute Portfolio Total Value ──────────────────────────


@engine.rule(
    Pat(TargetAllocation, portfolio_id=Var("p_id")),
    ~Pat(PortfolioSummary, portfolio_id=Var("p_id")),
)
def init_portfolio_summary(ctx, p_id):
    """Create a PortfolioSummary tracker if target allocations exist."""
    ctx.assert_fact(PortfolioSummary(portfolio_id=p_id))
    ctx.print(f"  [Portfolio] Initialized summary tracker for {p_id}")


@engine.rule(
    Pat(
        AssetAllocation,
        portfolio_id=Var("p_id"),
        asset_class=Var("ac"),
        current_value=Var("val"),
        processed=Eq(False),
    ),
    Pat(PortfolioSummary, portfolio_id=Var("p_id"), total_value=Var("tot")),
)
def sum_portfolio_holdings(ctx, p_id, ac, val, tot):
    """Aggregate holdings values into the summary and mark the allocation processed."""
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, PortfolioSummary):
            ctx.modify(wme, total_value=tot + val)
        elif isinstance(wme.fact, AssetAllocation):
            ctx.modify(wme, processed=True)
    ctx.print(
        f"  [Portfolio] Added holding '{ac}' (${val:,.2f}) to total value of {p_id}"
    )


# ── Step 2: Evaluate Drift & Generate Rebalance Actions ────────────


@engine.rule(
    # Barrier: Wait until all asset allocations for the portfolio have been totaled
    Pat(PortfolioSummary, portfolio_id=Var("p_id"), total_value=Var("tot")),
    Pat(
        AssetAllocation,
        portfolio_id=Var("p_id"),
        asset_class=Var("ac"),
        current_value=Var("curr"),
    ),
    Pat(
        TargetAllocation,
        portfolio_id=Var("p_id"),
        asset_class=Var("ac"),
        target_pct=Var("t_pct"),
    ),
    ~Pat(AssetAllocation, portfolio_id=Var("p_id"), processed=Eq(False)),
    ~Pat(RebalanceTrade, portfolio_id=Var("p_id"), asset_class=Var("ac")),
)
def check_drift_and_rebalance(ctx, p_id, ac, curr, t_pct, tot):
    """Detects if an asset's drift from target is > 5% and suggests buys/sells."""
    if tot <= 0.0:
        return

    current_pct = curr / tot
    drift = current_pct - t_pct
    threshold = 0.05  # 5% drift threshold

    if abs(drift) > threshold:
        # Calculate how much to trade to bring it back to target
        target_val = tot * t_pct
        trade_amount = abs(curr - target_val)
        action = "sell" if drift > 0 else "buy"

        ctx.assert_fact(
            RebalanceTrade(
                portfolio_id=p_id,
                asset_class=ac,
                action=action,
                amount=trade_amount,
                deviation_pct=drift,
            )
        )

        ctx.print(
            f"  [Rebalance Alert] {ac} in portfolio {p_id} has drifted by {drift * 100:+.1f}% "
            f"(Target: {t_pct * 100:.0f}%, Current: {current_pct * 100:.1f}%). "
            f"Action required: {action.upper()} ${trade_amount:,.2f}"
        )


# ── Execution ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Financial Portfolio Rebalancing Engine ===")

    # 1. Setup Portfolio Target Allocations
    # Target: US Equities: 40%, Int'l Equities: 20%, Bonds: 30%, Cash: 10%
    print("\nRegistering portfolio targets...")
    engine.assert_fact(
        TargetAllocation(
            portfolio_id="P_RETIRE", asset_class="US_EQUITIES", target_pct=0.40
        )
    )
    engine.assert_fact(
        TargetAllocation(
            portfolio_id="P_RETIRE", asset_class="INTL_EQUITIES", target_pct=0.20
        )
    )
    engine.assert_fact(
        TargetAllocation(portfolio_id="P_RETIRE", asset_class="BONDS", target_pct=0.30)
    )
    engine.assert_fact(
        TargetAllocation(portfolio_id="P_RETIRE", asset_class="CASH", target_pct=0.10)
    )

    # 2. Setup Current Asset Holdings (Totaling $100,000)
    # Drift setup:
    # US_EQUITIES:   $55,000 (55% vs 40% Target) -> Drift +15% (SELL)
    # INTL_EQUITIES: $18,000 (18% vs 20% Target) -> Drift -2%  (Within drift limit)
    # BONDS:         $22,000 (22% vs 30% Target) -> Drift -8%  (BUY)
    # CASH:          $5,000  (5% vs 10% Target)  -> Drift -5%  (Within drift limit, boundary case)
    print("\nUpdating current asset holdings...")
    engine.assert_fact(
        AssetAllocation(
            portfolio_id="P_RETIRE", asset_class="US_EQUITIES", current_value=55000.0
        )
    )
    engine.assert_fact(
        AssetAllocation(
            portfolio_id="P_RETIRE", asset_class="INTL_EQUITIES", current_value=18000.0
        )
    )
    engine.assert_fact(
        AssetAllocation(
            portfolio_id="P_RETIRE", asset_class="BONDS", current_value=22000.0
        )
    )
    engine.assert_fact(
        AssetAllocation(
            portfolio_id="P_RETIRE", asset_class="CASH", current_value=5000.0
        )
    )

    print("\nRunning rule calculations...")
    fired = engine.run()
    print(f"\nRules fired: {fired}")

    print("\n── Trade Actions Recommended ──")
    for trade in engine.facts(RebalanceTrade):
        print(
            f"  {trade.action.upper():4s} ${trade.amount:9,.2f} of {trade.asset_class:<15s} (Drift: {trade.deviation_pct * 100:+.1f}%)"
        )
