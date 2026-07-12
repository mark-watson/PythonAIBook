#!/usr/bin/env python3
"""E-Commerce Pricing & Discount Engine Example.

Demonstrates:
  - Complex pattern matching on order attributes (category, loyalty tier).
  - Multiple rules generating potential discounts.
  - Stateful accumulation using a pattern processing loop.
  - Using Negated Condition Elements (NCEs) as a completion barrier (join matching).
"""

from dataclasses import dataclass
from typing import Any

from rete import Eq, Fact, Pat, ReteEngine, RuleContext, Var


# ── Fact types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Customer(Fact):
    id: str
    loyalty_tier: str  # "regular", "silver", "gold"


@dataclass(frozen=True)
class CartItem(Fact):
    customer_id: str
    product_id: str
    category: str
    quantity: int
    price: float
    processed: bool = False  # Track if factored into subtotal


@dataclass(frozen=True)
class ActiveCampaign(Fact):
    category: str
    discount_pct: float  # e.g. 0.20 for 20% off


@dataclass(frozen=True)
class AppliedDiscount(Fact):
    customer_id: str
    product_id: str
    amount: float
    reason: str
    processed: bool = False  # Track if factored into total discount


@dataclass(frozen=True)
class CartSummary(Fact):
    customer_id: str
    subtotal: float = 0.0
    total_discount: float = 0.0


@dataclass(frozen=True)
class Invoice(Fact):
    customer_id: str
    subtotal: float
    discount: float
    total: float


# ── Engine & Rules ─────────────────────────────────────────────────

engine = ReteEngine(strategy="lex")


# ── Step 1: Matching and Creating Discounts ───────────────────────


@engine.rule(
    Pat(Customer, id=Var("c_id"), loyalty_tier=Eq("gold")),
    Pat(
        CartItem,
        customer_id=Var("c_id"),
        product_id=Var("p_id"),
        quantity=Var("q"),
        price=Var("p"),
    ),
    ~Pat(
        AppliedDiscount,
        customer_id=Var("c_id"),
        product_id=Var("p_id"),
        reason=Eq("Gold Loyalty (10%)"),
    ),
)
def apply_gold_discount(ctx: RuleContext, c_id: Any, p_id: Any, q: Any, p: Any) -> None:
    """Gold loyalty tier gets 10% discount on all items."""
    discount_amount = p * q * 0.10
    ctx.assert_fact(
        AppliedDiscount(
            customer_id=c_id,
            product_id=p_id,
            amount=discount_amount,
            reason="Gold Loyalty (10%)",
        )
    )
    ctx.print(
        f"  [Discounts] Gold customer {c_id}: 10% discount of {discount_amount:.2f} on {p_id}"
    )


@engine.rule(
    Pat(Customer, id=Var("c_id"), loyalty_tier=Eq("silver")),
    Pat(
        CartItem,
        customer_id=Var("c_id"),
        product_id=Var("p_id"),
        quantity=Var("q"),
        price=Var("p"),
    ),
    ~Pat(
        AppliedDiscount,
        customer_id=Var("c_id"),
        product_id=Var("p_id"),
        reason=Eq("Silver Loyalty (5%)"),
    ),
)
def apply_silver_discount(
    ctx: RuleContext, c_id: Any, p_id: Any, q: Any, p: Any
) -> None:
    """Silver loyalty tier gets 5% discount on all items."""
    discount_amount = p * q * 0.05
    ctx.assert_fact(
        AppliedDiscount(
            customer_id=c_id,
            product_id=p_id,
            amount=discount_amount,
            reason="Silver Loyalty (5%)",
        )
    )
    ctx.print(
        f"  [Discounts] Silver customer {c_id}: 5% discount of {discount_amount:.2f} on {p_id}"
    )


@engine.rule(
    Pat(
        CartItem,
        customer_id=Var("c_id"),
        product_id=Var("p_id"),
        category=Var("cat"),
        quantity=Var("q"),
        price=Var("p"),
    ),
    Pat(ActiveCampaign, category=Var("cat"), discount_pct=Var("pct")),
    ~Pat(
        AppliedDiscount,
        customer_id=Var("c_id"),
        product_id=Var("p_id"),
        reason=Eq("Active Category Campaign"),
    ),
)
def apply_campaign_discount(
    ctx: RuleContext, c_id: Any, p_id: Any, q: Any, p: Any, pct: Any
) -> None:
    """Applies active category-based campaign discount (e.g. 20% off apparel)."""
    discount_amount = p * q * pct
    ctx.assert_fact(
        AppliedDiscount(
            customer_id=c_id,
            product_id=p_id,
            amount=discount_amount,
            reason="Active Category Campaign",
        )
    )
    ctx.print(
        f"  [Discounts] Category campaign '{pct * 100}% off {pct}': discount of {discount_amount:.2f} on {p_id}"
    )


@engine.rule(
    Pat(
        CartItem,
        customer_id=Var("c_id"),
        product_id=Var("p_id"),
        quantity=Var("q"),
        price=Var("p"),
    ),
    ~Pat(
        AppliedDiscount,
        customer_id=Var("c_id"),
        product_id=Var("p_id"),
        reason=Eq("Bulk Discount (15%)"),
    ),
)
def apply_bulk_discount(ctx: RuleContext, c_id: Any, p_id: Any, q: Any, p: Any) -> None:
    """Applies a 15% bulk discount if ordering more than 5 units of any product."""
    if q <= 5:
        return
    discount_amount = p * q * 0.15
    ctx.assert_fact(
        AppliedDiscount(
            customer_id=c_id,
            product_id=p_id,
            amount=discount_amount,
            reason="Bulk Discount (15%)",
        )
    )
    ctx.print(
        f"  [Discounts] Bulk discount: 15% discount of {discount_amount:.2f} on {p_id}"
    )


# ── Step 2: Accumulating Cart Summary ──────────────────────────────


@engine.rule(
    Pat(Customer, id=Var("c_id")),
    ~Pat(CartSummary, customer_id=Var("c_id")),
)
def init_cart_summary(ctx: RuleContext, c_id: Any) -> None:
    """Initialize empty CartSummary when customer exists."""
    ctx.assert_fact(CartSummary(customer_id=c_id))
    ctx.print(f"  [Summary] Initialized summary tracker for {c_id}")


@engine.rule(
    Pat(
        CartItem,
        customer_id=Var("c_id"),
        product_id=Var("p_id"),
        quantity=Var("q"),
        price=Var("p"),
        processed=Eq(False),
    ),
    Pat(
        CartSummary,
        customer_id=Var("c_id"),
        subtotal=Var("sub"),
        total_discount=Var("td"),
    ),
)
def accumulate_subtotal(
    ctx: RuleContext, c_id: Any, p_id: Any, q: Any, p: Any, sub: Any, td: Any
) -> None:
    """Add items to subtotal and mark them processed."""
    added_subtotal = p * q
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, CartSummary):
            ctx.modify(wme, subtotal=sub + added_subtotal)
        elif isinstance(wme.fact, CartItem):
            ctx.modify(wme, processed=True)
    ctx.print(
        f"  [Summary] Factored item {p_id} (subtotal: +{added_subtotal:.2f}) into cart summary"
    )


@engine.rule(
    Pat(
        AppliedDiscount,
        customer_id=Var("c_id"),
        product_id=Var("p_id"),
        amount=Var("a"),
        processed=Eq(False),
    ),
    Pat(
        CartSummary,
        customer_id=Var("c_id"),
        subtotal=Var("sub"),
        total_discount=Var("td"),
    ),
)
def accumulate_discounts(
    ctx: RuleContext, c_id: Any, p_id: Any, a: Any, sub: Any, td: Any
) -> None:
    """Add applied discounts to total discount and mark them processed."""
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, CartSummary):
            ctx.modify(wme, total_discount=td + a)
        elif isinstance(wme.fact, AppliedDiscount):
            ctx.modify(wme, processed=True)
    ctx.print(f"  [Summary] Factored discount of {a:.2f} on {p_id} into cart summary")


# ── Step 3: Producing Invoice (Barrier Completion) ────────────────


@engine.rule(
    Pat(
        CartSummary,
        customer_id=Var("c_id"),
        subtotal=Var("sub"),
        total_discount=Var("td"),
    ),
    ~Pat(CartItem, customer_id=Var("c_id"), processed=Eq(False)),
    ~Pat(AppliedDiscount, customer_id=Var("c_id"), processed=Eq(False)),
    ~Pat(Invoice, customer_id=Var("c_id")),
)
def create_invoice(ctx: RuleContext, c_id: Any, sub: Any, td: Any) -> None:
    """Generate the final invoice when all items and discounts have been summarized."""
    total = max(0.0, sub - td)
    ctx.assert_fact(Invoice(customer_id=c_id, subtotal=sub, discount=td, total=total))
    ctx.print(
        f"  [Invoice] CREATED INVOICE FOR {c_id}: Subtotal={sub:.2f}, Discount={td:.2f}, Final Total={total:.2f}"
    )


# ── Execution ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== E-Commerce Pricing & Discount Engine ===")

    # Setup database/rules facts
    print("\nSetting up active campaigns...")
    engine.assert_fact(
        ActiveCampaign(category="apparel", discount_pct=0.20)
    )  # 20% off apparel

    print("\nAdding customers and items...")
    # Alice: Gold loyalty customer ordering multiple items
    engine.assert_fact(Customer(id="Alice", loyalty_tier="gold"))
    engine.assert_fact(
        CartItem(
            customer_id="Alice",
            product_id="jacket_1",
            category="apparel",
            quantity=1,
            price=150.00,
        )
    )
    engine.assert_fact(
        CartItem(
            customer_id="Alice",
            product_id="socks_5",
            category="clothing",
            quantity=10,
            price=12.00,
        )
    )  # Bulk socks

    # Bob: Silver loyalty customer
    engine.assert_fact(Customer(id="Bob", loyalty_tier="silver"))
    engine.assert_fact(
        CartItem(
            customer_id="Bob",
            product_id="shirt_2",
            category="apparel",
            quantity=2,
            price=45.00,
        )
    )

    print("\nRunning rule calculations...")
    fired = engine.run()
    print(f"\nRules fired: {fired}")

    print("\n── Final Invoices ──")
    for inv in engine.facts(Invoice):
        print(f"  Customer: {inv.customer_id}")
        print(f"    Subtotal:       ${inv.subtotal:6.2f}")
        print(f"    Total Discount: ${inv.discount:6.2f}")
        print(f"    Grand Total:    ${inv.total:6.2f}")
        print()
