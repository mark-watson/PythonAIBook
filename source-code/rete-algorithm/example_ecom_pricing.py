#!/usr/bin/env python3
"""E-Commerce Pricing & Discount Engine Example.

Demonstrates:
  - Complex pattern matching on order attributes (category, loyalty tier).
  - Multiple rules generating potential discounts.
  - Stateful accumulation using a pattern processing loop.
  - Using Negated Condition Elements (NCEs) as a completion barrier (join matching).
"""

from dataclasses import dataclass

from rete import Eq, Fact, Pat, ReteEngine, RuleContext, Var


def _apply_loyalty_discount(
    ctx: RuleContext,
    customer_id: str,
    product_id: str,
    quantity: int,
    price: float,
    discount_pct: float,
    tier_name: str,
) -> None:
    """Helper to apply a loyalty tier discount on items."""
    discount_amount = price * quantity * discount_pct
    ctx.assert_fact(
        AppliedDiscount(
            customer_id=customer_id,
            product_id=product_id,
            amount=discount_amount,
            reason=f"{tier_name} Loyalty ({discount_pct * 100:.0f}%)",
        )
    )
    ctx.print(
        f"  [Discounts] {tier_name} customer {customer_id}: {discount_pct * 100:.0f}% discount of {discount_amount:.2f} on {product_id}"
    )


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
    Pat(Customer, id=Var("customer_id"), loyalty_tier=Eq("gold")),
    Pat(
        CartItem,
        customer_id=Var("customer_id"),
        product_id=Var("product_id"),
        quantity=Var("q"),
        price=Var("p"),
    ),
    ~Pat(
        AppliedDiscount,
        customer_id=Var("customer_id"),
        product_id=Var("product_id"),
        reason=Eq("Gold Loyalty (10%)"),
    ),
)
def apply_gold_discount(
    ctx: RuleContext, customer_id: str, product_id: str, q: int, p: float
) -> None:
    """Gold loyalty tier gets 10% discount on all items."""
    _apply_loyalty_discount(ctx, customer_id, product_id, q, p, 0.10, "Gold")


@engine.rule(
    Pat(Customer, id=Var("customer_id"), loyalty_tier=Eq("silver")),
    Pat(
        CartItem,
        customer_id=Var("customer_id"),
        product_id=Var("product_id"),
        quantity=Var("q"),
        price=Var("p"),
    ),
    ~Pat(
        AppliedDiscount,
        customer_id=Var("customer_id"),
        product_id=Var("product_id"),
        reason=Eq("Silver Loyalty (5%)"),
    ),
)
def apply_silver_discount(
    ctx: RuleContext, customer_id: str, product_id: str, q: int, p: float
) -> None:
    """Silver loyalty tier gets 5% discount on all items."""
    _apply_loyalty_discount(ctx, customer_id, product_id, q, p, 0.05, "Silver")


@engine.rule(
    Pat(
        CartItem,
        customer_id=Var("customer_id"),
        product_id=Var("product_id"),
        category=Var("cat"),
        quantity=Var("q"),
        price=Var("p"),
    ),
    Pat(ActiveCampaign, category=Var("cat"), discount_pct=Var("pct")),
    ~Pat(
        AppliedDiscount,
        customer_id=Var("customer_id"),
        product_id=Var("product_id"),
        reason=Eq("Active Category Campaign"),
    ),
)
def apply_campaign_discount(
    ctx: RuleContext,
    customer_id: str,
    product_id: str,
    q: int,
    p: float,
    pct: float,
) -> None:
    """Applies active category-based campaign discount (e.g. 20% off apparel)."""
    discount_amount = p * q * pct
    ctx.assert_fact(
        AppliedDiscount(
            customer_id=customer_id,
            product_id=product_id,
            amount=discount_amount,
            reason="Active Category Campaign",
        )
    )
    ctx.print(
        f"  [Discounts] Category campaign '{pct * 100}% off': discount of {discount_amount:.2f} on {product_id}"
    )


@engine.rule(
    Pat(
        CartItem,
        customer_id=Var("customer_id"),
        product_id=Var("product_id"),
        quantity=Var("q"),
        price=Var("p"),
    ),
    ~Pat(
        AppliedDiscount,
        customer_id=Var("customer_id"),
        product_id=Var("product_id"),
        reason=Eq("Bulk Discount (15%)"),
    ),
)
def apply_bulk_discount(
    ctx: RuleContext, customer_id: str, product_id: str, q: int, p: float
) -> None:
    """Applies a 15% bulk discount if ordering more than 5 units of any product."""
    if q <= 5:
        return
    discount_amount = p * q * 0.15
    ctx.assert_fact(
        AppliedDiscount(
            customer_id=customer_id,
            product_id=product_id,
            amount=discount_amount,
            reason="Bulk Discount (15%)",
        )
    )
    ctx.print(
        f"  [Discounts] Bulk discount: 15% discount of {discount_amount:.2f} on {product_id}"
    )


# ── Step 2: Accumulating Cart Summary ──────────────────────────────


@engine.rule(
    Pat(Customer, id=Var("customer_id")),
    ~Pat(CartSummary, customer_id=Var("customer_id")),
)
def init_cart_summary(ctx: RuleContext, customer_id: str) -> None:
    """Initialize empty CartSummary when customer exists."""
    ctx.assert_fact(CartSummary(customer_id=customer_id))
    ctx.print(f"  [Summary] Initialized summary tracker for {customer_id}")


@engine.rule(
    Pat(
        CartItem,
        customer_id=Var("customer_id"),
        product_id=Var("product_id"),
        quantity=Var("q"),
        price=Var("p"),
        processed=Eq(False),
    ),
    Pat(
        CartSummary,
        customer_id=Var("customer_id"),
        subtotal=Var("sub"),
        total_discount=Var("td"),
    ),
)
def accumulate_subtotal(
    ctx: RuleContext,
    customer_id: str,
    product_id: str,
    q: int,
    p: float,
    sub: float,
    td: float,
) -> None:
    """Add items to subtotal and mark them processed."""
    added_subtotal = p * q
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, CartSummary):
            ctx.modify(wme, subtotal=sub + added_subtotal)
        elif isinstance(wme.fact, CartItem):
            ctx.modify(wme, processed=True)
    ctx.print(
        f"  [Summary] Factored item {product_id} (subtotal: +{added_subtotal:.2f}) into cart summary"
    )


@engine.rule(
    Pat(
        AppliedDiscount,
        customer_id=Var("customer_id"),
        product_id=Var("product_id"),
        amount=Var("a"),
        processed=Eq(False),
    ),
    Pat(
        CartSummary,
        customer_id=Var("customer_id"),
        subtotal=Var("sub"),
        total_discount=Var("td"),
    ),
)
def accumulate_discounts(
    ctx: RuleContext,
    customer_id: str,
    product_id: str,
    a: float,
    sub: float,
    td: float,
) -> None:
    """Add applied discounts to total discount and mark them processed."""
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, CartSummary):
            ctx.modify(wme, total_discount=td + a)
        elif isinstance(wme.fact, AppliedDiscount):
            ctx.modify(wme, processed=True)
    ctx.print(
        f"  [Summary] Factored discount of {a:.2f} on {product_id} into cart summary"
    )


# ── Step 3: Producing Invoice (Barrier Completion) ────────────────


@engine.rule(
    Pat(
        CartSummary,
        customer_id=Var("customer_id"),
        subtotal=Var("sub"),
        total_discount=Var("td"),
    ),
    ~Pat(CartItem, customer_id=Var("customer_id"), processed=Eq(False)),
    ~Pat(AppliedDiscount, customer_id=Var("customer_id"), processed=Eq(False)),
    ~Pat(Invoice, customer_id=Var("customer_id")),
)
def create_invoice(ctx: RuleContext, customer_id: str, sub: float, td: float) -> None:
    """Generate the final invoice when all items and discounts have been summarized."""
    total = max(0.0, sub - td)
    ctx.assert_fact(
        Invoice(customer_id=customer_id, subtotal=sub, discount=td, total=total)
    )
    ctx.print(
        f"  [Invoice] CREATED INVOICE FOR {customer_id}: Subtotal={sub:.2f}, Discount={td:.2f}, Final Total={total:.2f}"
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
