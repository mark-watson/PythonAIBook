# Expert Systems Using the Rete Algorithm

For decades, rule-based expert systems stood at the forefront of Artificial Intelligence. While deep learning models excel at recognition, perception, and approximate reasoning, rule-based systems remain the gold standard for applications requiring deterministic reasoning, absolute compliance, explicit logic, and high interpretability. 

At the heart of the most powerful forward-chaining expert systems is the **Rete Algorithm**, designed by Charles Forgy in 1974. Rete solves a fundamental scaling challenge: in a naive production system, evaluating $R$ rules against $W$ working memory elements requires $O(R \times W)$ checks on every cycle. As working memory or the rule base grows, performance collapses. Rete achieves high efficiency by compiling rules into a dataflow network that evaluates conditions incrementally, trading computer memory (to cache partial matches) for execution speed.

This chapter walks through a modern, lightweight, and idiomatic Python implementation of the Rete algorithm. We will discuss its design, examine some of the trickiest details in Rete implementation, and analyze six complete, real-world case studies to offer practical advice on designing production-grade expert systems.

All code for this chapter is located in the **source-code/rete-algorithm** directory.

---

## 1. Rete Engine Architecture & Design

Our implementation compiles declarative rules into a unified dataflow graph containing two main sections: the **Alpha Network** and the **Beta Network**.

### The Working Memory (WM)
Working memory stores active assertions, represented as Working Memory Elements (WMEs). In **source-code/rete-algorithm/rete/facts.py**, user facts are defined as frozen Python `@dataclass` objects inheriting from `Fact`. This gives us hashability and immutability out of the box. 

When a fact is asserted, the `WorkingMemory` wraps it in a `WME` container with a unique, monotonically increasing `id` and a `timestamp` indicating its recency.

### The Alpha Network (Intra-Fact Filtering)
The Alpha Network performs single-fact filtering (e.g., checking if a patient's temperature is greater than 38.5).
- **Type Dispatch**: The root `AlphaNetwork` directs incoming WMEs directly to branches corresponding to their specific Python class type using an $O(1)$ type lookup dictionary.
- **Alpha Test Chains**: A chain of `AlphaTestNode` gates filters the fact's fields against constant criteria.
- **Node Sharing**: If multiple rules check the exact same condition (e.g., `temperature > 38.5`), the network shares the corresponding alpha test nodes and routes them to a single shared `AlphaMemory` node, avoiding redundant checks.

### The Beta Network (Inter-Fact Joining)
The Beta Network manages multi-fact relations, checking for variable consistency across different conditions (e.g., joining `Patient(name=Var("n"))` with `Diagnosis(patient=Var("n"))` where both must refer to the same patient name).
- **Tokens**: A `Token` represents a partial match flowing down the beta network. It is structured as a linked list chain pointing to a parent token and a matched WME, allowing memory-efficient prefix sharing.
- **Join Nodes**: `JoinNode` instances cross-reference incoming tokens from their left input (Beta Memory) with WMEs from their right input (Alpha Memory) to verify variable bindings.
- **Negative Nodes**: `NegativeNode` instances implement Negated Condition Elements (NCEs), allowing rules to match only in the *absence* of matching facts.
- **Production Nodes**: Terminal `ProductionNode` instances represent fully matched rules. When a token propagates here, it constructs a rule `Instantiation` and adds it to the conflict set.

### Conflict Resolution & Recognizing Cycles
On each cycle, the `ReteEngine` selects one rule instantiation to fire from the conflict set. It resolves conflicts using:
- **Refraction**: Ensures that a specific combination of facts fires a rule at most once, preventing infinite execution loops on stable data.
- **Salience**: A developer-defined priority number (higher priority rules fire first).
- **Recency (LEX/MEA)**: Prefers instantiations matching the most recently asserted facts (highest timestamps) to focus reasoning on current events.

---

## 2. Implementing the Trickier Parts of Rete

Writing a Rete network reveals several subtle challenges. Below, we examine three of the most complex implementation details in our engine.

### 2.1 The Retraction Cleanup & Token Descendant Bug
When a fact is retracted, the change must propagate down the entire Rete network to remove invalid partial matches. Naive Rete implementations often clean up memory lists checking `t is not token`. This fails because as a token flows through `JoinNode` instances, it is extended to a new child token object containing the newly joined WME.

To fix this, we implemented a robust ancestor-descendant lookup function `is_descendant` that traverses parent pointers up the token chain:

```python
def is_descendant(t: Token, token: Token) -> bool:
    """Return True if *t* is a descendant of *token* (or is *token* itself)."""
    curr: Token | None = t
    while curr is not None:
        if curr is token:
            return True
        curr = curr.parent
    return False
```

Using this helper, any memory node (`BetaMemory`, `NegativeNode` tables, or `ProductionNode` sets) can confidently purge matches downstream:

```python
def left_remove_token(self, token: Token) -> None:
    """Remove all tokens that are descendants of *token* and propagate."""
    surviving = [t for t in self.tokens if not is_descendant(t, token)]
    self.tokens = surviving
    for child in self.children:
        child.left_remove_token(token)
```

### 2.2 Negated Condition Elements (NCEs)
Implementing negative matching (e.g., `~Pat(Diagnosis, patient=Var("n"))`) requires tracking blocker relationships inside `NegativeNode`. 
- A token is blocked if one or more WMEs in the alpha memory satisfy the join constraints.
- The node maintains a `_blocked` table mapping tokens to their blocking WMEs, and a `_passed` list for tokens with zero blockers.
- If a new WME arrives and matches a passed token, that token is retracted from downstream nodes and moved to the blocked table.
- If a blocking WME is retracted, the node checks if the token has any remaining blockers. If its blockers count hits zero, the token is released downstream.

### 2.3 Deferred Working Memory Mutations
Rule actions (RHS) frequently assert, retract, or modify facts. If these operations propagate immediately during RHS execution, they can alter the Rete network state mid-execution, leading to recursive corruption.
We resolve this by introducing `RuleContext`, which queues mutations inside deferred lists:

```python
class RuleContext:
    def assert_fact(self, fact: Fact) -> None:
        self._pending_asserts.append(fact)
```

The `ReteEngine` runs the action, lets it return, and then replays the mutations in a safe, atomic sequence (retractions first, then assertions) to update the network cleanly.

---

## 3. Case Studies & Design Advice

We now examine six case studies showcasing the engine in action, highlighting how to write and structure rules.

### 3.1 Medical Diagnosis (Forward Chaining & Negation)
The medical diagnosis expert system (**source-code/rete-algorithm/example_medical.py**) implements standard clinical decision support.

```python
@engine.rule(
    Pat(Patient, name=Var("n"), temperature=Gt(38.5), symptoms=Contains("cough")),
    ~Pat(Diagnosis, patient=Var("n"), condition=Eq("flu")),
    salience=10
)
def diagnose_flu(ctx, n):
    ctx.assert_fact(Diagnosis(patient=n, condition="flu"))
```

#### Explanation & Design Advice
This case study uses forward chaining to derive high-level conclusions (`Diagnosis`) from raw measurements (`Patient`). 
- **Variable Joins**: The variable `Var("n")` binds the patient's name in the first pattern and enforces that the negative condition checks for a flu diagnosis for *that specific patient*.
- **NCE Guards**: The negative pattern `~Pat(...)` serves as a guard. It prevents the rule from firing repeatedly and generating redundant diagnoses. 
- **System Design Lesson**: Always use negated patterns to check for the existence of your inferred facts before asserting them, avoiding cycle bloat.

### 3.2 Smart Home Automation & Sensor Fusion
The smart home coordinator (**source-code/rete-algorithm/example_smart_home.py**) handles device automation based on occupancy, temperature, and light sensors.

```python
@engine.rule(
    Pat(Occupancy, room=Var("r"), status=Eq("occupied")),
    Pat(SensorReading, room=Var("r"), type=Eq("light"), value=Lt(100.0)),
    Pat(DeviceStatus, device_id=Var("d"), room=Var("r"), type=Eq("light"), state=Eq("off")),
    ~Pat(HouseMode, mode=Eq("away")),
)
def turn_on_lights(ctx, r, d):
    ctx.modify(d, state="on")
```

#### Explanation & Design Advice
This example highlights a critical rule system hazard: **infinite rule loops**.
In our first iteration, when the house switched to `"away"` mode, the `auto_shutoff_away` rule turned off all active devices. However, because the room was occupied and dark, the `turn_on_lights` rule immediately reactivated them, creating a continuous loop of turning devices on and off.
- **The Solution**: We added the negated condition `~Pat(HouseMode, mode=Eq("away"))` to block the automation rules when the house is vacant.
- **System Design Lesson**: When writing rules that perform mutations, always map out their interactions. If Rule A changes a state that Rule B reacts to, ensure they have explicit state guards (such as active modes or status flags) to establish a clear hierarchy.

### 3.3 Network Intrusion Detection & Threat Alerting
The network intrusion detection system (**source-code/rete-algorithm/example_network_security.py**) monitors connection attempts and log events.

```python
@engine.rule(
    Pat(LoginAttempt, ip=Var("ip"), result=Eq("fail")),
    Pat(FailedLoginCounter, ip=Var("ip"), count=Var("c")),
)
def increment_failed_login_counter(ctx, ip, c):
    for wme in ctx.token_wmes:
        if isinstance(wme.fact, FailedLoginCounter):
            ctx.modify(wme, count=c + 1)
        elif isinstance(wme.fact, LoginAttempt):
            ctx.retract(wme) # Consume the event
```

#### Explanation & Design Advice
Network systems must process a high frequency of raw incoming events (`LoginAttempt`). If we simply asserted every event into memory, the Rete network would suffer from a combinatorial explosion of joins.
- **Event Consumption Pattern**: Rather than retaining raw events, the rules consume them. When a login fails, we modify the persistent `FailedLoginCounter` and immediately **retract** the raw `LoginAttempt` event.
- **Stateful Aggregates**: High-level rules then reason about the consolidated counters (e.g., blocking the IP if `count > 3`).
- **System Design Lesson**: Categorize your facts into *events* (temporary, trigger rules, and then retracted) and *states* (persistent summaries updated by rules). This keeps working memory small and ensures the Rete network remains highly responsive.

### 3.4 E-Commerce Pricing & Discount Engine
The pricing system (**source-code/rete-algorithm/example_ecom_pricing.py**) calculates orders by combining loyalty tier discounts, bulk order price drops, and seasonal promotional campaigns.

```python
@engine.rule(
    Pat(CartSummary, customer_id=Var("c_id"), subtotal=Var("sub"), total_discount=Var("td")),
    ~Pat(CartItem, customer_id=Var("c_id"), processed=Eq(False)),
    ~Pat(AppliedDiscount, customer_id=Var("c_id"), processed=Eq(False)),
    ~Pat(Invoice, customer_id=Var("c_id")),
)
def create_invoice(ctx, c_id, sub, td):
    ctx.assert_fact(Invoice(customer_id=c_id, subtotal=sub, discount=td, total=sub-td))
```

#### Explanation & Design Advice
This case study demonstrates the **Barrier Completion** pattern. Generating an invoice must happen only *after* all cart items are totaled and all eligible discounts are fully applied.
- **NCE Barriers**: We declare negated patterns checking for any `CartItem` or `AppliedDiscount` that has `processed=False`. 
- **Execution Phases**: The Rete engine processes individual item accumulations incrementally. Only when the count of unprocessed items hits zero does the `create_invoice` rule satisfy its negative conditions and execute.
- **System Design Lesson**: Use negated attributes as synchronization barriers to structure multi-phase workflows in an otherwise asynchronous rule evaluation cycle.

### 3.5 Financial Portfolio Rebalancing
The portfolio advisor (**source-code/rete-algorithm/example_portfolio_rebalance.py**) monitors asset values and recommends buy/sell trades to align portfolios with target percentages.

```python
@engine.rule(
    Pat(PortfolioSummary, portfolio_id=Var("p_id"), total_value=Var("tot")),
    Pat(AssetAllocation, portfolio_id=Var("p_id"), asset_class=Var("ac"), current_value=Var("curr")),
    Pat(TargetAllocation, portfolio_id=Var("p_id"), asset_class=Var("ac"), target_pct=Var("t_pct")),
    ~Pat(AssetAllocation, portfolio_id=Var("p_id"), processed=Eq(False)),
    ~Pat(RebalanceTrade, portfolio_id=Var("p_id"), asset_class=Var("ac")),
)
def check_drift_and_rebalance(ctx, p_id, ac, curr, t_pct, tot):
    current_pct = curr / tot
    drift = current_pct - t_pct
    if abs(drift) > 0.05:
        ctx.assert_fact(RebalanceTrade(portfolio_id=p_id, asset_class=ac, action="sell" if drift > 0 else "buy", ...))
```

#### Explanation & Design Advice
This case study highlights the balance between LHS matching and RHS execution logic.
- **Coarse-Grained LHS**: We use the Rete LHS to match the relevant holdings, targets, and total portfolio sums.
- **Fine-Grained RHS Math**: Rather than trying to express complex mathematical functions (like absolute percentage deviations) directly in the declarative patterns, we evaluate them using Python code in the RHS action. If the threshold is crossed, we assert the trade recommendation.
- **System Design Lesson**: Keep Rete patterns focused on structural matching and variable binding. Offload complex mathematical equations and multi-variable threshold calculations to the RHS action to keep rules maintainable and performant.

### 3.6 Hospital Patient Triage & Resource Allocation
The hospital triage scheduler (**source-code/rete-algorithm/example_hospital_triage.py**) routes arriving patients to empty beds and available staff.

```python
@engine.rule(
    Pat(Patient, id=Var("p_id"), severity=Gt(3)),
    Pat(PatientStatus, patient_id=Var("p_id"), status=Eq("waiting")),
    Pat(Bed, id=Var("b_id"), occupied=Eq(False)),
    Pat(Staff, id=Var("s_id"), role=Eq("doctor"), busy=Eq(False)),
    salience=10,
)
def triage_critical_patient(ctx, p_id, b_id, s_id):
    ctx.assert_fact(Assignment(patient_id=p_id, bed_id=b_id, staff_id=s_id))
    # Mark resources as occupied/busy...
```

#### Explanation & Design Advice
This scheduler resolves a classic resource allocation problem where multiple resources must be matched dynamically under priority constraints.
- **Salience Priorities**: By setting `salience=10` on the critical triage rule and `salience=5` on the standard triage rule, the engine guarantees that severe cases are assigned first.
- **Dynamic Resource Releasing**: A lower salience rule (`salience=1`) processes treatment completion, retracting assignments and marking beds and staff as available. Once resources are released, the high-salience rules immediately reactivate on the remaining patient queue.
- **System Design Lesson**: Use salience systematically to establish scheduling priorities, and design resource-release cycles to allow continuous, incremental optimization of resource allocations.

---

## Wrap Up

### Why Use Embedded Expert Systems?
Even in an era dominated by large language models and deep neural networks, small, embedded expert systems using algorithms like Rete remain highly relevant for specific design constraints:
1. **Verifiability and Safety**: Expert systems are fully deterministic. Unlike LLMs, they do not suffer from "hallucinations," and they provide an explicit execution audit trail. This makes them ideal for mission-critical tasks like medical triage, security filtering, or automated financial trading where logic must be mathematically verifiable.
2. **Speed and Efficiency**: Rete compiles rule logic into an optimized, in-memory dataflow network. A local Rete evaluation cycle executes in fractions of a millisecond with zero API latency, zero token costs, and negligible compute overhead, making it perfectly suited for real-time edge environments (like home automation and IoT devices).
3. **Decoupled Business Logic**: Expert systems allow developers to write rules in a highly declarative format. Business logic is separated from database routing or application control flow, allowing rules to be modified, extended, or audited independently.

### When to Use Other Techniques
While expert systems are powerful, they are not a silver bullet in the modern age of strong AI implemented with LLMs. You should consider alternative approaches when dealing with other problem shapes:
1. **Deep Learning and Machine Learning**: For perception tasks (image classification, audio recognition) or statistical prediction (forecasting churn, estimating house values) where logic cannot be easily written as binary rules, machine learning is essential. Neural networks excel at mapping messy, high-dimensional inputs to predictions.
2. **Large Language Models (LLMs)**: When dealing with unstructured natural language, semantic reasoning, or open-ended text synthesis (like draft generation or user support), LLMs are the superior choice.
3. **Constraint Satisfaction Solvers (e.g., MiniZinc)**: When a scheduling or layout problem is highly constrained and requires search optimization over a huge space of combinations (like creating a school timetable or routing delivery trucks), a dedicated Constraint Satisfaction solver is far more efficient than writing a network of heuristic forward-chaining rules.
4. **Graph Databases and Semantic Web (SPARQL/RDF)**: If your domain model requires querying vast, richly interconnected networks of relationship data rather than running active production loops, graph databases are much cleaner and more performant than loading millions of facts into a Rete engine.

The Rete Algorithm scales poorly (order of N^2 where N is working memory size) for large working memories. The Rete Algorithm is optimized for large numbers of rules. In the modern age of LLMs, you can use coding agents help write and maintain large rule sets.
