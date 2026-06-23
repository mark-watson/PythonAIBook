# Open Knowledge Format (OKF) for Human-Agent Systems

As artificial intelligence systems and autonomous agents play an increasingly central role in data analytics, a new challenge has emerged: how do we share context, curated metadata, and operational playbooks between humans and AI systems? Historically, metadata has been stored in specialized database catalog systems (like Apache Atlas, Google Cloud Dataplex, or proprietary enterprise wikis) that are difficult for agents to access without custom integrations, or formatted as raw JSON/YAML payloads that are dry and hard for humans to collaborate on.

Dear reader, in this chapter we explore the **Open Knowledge Format (OKF)**—a minimal, lightweight, human- and agent-friendly convention for representing knowledge surrounding data systems. Originally proposed in draft specs by Google Cloud Platform, OKF takes the position that knowledge should be readable, writeable, diffable (i.e., think output like `git diff`), and portable using nothing more than markdown files, YAML frontmatter, and standard version control (such as Git). 

In the preface to this chapter, we focus on creating a simple Python implementation of a system using OKF and  in the final **Wrap Up** section, we will catalog the diverse ways you can use this example code for building knowledge bases designed to serve both humans and AI systems, alongside practical project ideas for you to build upon.

The examples for this chapter are located in the directory **source-code/openknowledge_format**.

## References & Inspiration

This implementation is based on the concepts and drafts defined by Google:
* **Concept Blog Post:** [How the Open Knowledge Format can improve data sharing](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing/)
* **Format Specification:** [Google Cloud Platform Knowledge Catalog - OKF Specification](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)

*Note: This is a clean-room implementation of the OKF conceptual specification and does not use Google's proprietary libraries.*

## What is Open Knowledge Format (OKF)?

According to the open specification draft, the format is intentionally minimal:
* **Human-Readable:** It consists of standard Markdown documents that can be read directly in a terminal, text editor, or rendered in a browser.
* **Agent-Parseable:** The top of each document contains a YAML frontmatter block that standardizes metadata attributes (such as the asset type, canonical URI, and classification tags).
* **Diffable:** Because it consists of plain-text markdown, changes can be reviewed, tracked, and merged using standard version control like Git.
* **Portable:** There is no schema registry, database, or SDK dependency. If you can `git clone` or zip a directory, you can share an OKF knowledge bundle.

An OKF repository is organized into a **Knowledge Bundle**—a self-contained, hierarchical collection of knowledge documents representing *concepts*. A concept can describe anything: a database table, an API endpoint, a financial KPI metric, or an operational playbook.

## Sample Knowledge Bundle Structure

Let's look at the `/bundle` directory structure for our sample retail analytics system:

```text
bundle/
├── index.md                      # Bundle root listing/TOC
├── tables/
│   ├── sales_events.md          # Database Table: point-of-sale stream
│   ├── products.md              # Database Table: product dimension
│   └── customers.md             # Database Table: anonymized customers
├── metrics/
│   ├── daily_revenue.md         # Metric: formula and grain
│   └── customer_ltv.md          # Metric: predictive LTV model
└── playbooks/
    └── revenue_drop_investigation.md # Playbook: step-by-step incident response
```

### An OKF Concept Document Example

Here is the markdown content of `bundle/metrics/daily_revenue.md`. Notice the metadata block (YAML frontmatter) separated by `---` lines:

```markdown
---
type: Metric
title: Daily Revenue
description: Total net revenue (after discounts, excluding returns) aggregated per calendar day per store.
resource: bigquery://retail-analytics/metrics/daily_revenue
tags: [revenue, daily, finance, KPI]
timestamp: 2026-06-10T00:00:00Z
owner: finance-analytics@example.com
sla: available by 03:00 UTC each morning
---

# Daily Revenue

**Daily Revenue** is the primary top-line financial KPI for the retail platform.
It answers the question: *"How much did we sell today?"*

## Definition

```sql
daily_revenue =
  SUM(
    sales_events.quantity
    * sales_events.unit_price_usd
    * (1 - sales_events.discount_pct / 100)
  )
WHERE
  sales_events.quantity > 0          -- exclude returns
  GROUP BY DATE(sales_events.event_ts), sales_events.store_id

## Grain

One row per `(date, store_id)`.

## Source Tables

* [tables/sales_events](../tables/sales_events.md) — primary fact source

## Important Caveats

* Returns (negative `quantity`) are **excluded**.
* If `daily_revenue` drops sharply, check the [revenue drop investigation playbook](../playbooks/revenue_drop_investigation.md).
```


## Python Architecture: The OKF Explorer

To query and browse this knowledge bundle, we wrote a Python program that acts as a **consumption agent**. It reads the folder tree, parses frontmatter metadata and markdown bodies, builds a local keyword index, and feeds relevant knowledge concepts to a local Ollama model to answer natural language questions.

We define the dependencies in `pyproject.toml` using `uv`:

```toml
[project]
name = "openknowledge-format"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "ollama>=0.6.1",
]
```

### The OKF Parser and Loader

Below is the code in `okf_explorer.py` that recursively traverses the bundle, parses the YAML blocks, and structures them into `Concept` objects:

```python
import re
from pathlib import Path
from dataclasses import dataclass, field

RESERVED_FILENAMES = {"index.md", "log.md"}

@dataclass
class Concept:
    concept_id: str          # e.g., "tables/sales_events"
    path: Path               # absolute filesystem path
    frontmatter: dict        # parsed metadata dictionary
    body: str                # raw markdown content

    @property
    def type(self) -> str:
        return self.frontmatter.get("type", "Unknown")

    @property
    def title(self) -> str:
        return self.frontmatter.get("title", self.concept_id)

    @property
    def description(self) -> str:
        return self.frontmatter.get("description", "")

    @property
    def tags(self) -> list[str]:
        return self.frontmatter.get("tags", [])

    def as_context_block(self) -> str:
        """Serializes the concept into a format optimized for LLM context."""
        lines = [
            f"## Concept: {self.title}",
            f"**ID**: {self.concept_id}",
            f"**Type**: {self.type}",
            f"**Description**: {self.description}",
        ]
        if self.tags:
            lines.append(f"**Tags**: {', '.join(self.tags)}")
        lines.append("")
        lines.append(self.body.strip())
        return "\n".join(lines)


def _parse_simple_yaml(yaml_text: str) -> dict:
    """Minimal inline YAML parser to extract frontmatter keys and arrays."""
    result = {}
    for line in yaml_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        # Handle inline lists: [tag1, tag2]
        if rest.startswith("[") and rest.endswith("]"):
            items = [i.strip().strip('"').strip("'") for i in rest[1:-1].split(",")]
            result[key] = [i for i in items if i]
        else:
            result[key] = rest.strip('"').strip("'")
    return result


def parse_concept(path: Path, bundle_root: Path) -> Concept | None:
    """Parses an individual OKF document, separating frontmatter and body."""
    if path.name in RESERVED_FILENAMES:
        return None

    text = path.read_text(encoding="utf-8")
    frontmatter = {}
    body = text
    
    # Locate frontmatter blocks bounded by --- lines
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if fm_match:
        frontmatter = _parse_simple_yaml(fm_match.group(1))
        body = text[fm_match.end():]

    concept_id = str(path.relative_to(bundle_root).with_suffix(""))
    concept_id = concept_id.replace("\\", "/")  # Normalize Windows paths
    
    return Concept(concept_id, path, frontmatter, body)
```

We load the complete bundle into a `KnowledgeBundle` class, which offers simple, database-free search helpers:

```python
@dataclass
class KnowledgeBundle:
    root: Path
    concepts: list[Concept] = field(default_factory=list)

    @classmethod
    def load(cls, root: Path) -> "KnowledgeBundle":
        bundle = cls(root=root)
        for md_path in sorted(root.rglob("*.md")):
            concept = parse_concept(md_path, root)
            if concept:
                bundle.concepts.append(concept)
        return bundle

    def search(self, query: str) -> list[Concept]:
        """Rank concepts by keyword match frequency inside title, desc, and body."""
        keywords = [w.lower() for w in query.split() if len(w) > 2]
        scored = []
        for concept in self.concepts:
            haystack = (concept.title + " " + concept.description + " " + concept.body).lower()
            score = sum(haystack.count(kw) for kw in keywords)
            if score > 0:
                scored.append((score, concept))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]
```

Our method for search is a simple bag of matching words approach. For a production system I would use a tool like zvec as I did in the example in my [`Ollama in Action` book in chapter “RAG Using zvec Vector Datastore and Local Model”](https://leanpub.com/read/ollama/rag-using-zvec-vector-datastore-and-local-model) (link to read online).

### The LLM Consumption Agent

To build a consumption agent, we wrap the search catalog and hook it up to the local Ollama service. We prompt the model to restrict its answers to the contexts provided, requiring it to cite the concept ID:

```python
import ollama

class OKFAgent:
    SYSTEM_PROMPT = """\
You are a data knowledge assistant. You have been given excerpts from
an Open Knowledge Format (OKF) knowledge bundle describing data tables,
metrics, and operational playbooks.

Answer the user's question using ONLY the provided knowledge context.
Be concise, accurate, and cite the concept ID (e.g., tables/sales_events)
when referring to specific assets. If the context does not contain the
information, state that clearly instead of guessing.
"""

    def __init__(self, bundle: KnowledgeBundle, model: str = "gemma4:e2b-it-qat"):
        self.bundle = bundle
        self.model = model

    def _build_context(self, query: str, top_k: int = 3) -> str:
        relevant = self.bundle.search(query)[:top_k]
        if not relevant:
            relevant = self.bundle.concepts[:top_k]
        return "\n\n---\n\n".join(c.as_context_block() for c in relevant)

    def ask(self, question: str) -> str:
        context = self._build_context(question)
        user_message = f"## Knowledge Context\n\n{context}\n\n---\n\n## Question\n\n{question}"

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )
        return response.message.content
```

Dear reader, this example code is meant to get you started: `hack away` or `vibe code` your own applications.

## Example Output

Here we round the example program that contains several test queries:

```
 $ uv run okf_explorer.py 

======================================================================
  Loading OKF Knowledge Bundle
======================================================================
Bundle root : /Users/markwatson/GITHUB/PythonAIBook/source-code/openknowledge_format/bundle
Contents    : 6 concepts (3× Database Table, 2× Metric, 1× Playbook)

======================================================================
  All Concepts in Bundle
======================================================================
  [Metric            ]  metrics/customer_ltv
                        Predicted total net revenue a customer will generate over their entire r
  [Metric            ]  metrics/daily_revenue
                        Total net revenue (after discounts, excluding returns) aggregated per ca
  [Playbook          ]  playbooks/revenue_drop_investigation
                        Step-by-step guide for on-call analysts to diagnose an unexpected drop i
  [Database Table    ]  tables/customers
                        Anonymized customer dimension with loyalty tier and demographic segment 
  [Database Table    ]  tables/products
                        Product catalog containing SKU-level attributes, category hierarchy, and
  [Database Table    ]  tables/sales_events
                        Raw point-of-sale event stream capturing every transaction at the regist

======================================================================
  Search: 'revenue'
======================================================================
  metrics/daily_revenue  —  Total net revenue (after discounts, excluding returns) aggre
  playbooks/revenue_drop_investigation  —  Step-by-step guide for on-call analysts to diagnose an unexp
  tables/sales_events  —  Raw point-of-sale event stream capturing every transaction a

======================================================================
  Filter by type: 'Metric'
======================================================================
  metrics/customer_ltv  —  Customer Lifetime Value (LTV)
  metrics/daily_revenue  —  Daily Revenue

======================================================================
  Filter by tag: 'KPI'
======================================================================
  metrics/customer_ltv  —  Customer Lifetime Value (LTV)
  metrics/daily_revenue  —  Daily Revenue

======================================================================
  LLM Q&A  (model: gemma4:e2b-it-qat)
======================================================================

Q1: How is daily revenue calculated and what tables does it use?
------------------------------------------------------------
Daily revenue is calculated as follows:

$$
\text{daily\_revenue} = \sum (\text{sales\_events.quantity} \times
    \text{sales\_events.unit\_price\_usd} \times (1 -
    \frac{\text{sales\_events.discount\_pct}}{100}))
$$

The calculation excludes returns by filtering for
    $\text{sales\_events.quantity} > 0$. The result is aggregated per
    calendar day and per store ID
    ($\text{DATE(sales\_events.event\_ts)},
    \text{sales\_events.store\_id}$).

**Source Tables:**
The primary source table used for this calculation is
    **[tables/sales_events](../tables/sales_events.md)**.

Q2: What should I do if daily revenue drops suddenly?
------------------------------------------------------------
If daily revenue drops suddenly, you should follow the **Revenue Drop
    Investigation Playbook** (#playbooks/revenue_drop_investigation).
    This investigation is triggered when the
    [daily_revenue](../metrics/daily_revenue.md) metric shows a
    significant unexpected decline (> 10 % day-over-day or > 2 σ below
    the 30-day rolling average).

### Step 1 — Confirm the drop is real (not a data issue)
1. Check the pipeline run log to ensure the nightly aggregation
    completed successfully.
2. Verify row counts in [sales_events](../tables/sales_events.md) for
    the affected day. A count near zero typically indicates a pipeline
    or CDC failure rather than a business drop.

### Step 2 — Isolate by dimension
Run the revenue query broken down by:
*   **Store**: To see if the drop is isolated to one location.
*   **Category**: To check if a specific product category is affected.
*   **Payment method**: To identify shifts in payment mix caused by
    processor issues.
*   **Hour of day**: To detect mid-day system outage windows.

### Step 3 — Check for external signals
1. Review the incident board for ongoing POS system outages.
2. Check weather and calendar (holidays or local events) for affected
    stores.
3. Consult the merchandising team to see if planned promotions have
    ended.

### Step 4 — Escalate if needed
If the root cause is not identified within 30 minutes, escalate based on
    the symptom:

*   **Pipeline / data quality**: Page `#data-engineering-oncall`.
*   **POS system outage**: Page `#it-ops-oncall`.
*   **Payment processor**: Page `#payments-oncall`.
*   **Genuine business drop**: Notify the `VP of Retail` via the
    standard business escalation path.

Q3: What percentage of sales events have a customer ID? And what does that tell us about LTV calculations?
------------------------------------------------------------
Based on the provided documentation for **Sales Events**:

1.  **Percentage of Sales Events with a Customer ID:**
    Roughly 38% of transactions have a NULL `customer_id`, which
    represents anonymous cash sales. This means only approximately 62%
    of rows in the `sales_events` table contain a non-NULL identifier
    for a specific customer.

2.  **Impact on LTV Calculations:**
    The **Customer Lifetime Value (LTV)** metric is computed using
    historical purchase sequences from the `sales_events` table via
    machine learning models (BG/NBD and Gamma-Gamma) that rely on the
    `customer_id`. This allows for a forward-looking estimate of the
    total net revenue a customer will generate, linking transactional
    data to specific customers.

Q4: How do I join sales_events to products correctly for historical reports?
------------------------------------------------------------
To join `sales_events` to `products` for historical reports, you must
    join on `product_id` and filter using a condition that matches the
    event timestamp against the product's validity period:

*   Join `sales_events` and `products` on `product_id`.
*   Filter by: `valid_from <= event_ts::date AND (valid_to IS NULL OR
    valid_to >= event_ts::date)` to retrieve the correct historical
    version of the product.
```

## Wrap Up

The Open Knowledge Format represents a pragmatic bridge between two paradigms: traditional, human-focused documentation, and structural, API-centric schema specifications. Because it defaults to raw text markdown in Git, engineers do not have to leave their development workflows to document their systems, and automated pipelines can generate or read metadata files directly.

### Catalog of Uses for OKF Bundles

1. **AI Copilots & RAG Context:** By formatting internal documentation in OKF, agents can easily parse metadata and trace lineage relationships using standard folder parsing. Because each concept is small, it fits comfortably into LLM context windows, reducing token consumption and hallucination rates.
2. **Interactive Developer Wikis:** Static site generators (like Hugo, Docusaurus, or MkDocs) can read the `/bundle` directory directly to render human-navigable wikis. Humans read the markdown in a portal, while consumption agents process the exact same files programmatically.
3. **Data Lineage and Operational Audits:** By adding timestamps, ownership keys, and pipeline run logs to the frontmatter of table and metric documents, operations teams can quickly trace data dependencies during outages.
4. **On-call Automation:** Incident systems can search the bundle for playbooks tagged with relevant metrics (e.g. searching for `daily_revenue` matches `playbooks/revenue_drop_investigation.md`) and automatically attach diagnostic instructions to incoming pager alerts.

### Reader Projects and Exercises

* **Project 1: Automatic OKF Generators.** Write a Python pipeline script that inspects a PostgreSQL or BigQuery schema, extracts the column names and comments, and automatically generates or updates the frontmatter and schema tables in `bundle/tables/<table_name>.md`.
* **Project 2: Vector Search for OKF Bundles.** Replace the simple substring-matching index in `KnowledgeBundle.search()` with a vector database. Write a script to generate text embeddings for each concept using an Ollama embedding model (like `nomic-embed-text`) and perform semantic retrieval instead of keyword search.
* **Project 3: Git Hook Validator.** Build a pre-commit Git hook that validates OKF bundles. The hook should check that every markdown file contains valid YAML frontmatter, contains the required `type` field, and verify that any links to other concepts (`[text](../path/to/concept.md)`) represent actual files existing in the bundle.

## Optional Practice Problems

Here are some optional practice problems to help you master the concepts covered in this chapter:

### 1. Warm-Up: Robust Frontmatter Parser (Easy)
The custom parser `_parse_simple_yaml` in `okf_explorer.py` is lightweight and requires no external libraries, but it cannot handle nested YAML dictionaries or multiline values.
* **Task**: Enhance the parser in one of the following ways:
  1. Integrate the standard library `tomllib` or implement a more robust parser using regular expressions to support multi-line values (like descriptions spanning multiple lines).
  2. Implement automatic type coercion. For example, if a key is `timestamp`, parse the value into a Python `datetime` object. If a key is `sla`, strip whitespace and standardize its casing.
  3. Write a small test suite in Python to verify your parser correctly handles edge cases, such as values that contain colons (e.g. `sla: available by 03:00 UTC`).

### 2. OKF Link Integrity Checker (Medium)
In an OKF bundle, documentation files link to each other using relative markdown links (e.g., `[sales_events](../tables/sales_events.md)`). If a file name is changed or deleted, these references break.
* **Task**: Write a validation utility in `okf_explorer.py` that checks the link integrity of the entire bundle:
  1. Extend `KnowledgeBundle` to parse the markdown body of each concept and locate all Markdown-style links: `[link text](relative_path)`.
  2. Resolve the `relative_path` relative to the current concept's filesystem location.
  3. Check if the file at the resolved path exists on disk.
  4. Collect all broken links and output them as a structured report (showing the source file, line number, and broken path).

### 3. Graph-Augmented RAG Retrieval (Hard)
The standard `OKFAgent._build_context` retrieves the top-scoring concepts independently based on keywords. However, data concepts are highly relational: a metric like `daily_revenue` references `tables/sales_events` in its definition, which in turn points to `playbooks/revenue_drop_investigation`.
* **Task**: Enhance the retrieval agent to support graph-based context traversal:
  1. Build a dependency graph where each node is a `Concept` and directed edges represent references (parsed from Markdown links or specified in frontmatter metadata).
  2. When a user asks a question, retrieve the initial `top_k` concepts using keyword search.
  3. Automatically traverse the graph to retrieve all directly connected neighbors (1-hop relation) of these concepts.
  4. Combine the initial concepts and their neighbors, de-duplicate them, and construct the prompt context.
  5. Test this implementation by asking: *"What playbook should I run if the main metric depending on sales_events fails?"* Verify that the agent successfully pulls the `revenue_drop_investigation` playbook even if the playbook itself doesn't contain the keyword "sales_events".

