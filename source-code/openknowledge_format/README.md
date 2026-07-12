# Open Knowledge Format (OKF) Bundle Explorer

This directory contains an example implementation of an **Open Knowledge Format (OKF)** "consumption agent" and a sample knowledge bundle. 

OKF is a lightweight, human- and agent-friendly convention for representing knowledge—the metadata, context, and curated insights surrounding data and systems. Rather than defining a rigid schema or registry, OKF organizes concepts as markdown files with YAML frontmatter inside a git-compatible folder structure.

This demo demonstrates how to parse a local OKF bundle, index it, and query it using a local LLM via Ollama.

---

## References & Inspiration

This implementation is based on the concepts and drafts defined by Google:
* **Concept Blog Post:** [How the Open Knowledge Format can improve data sharing](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing/)
* **Format Specification:** [Google Cloud Platform Knowledge Catalog - OKF Specification](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)

*Note: This is a clean-room implementation of the OKF conceptual specification and does not use Google's proprietary libraries.*

---

## Knowledge Bundle Structure

The sample directory `/bundle` mimics a standard OKF bundle for a retail analytics team:

```text
bundle/
├── index.md                      # Bundle root index / table of contents
├── tables/
│   ├── sales_events.md          # Raw transaction data catalog
│   ├── products.md              # Canonical product dimension (SCD Type 2)
│   └── customers.md             # Privacy-safe customer dimension
├── metrics/
│   ├── daily_revenue.md         # Revenue metric formula and grain
│   └── customer_ltv.md          # LTV predictive model definition
└── playbooks/
    └── revenue_drop_investigation.md # Operational guide for incident management
```

Each markdown file starts with a standard YAML frontmatter block that defines metadata attributes such as `type`, `title`, `description`, `resource` URI, and `tags`:

```yaml
---
type: Metric
title: Daily Revenue
description: Total net revenue aggregated per calendar day per store.
resource: bigquery://retail-analytics/metrics/daily_revenue
tags: [revenue, daily, finance, KPI]
timestamp: 2026-06-10T00:00:00Z
---
```

---

## Project Structure

* [okf_explorer.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/openknowledge_format/okf_explorer.py): The main Python explorer script. It parses the Markdown files, extracts YAML frontmatter, loads concepts into memory, indexes them, and queries the local LLM.
* [pyproject.toml](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/openknowledge_format/pyproject.toml): Configures the Python project dependencies, strictly relying on `ollama`.
* [bundle/index.md](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/openknowledge_format/bundle/index.md): The main landing index for the OKF bundle catalog.

---

## Setup & Running the Code

This project utilizes the `uv` tool for fast, reliable Python environment management.

### Prerequisites

1. Install **Ollama** and pull the local Gemma model:
   ```bash
   ollama pull gemma4:e2b-it-qat
   ```

2. Ensure `uv` is installed on your system.

### Running the Explorer

1. Navigate to this directory in your terminal:
   ```bash
   cd /Users/markwatson/GITHUB/PythonAIBook/source-code/openknowledge_format
   ```

2. Sync the dependencies and run the program:
   ```bash
   uv run okf_explorer.py
   ```

---

## How It Works

1. **Loader & Parsing:** The `KnowledgeBundle` class walks the `bundle/` directory recursively. It filters out reserved files (`index.md` and `log.md`) and loads each markdown file. It extracts the YAML block at the top and populates a Python `Concept` object.
2. **Search Indexing:** A simple, lightweight in-memory keyword search is built directly in Python to match queries against concept titles, descriptions, and markdown body copy.
3. **Retrieval-Augmented Q&A:** When a user asks a question, the `OKFAgent` gathers the top relevant markdown concept files, constructs a structured context, and prompts the local `gemma4:e2b-it-qat` model to answer the query referencing exact OKF concepts (such as citing `tables/sales_events`).

## Development workflow

Uses [`uv`](https://docs.astral.sh/uv/) for dependency management and [`just`](https://just.systems/) as the task runner. Install both, then:

```bash
uv sync
just check       # fmt-check + lint + typecheck + test
just fmt         # ruff format
just lint        # ruff --fix
just typecheck   # pyrefly (strict)
just test        # pytest with testmon (fast)
just test-all    # full parallel pytest run
```

Under Claude Code, `.claude/hooks/py-check.sh` runs after every edit (format + lint + per-file typecheck) and `.claude/hooks/py-stop.sh` runs the full gate before the turn ends. See `CLAUDE.md` for the workflow contract.
