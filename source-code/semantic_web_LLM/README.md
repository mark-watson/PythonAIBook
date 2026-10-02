# LLM_semweb — Semantic Web QA with SPARQL + DBpedia + Fireworks.ai

A command-line tool that answers natural-language questions by combining
large-language-model entity extraction with live SPARQL queries against the
DBpedia knowledge base. The LLM calls go through
[`litellm`](https://github.com/BerriAI/litellm), the book's uniform interface
(`fireworks_ai/...` model strings).

## How it works

1. **Entity extraction** — The question is sent to a Fireworks.ai LLM
   (`deepseek-v4p1-flash`) with a one-shot prompt that classifies every named
   entity into one of four types: `PERSON`, `ORG`, `GPE`, or `MISC`.
2. **Relationship detection** — The question is scanned against a hash table
   of ~20 common English phrases (`"capital"`, `"born"`, `"married to"`,
   `"founded"`, `"headquarters"`, `"population"`, …) that map to DBpedia
   ontology property URIs. When a match is found, a targeted SPARQL
   relationship query is executed first.
3. **DBpedia lookup** — For each entity, a SPARQL query retrieves the
   `dbo:description` / `rdfs:comment` / `dbo:abstract` (all three are tried,
   since the live endpoint has changed predicates over time).
4. **LLM answer** — Retrieved DBpedia context is fed back to the LLM to
   produce a natural-language answer.

## Setup

### Prerequisites

- Python ≥ 3.14
- [uv](https://docs.astral.sh/uv/) package manager
- A Fireworks.ai API key ([get one here](https://fireworks.ai))

### Install

```bash
cd source-code/semantic_web_LLM
uv sync          # creates .venv and installs dependencies
```

### Configure

```bash
export FIREWORKS_API_KEY="your-api-key"
```

Add this to your shell profile (`~/.zshrc`, `~/.bashrc`, …) to make it
persistent.

## Running

### Interactive mode

```bash
uv run DBPedia.py
```

You'll be prompted to enter a question:

```
DBPedia.py - QA with SPARQL + LLM
--------------------------------------------------
Enter your question: What is the capital of France
```

### Command-line mode

Pass the question as arguments:

```bash
uv run DBPedia.py "What is the capital of France"
uv run Wikidata.py "What is the capital of France"
uv run DBPedia_and_Wikidata.py "What is the capital of France"
```

### Multi-turn chat

The `chat_with_context` function in `DBPedia.py` provides an interactive
multi-turn session where DBpedia context is injected into each turn. Call it
from a Python shell:

```python
from DBPedia import chat_with_context

chat_with_context()
```

Type `quit`, `exit`, or `bye` to end the session.

## Example queries

### Relationship queries

These match against the built-in property hash table and return specific
related entities from DBpedia:

| Query | What it returns |
|---|---|
| `What is the capital of France` | Paris |
| `What is the capital of Germany` | Berlin |
| `Where was Bill Gates born` | Seattle |
| `Who is Bill Gates married to` | Melinda French Gates |
| `Who founded IBM` | Charles Ranlett Flint, George Winthrop Fairchild, Herman Hollerith |
| `What is the population of Paris` | Population figure |
| `What currency does Germany use` | Euro |
| `What language do they speak in Canada` | English / French |
| `Where is IBM headquartered` | Armonk, New York |
| `What industry is Microsoft in` | Information technology |

### Entity description queries

Passing a list of entity names returns DBpedia descriptions for each:

```bash
uv run DBPedia.py "California, Texas, IBM, Microsoft, Germany, Canada"
```

```bash
uv run DBPedia.py "IBM, Pepsi, Canada"
```

```bash
uv run DBPedia.py "Germany, Canada, Pepsi, IBM, California, Biology, Physics"
```

## Supported relationship keywords

The following English words/phrases are recognized and mapped to DBpedia
properties (matching is case-insensitive, longest phrase first):

| Keyword(s) | DBpedia property |
|---|---|
| `capital`, `capital of` | `dbo:capital` |
| `birthplace`, `born`, `born in` | `dbo:birthPlace` |
| `deathplace`, `died`, `died in` | `dbo:deathPlace` |
| `spouse`, `married to` | `dbo:spouse` |
| `founded`, `founded by`, `founder`, `who founded` | `dbo:foundedBy` |
| `industry` | `dbo:industry` |
| `location`, `headquartered`, `headquarters` | `dbo:locationCity` |
| `country` | `dbo:country` |
| `population` | `dbo:populationTotal` |
| `leader`, `president`, `prime minister` | `dbo:leaderName` |
| `currency` | `dbo:currency` |
| `area` | `dbo:areaTotal` |
| `language`, `official language` | `dbo:language` |

## Project layout

```
semantic_web_LLM/
├── library.py                 # shared LLM + SPARQL utilities
├── DBPedia.py                 # QA over DBpedia
├── Wikidata.py                # QA over Wikidata
├── DBPedia_and_Wikidata.py    # federated QA across both
├── pyproject.toml             # project metadata + dependencies (litellm, requests)
├── uv.lock                    # lock file
└── README.md                  # this file
```

## Troubleshooting

**`FIREWORKS_API_KEY` not set** — litellm fails the first LLM call with an
error naming the missing `FIREWORKS_API_KEY` variable (`APIConnectionError:
FIREWORKS_API_KEY is not set`). Run `export FIREWORKS_API_KEY="..."` and try
again.

**SPARQL queries return `[]`** — DBpedia's live endpoint occasionally changes
which predicates are available. The query templates already try
`dbo:description`, `rdfs:comment`, and `dbo:abstract` in parallel. If all
three are down, wait and retry.

**Entity not found** — The LLM may split or rephrase an entity name that
doesn't exactly match the DBpedia `rdfs:label`. Try the canonical name
(e.g. `"United States"` instead of `"America"`).

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
