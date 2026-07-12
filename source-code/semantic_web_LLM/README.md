# LLM_semweb — Semantic Web QA with SPARQL + DBpedia + Fireworks.ai

A command-line tool that answers natural-language questions by combining
large-language-model entity extraction with live SPARQL queries against the
DBpedia knowledge base.

## How it works

1. **Entity extraction** — The question is sent to a Fireworks.ai LLM
   (`deepseek-v4-flash`) with a one-shot prompt that classifies every named
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

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- A Fireworks.ai API key ([get one here](https://fireworks.ai))

### Install

```bash
cd source-code/semantic_wem_LLM
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
uv run LLM_semweb.py
```

You'll be prompted to enter a question:

```
LLM_semweb.py - QA with SPARQL + DBpedia
--------------------------------------------------
Enter your question: What is the capital of France
```

### Command-line mode

Pass the question as arguments:

```bash
uv run LLM_semweb.py "What is the capital of France"
```

### Multi-turn chat

The `chat_with_context` function provides an interactive multi-turn session
where DBpedia context is injected into each turn. Call it from a Python
shell:

```python
from LLM_semweb import chat_with_context
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
uv run LLM_semweb.py "California, Texas, IBM, Microsoft, Germany, Canada"
```

```bash
uv run LLM_semweb.py "IBM, Pepsi, Canada"
```

```bash
uv run LLM_semweb.py "Germany, Canada, Pepsi, IBM, California, Biology, Physics"
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
semantic_wem_LLM/
├── LLM_semweb.py     # Main application
├── pyproject.toml    # Project metadata + dependencies (openai, SPARQLWrapper)
├── uv.lock           # Lock file
└── README.md         # This file
```

## Troubleshooting

**`FIREWORKS_API_KEY` not set** — The script silently passes `None` to the
OpenAI client, which will raise an authentication error on the first LLM
call. Run `export FIREWORKS_API_KEY="..."` and try again.

**SPARQL queries return `[]`** — DBpedia's live endpoint occasionally changes
which predicates are available. The query templates already try
`dbo:description`, `rdfs:comment`, and `dbo:abstract` in parallel. If all
three are down, wait and retry.

**Entity not found** — The LLM may split or rephrase an entity name that
doesn't exactly match the DBpedia `rdfs:label`. Try the canonical name
(e.g. `"United States"` instead of `"America"`).
