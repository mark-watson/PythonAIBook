# Semantic Web Question Answering with SPARQL and Large Language Models

The Resource Description Framework (RDF) is the data model that underpins the Semantic Web. In RDF, every fact is expressed as a **triple** of subject, predicate, and object, where the subject and object are nodes in a graph and the predicate is a typed, named edge. Because every entity and every relationship is identified by a globally unique URI, data published by different organizations can be merged into a single shared graph without ambiguity. This graph-based foundation is what distinguishes RDF from ordinary tables or JSON documents: relationships are first-class citizens, and any two datasets that share a URI can be joined seamlessly.

One of the great advantages of RDF is that several large knowledge bases are published as public SPARQL endpoints, meaning anyone with an HTTP client can query them free of charge and without registration. **DBpedia** exposes `http://dbpedia.org/sparql`, and **Wikidata** exposes `https://query.wikidata.org/sparql`; both return results in the same standardized JSON format, so the same client code can talk to either endpoint. This means that a developer can, in a handful of lines of Python, ask structured questions about millions of people, places, organizations, and concepts curated by the Wikipedia and Wikimedia communities with no API keys, no licensing fees, and no local data storage required.

Public SPARQL endpoints also compose naturally. Because DBpedia and Wikidata both speak SPARQL and both model their data as RDF, a program can query both endpoints for the same entities and merge the results into a single context for downstream processing. Each source contributes different strengths: DBpedia carries richer infobox-derived product lists and financial figures, while Wikidata provides more comprehensive subsidiary and leadership relationships, so combining them yields a fuller picture than either alone. The remainder of this chapter shows exactly how to build such a system, pairing these public endpoints with a large language model that turns structured triples into natural-language answers.

## A Live Example to Motivate the Reader

Before we dig into the example code implementation, let us see what the finished programs actually do. We have three scripts that all answer the same question but draw on different knowledge bases.

### DBpedia

```bash
uv run DBPedia.py "tell me about the following: IBM, Microsoft"
```

```
Answer:
Based on the provided knowledge, here is a detailed overview of IBM
and Microsoft.

IBM (International Business Machines) is an American multinational
technology corporation and a public company in the information
technology industry. It was founded on June 16, 1911, by Charles
Ranlett Flint, George Winthrop Fairchild, and Herman Hollerith. Key
persons include Arvind Krishna and Gary Cohn. IBM's products span
Artificial intelligence, Cloud computing, Computer hardware, Quantum
computing, Robotics, Software, Automation, and Blockchain. Its
services include Professional services, Outsourcing, and Managed
services. Financial figures: revenue of approximately $62.73 billion,
net income of $6.023 billion, and 270,300 employees.

Microsoft is an American multinational technology corporation and a
public company in the information technology industry, headquartered
in Redmond, Washington. It was founded on April 4, 1975, by Bill
Gates and Paul Allen. Microsoft's products include Cloud computing,
Computer hardware, Consumer electronics, and Software development.
Its services include GitHub, LinkedIn, Microsoft 365, Microsoft Azure,
Microsoft Bing, Microsoft Edge, Microsoft Outlook, OneDrive, Xbox Game
Pass, and Xbox network. Subsidiaries include GitHub and LinkedIn.
Financial figures: net income of approximately $101.8 billion, and
228,000 employees.
```

### Wikidata

```bash
uv run Wikidata.py "tell me about the following: IBM, Microsoft"
```

```
Answer:
Based on the provided Wikidata knowledge, here is a detailed overview
of IBM and Microsoft.

IBM (Q37156): American multinational technology corporation.
- Country: United States
- Inception: June 16, 1911
- Founders: Charles Ranlett Flint and Thomas John Watson, Sr.
- Headquarters location: Armonk
- Chief Executive Officer: Arvind Krishna
- Chairpersons: Ginni Rometty and John Fellows Akers
- Net profit: 6,023,000,000 (USD)
- Industries: computer industry, computer hardware industry, IT
  service management, information technology consulting, software
- Child organizations: Red Hat, Rational Software, IBM Canada, IBM
  Research, IBM India, IBM Israel, The Weather Company, SoftLayer,
  Lotus Software, and others

Microsoft (Q2283): American multinational technology corporation.
- Country: United States
- Chief Executive Officer: Satya Nadella
- Net profit: 88,136,000,000 and 101,832,000,000 (different years)
- Product or material produced: Microsoft Mobile
- Child organizations: LinkedIn, Nuance Communications, Activision
  Blizzard, ZeniMax Media, Skype Technologies, Turn 10 Studios,
  Avanade, aQuantive, and others

Both corporations are American multinationals with extensive global
operations, numerous subsidiaries, and significant financial
performance. IBM was founded earlier (1911) and has a broader range
of industrial classifications, while Microsoft is known for its
software and recent acquisitions in gaming, social media, and AI.
```

### Combined DBpedia + Wikidata

```bash
uv run DBPedia_and_Wikidata.py "tell me about the following: IBM, Microsoft"
```

```
Answer:
Based on the provided knowledge from DBpedia and Wikidata, here is a
detailed overview of IBM and Microsoft.

IBM (International Business Machines Corporation)
- Type: Public company (American multinational technology corporation)
- Industry: Information technology, computer industry, computer
  hardware industry, IT service management, software industry
- Founded: June 16, 1911
- Founders: Charles Ranlett Flint, George Winthrop Fairchild, Herman
  Hollerith, Thomas John Watson Sr.
- Headquarters: Armonk, United States
- Key People: Arvind Krishna (CEO), Ginni Rometty (chairperson),
  John Fellows Akers (former chairperson)
- Products: AI, Cloud computing, Computer hardware, Quantum computing,
  Robotics, Software, Automation, Blockchain
- Services: Professional services, Outsourcing, Managed services
- Subsidiaries: Red Hat, Rational Software, IBM Canada, IBM Research,
  IBM Israel, The Weather Company, SoftLayer, Lotus Software
- Revenue: $62.73 billion | Net income: $6.023 billion
- Employees: 270,300

Microsoft Corporation
- Type: Public company (American multinational technology corporation)
- Industry: Information technology
- Founded: April 4, 1975
- Founders: Bill Gates, Paul Allen
- Headquarters: Redmond, Washington, United States
- CEO: Satya Nadella
- Products: Cloud computing, Computer hardware, Consumer electronics,
  Software development, Video game industry
- Services: GitHub, LinkedIn, Microsoft 365, Microsoft Azure, Microsoft
  Bing, Microsoft Dynamics, Microsoft Edge, OneDrive, Xbox Game Pass
- Subsidiaries: GitHub, LinkedIn, Nuance Communications, Activision
  Blizzard, ZeniMax Media, Skype Technologies
- Revenue: $281.7 billion | Net income: $101.8 billion
- Employees: 228,000

The DBpedia source provides richer financial figures and product
lists, while Wikidata adds more subsidiaries and leadership details.
Together they give a fuller picture than either source alone.
```

That last example is the key insight of this chapter: because both DBpedia and Wikidata use the same RDF data model, a client program can query both and merge the results. The LLM then synthesizes a single answer that draws on the strengths of each source.

## What This Program Does

The code in this directory is split across four files:

| File | Purpose |
|------|---------|
| `library.py` | Shared utilities: LLM client, entity extraction, answer synthesis, SPARQL query execution, CLI helper |
| `DBPedia.py` | DBpedia-specific templates, property mappings, enrichment, and answer pipeline |
| `Wikidata.py` | Wikidata-specific templates, property mappings, enrichment, and answer pipeline |
| `DBPedia_and_Wikidata.py` | Federated example that queries both knowledge bases and merges results |

Each of the three example scripts implements the same four-stage pipeline:

1. **Entity extraction** - An LLM identifies named entities in the question and classifies them by type (person, organization, place, or misc).
2. **Relationship detection** - A keyword-matching layer checks whether the question asks about a specific relationship such as "capital of" or "married to."
3. **SPARQL retrieval** - The program constructs and executes SPARQL queries against one or more knowledge bases, gathering descriptions and structured facts.
4. **Answer synthesis** - An LLM synthesizes the retrieved facts into a natural-language answer.

The shared `library.py` handles the parts that are identical across all three scripts: the Fireworks.ai LLM client, the entity-extraction prompt, the answer-synthesis prompt, and the SPARQL HTTP transport. The KB-specific scripts handle the parts that differ: SPARQL query templates, property mappings, and enrichment logic.

## The Semantic Web and RDF

The Resource Description Framework (RDF) is the data model underlying the Semantic Web. In RDF, every piece of knowledge is a **triple**:

```
subject  predicate  object
```

The subject and object are nodes in a graph (entities), and the predicate is a typed edge (a relationship). For example, the statement "Paris is the capital of France" becomes the triple:

```
<http://dbpedia.org/resource/Paris>
    <http://dbpedia.org/ontology/capital>
        <http://dbpedia.org/resource/France>
```

Each URI is a globally unique identifier. The predicate URI `http://dbpedia.org/ontology/capital` is not just a label; it is a defined property in the DBpedia ontology with a documented meaning. Because every entity and every relationship has a URI, data from different sources can be merged unambiguously into a single graph.

### DBpedia and Wikidata: Two RDF Knowledge Bases

**DBpedia** is a community project that extracts structured data from Wikipedia infoboxes. It uses human-readable URIs like `http://dbpedia.org/resource/IBM` for entities and `http://dbpedia.org/ontology/foundingDate` for properties. The DBpedia ontology defines classes (Person, Organisation, Place) and properties (birthDate, foundedBy, capital) in a way that mirrors common-sense categories.

**Wikidata** is the structured-data sister project of Wikipedia, maintained by the Wikimedia Foundation. It uses opaque identifiers: entities are Q-numbers (`Q2283` for Microsoft) and properties are P-numbers (`P159` for headquarters location). This design allows Wikidata to be language-neutral and to support claims with qualifiers, references, and ranks, but it makes queries less readable without label resolution.

Both knowledge bases expose public SPARQL endpoints:

| Knowledge base | SPARQL endpoint |
|----------------|-----------------|
| DBpedia | `http://dbpedia.org/sparql` |
| Wikidata | `https://query.wikidata.org/sparql` |

Because both speak SPARQL and return results in the same JSON format, the same client code can query either one.

### Literals and Language Tags

Objects in RDF triples can be URIs (links to other entities) or **literals** (plain values such as strings, numbers, or dates). String literals can carry a **language tag**. For example, the short description of IBM in DBpedia is stored as:

```
"American multinational technology corporation"@en
```

The `@en` suffix marks the string as English. Both DBpedia and Wikidata contain descriptions in many languages, so SPARQL queries almost always include a language filter to avoid mixing languages in the results.

## SPARQL Query Language

SPARQL (SPARQL Protocol and RDF Query Language) is the standard query language for RDF graphs. A SPARQL query matches graph patterns using variables prefixed with `?`. Here is a minimal example that finds the English label of any entity whose capital is France:

```sparql
SELECT ?s WHERE {
  ?s <http://dbpedia.org/ontology/capital> <http://dbpedia.org/resource/France> .
  ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label .
  FILTER(lang(?label) = 'en')
}
```

The `SELECT` clause lists which variables to return. The block inside `{ }` is the graph pattern: one or more triple patterns that the data must match. The `FILTER` function constrains results, in this case requiring the label to be in English.

### Key SPARQL Features Used in This Program

The queries in these scripts use several SPARQL features worth understanding:

**`VALUES`** enumerates a set of allowed values for a variable. Instead of writing three separate triple patterns for three predicates, we write one pattern and constrain the predicate variable:

```sparql
?s ?p ?comment .
VALUES ?p {
  <http://dbpedia.org/ontology/abstract>
  <http://www.w3.org/2000/01/rdf-schema#comment>
  <http://dbpedia.org/ontology/description>
}
```

A powerful variant uses **two-variable `VALUES`** to pair a URI with a literal label in each row. The enrichment queries use this to carry a human-readable property name alongside each property URI:

```sparql
VALUES (?prop ?propLabel) {
  (<http://dbpedia.org/ontology/foundingDate> "foundingDate")
  (<http://dbpedia.org/ontology/industry>    "industry")
}
```

When the query matches a triple `?s dbo:foundingDate ?obj`, the variable `?propLabel` is automatically bound to the string `"foundingDate"`.

**`OPTIONAL`** specifies a pattern that may or may not match. If the optional pattern fails, the rest of the query still returns results, with the optional variables unbound. We use this so that an `rdf:type` constraint does not eliminate entities whose type does not perfectly match our expectation.

**`FILTER`** applies a boolean condition to each candidate row. We use it for language filtering (`lang(?comment) = 'en'`), for excluding Wikipedia category pages (`!CONTAINS(STR(?s), 'dbpedia.org/resource/Category:')`), and for requiring URI-valued objects (`isURI(?obj)`).

**`ORDER BY`** sorts results. We sort by `?p` so that the preferred predicate (the full abstract) appears first in the results, and the shorter description fields come later as fallback.

**`LIMIT`** caps the number of rows returned, which keeps queries fast and prevents oversized responses.

### The Wikidata Label Service

Wikidata queries often use the `wikibase:label` service to resolve Q-numbers to human-readable labels. However, our enrichment queries use a simpler approach: an `OPTIONAL` block that fetches `rdfs:label` for each object value. This works with any SPARQL endpoint and does not require Wikidata-specific extensions.

## The Shared Library: `library.py`

The `library.py` module contains all the code that is identical across the three example scripts. Let us walk through each piece.

### Fireworks.ai Client Setup

The program uses the OpenAI Python SDK to talk to Fireworks.ai, which provides an OpenAI-compatible API. This means we can use the familiar `client.chat.completions.create` interface with a different `base_url`:

```python
import os
import json
import re
import requests
from openai import OpenAI

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.getenv("FIREWORKS_API_KEY"),
)

MODEL_ID = "accounts/fireworks/models/deepseek-v4-flash"
```

The API key is read from the `FIREWORKS_API_KEY` environment variable. If it is not set, the client receives `None`, and the first LLM call raises an authentication error. The `MODEL_ID` constant identifies which model to use; we picked `deepseek-v4-flash` for its speed and low cost, which matters because the program makes up to three LLM calls per question.

A descriptive `User-Agent` string is defined so that SPARQL endpoints (especially Wikidata) do not reject our queries as unidentified bot traffic:

```python
_USER_AGENT = (
    "PythonAIBook/1.0 (educational project; "
    "+https://github.com/markw/PythonAIBook)"
)
```

### The `llm_complete` Helper

Every LLM call in the program follows the same pattern, so we factor it into a single helper:

```python
def llm_complete(prompt: str, max_tokens: int = 3000,
                 temperature: float = 0) -> str:
    """Send a single user message to the Fireworks.ai LLM and return the text."""
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()
```

The `temperature=0` default makes the model deterministic, which is important for entity extraction: we want the same question to produce the same entities every time. Callers that want more creative output (such as answer synthesis) can override the temperature.

### Entity Extraction

Natural-language questions mention entities by name: "Bill Gates," "Paris," "IBM." Before we can query any knowledge base, we need to identify these entities and classify them by type. The `extract_entities` function uses an LLM with a **one-shot prompt** to do this in a single API call.

A one-shot prompt gives the model one example before asking it to perform the task on new input. This anchors the model on the desired output format:

```python
def extract_entities(text: str) -> dict:
    one_shot_prompt = """You are an expert named-entity recognition system.

Identify the named entities in the user's text and classify each one
into exactly one of these types:
- PERSON  (people: individuals, fictional characters)
- ORG     (organizations: companies, bands, teams, government bodies)
- GPE     (geo-political entities: countries, states, cities, regions)
- MISC    (anything else worth looking up: scientific fields, products,
           works of art, concepts)

Return ONLY a JSON object of the form {"TYPE": ["Name", ...]}.
Omit types with no entities. No prose, no markdown fences.

# Example
Text: "Where does Bill Gates work and what is the population of Paris?"
Output: {"PERSON": ["Bill Gates"], "GPE": ["Paris"]}

# Now classify this text
Text: "{text}"
Output: """
```

Several design choices in this prompt are worth noting:

- **Role assignment** ("You are an expert named-entity recognition system") primes the model to adopt the persona of a specialist system.
- **Explicit type definitions** with parenthetical examples prevent the model from guessing what each type means.
- **Format constraint** ("Return ONLY a JSON object") tells the model we want structured output, not prose.
- **A worked example** shows the exact input/output shape, which is more reliable than describing the format abstractly.
- **Negative instructions** ("No prose, no markdown fences") anticipate and suppress common failure modes.

The `{text}` placeholder is replaced with the user's actual question before the prompt is sent to `llm_complete`.

### Parsing and Normalizing the LLM Output

Even with instructions to return only JSON, LLMs sometimes wrap their output in markdown code fences. The code handles this with a regex fallback:

```python
    try:
        raw = llm_complete(
            one_shot_prompt.replace("{text}", text),
            max_tokens=3000,
            temperature=0,
        )
        # Strip markdown code fences if the model wrapped its output
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if fence_match:
            raw = fence_match.group(1).strip()
        entities = json.loads(raw)
        # Normalize: ensure values are lists of strings
        cleaned = {}
        for etype, names in entities.items():
            if isinstance(names, str):
                names = [names]
            cleaned[etype] = [str(n) for n in names]
        return cleaned
    except Exception as e:
        print(f"[Warning] LLM entity extraction failed: {e}")
        return {}
```

The regex `r"```(?:json)?\s*(.*?)```"` matches a fenced code block with an optional `json` language tag. `re.DOTALL` lets `.` match newlines so the fence can span multiple lines. If a fence is found, we extract its inner content; otherwise we parse the raw string directly.

After parsing, the normalization loop converts single strings to one-element lists (the model might return `{"PERSON": "Bill Gates"}` instead of `{"PERSON": ["Bill Gates"]}`) and coerces all names to strings. The broad `except` clause ensures that a malformed LLM response does not crash the program; it returns an empty dict, and the pipeline continues with no entities.

### Answer Synthesis

The `synthesize_answer` function sends the question and the retrieved context to the LLM, asking it to produce a natural-language answer. It is shared by all three scripts:

```python
def synthesize_answer(question: str, context: str,
                      source_label: str = "the knowledge base") -> str:
    short_context = context[:3000] if len(context) > 3000 else context
    prompt = (
        f'Given question: "{question}"\n\n'
        f"Knowledge from {source_label} about relevant entities:\n"
        f"{short_context}\n\n"
        "Please answer the question based on this knowledge. "
        "Be specific, address every entity the user asked about, and "
        "cite entity names where possible."
    )
    try:
        return llm_complete(prompt, max_tokens=3500)
    except Exception as e:
        print(f"[Warning] LLM synthesis failed: {e}")
        return ""
```

The prompt has three parts: the original question, the retrieved knowledge, and instructions on how to answer. The instruction to "address every entity the user asked about" is critical for multi-entity questions like "tell me about IBM, Microsoft." Without it, the model might focus on only the first entity and ignore the rest.

The `source_label` parameter lets each script name its source: "DBpedia", "Wikidata", or "DBpedia and Wikidata". The context is truncated to 3000 characters to stay within the model's context window and control cost.

### SPARQL Query Execution

All SPARQL queries go through a single function that uses the `requests` library to send the query over HTTP:

```python
def query_sparql(endpoint: str, sparql_query: str) -> list[dict]:
    print(f"SPARQL query ({endpoint}):\n{sparql_query}")
    resp = requests.get(
        endpoint,
        params={"query": sparql_query, "format": "json"},
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/sparql-results+json",
        },
        timeout=60,
    )
    resp.raise_for_status()
    ret = resp.json()["results"]["bindings"]
    print(f"Results: {len(ret)} bindings\n")
    return ret
```

The function is endpoint-agnostic: it works with DBpedia, Wikidata, or any other SPARQL 1.1 endpoint that supports JSON results. The `User-Agent` header is especially important for Wikidata, which rejects requests without one. The `format=json` query parameter tells the endpoint to return results in the SPARQL 1.1 JSON Results format.

The JSON Results format represents each value as a dict with a `type` (`uri`, `literal`, `typed-literal`, or `bnode`) and a `value`. For literals, an optional `xml:lang` key carries the language tag. The rest of the code accesses values with patterns like `result["comment"]["value"]` and `result["objLabel"]["value"]`.

### Relationship Detection

Many questions ask about a specific relationship: "What is the **capital** of France?" "Where was Bill Gates **born**?" The `detect_relationship` function is generic: it takes the question and a caller-supplied mapping table, and returns the matching property URI and verb:

```python
def detect_relationship(question: str,
                        relationship_table: dict) -> tuple[str, str] | None:
    lower = question.lower()
    for key in sorted(relationship_table, key=len, reverse=True):
        if key in lower:
            uri, verb = relationship_table[key]
            return uri, verb
    return None
```

The keys are sorted by descending length so that multi-word phrases are checked before their single-word sub-phrases. Without this, the question "What is the capital of France?" would match "capital" before "capital of," and while both map to the same URI in this case, the ordering matters for phrases where a shorter key maps to a different property than the longer one.

### Value Resolution Helper

Both DBpedia and Wikidata return values that can be either literals (dates, numbers) or URIs (links to other entities). The `resolve_value` helper converts any value to a human-readable string:

```python
def resolve_value(obj: str, obj_label: str | None) -> str:
    if obj_label:
        return obj_label
    if obj.startswith("http"):
        return obj.rsplit("/", 1)[-1].replace("_", " ")
    return obj
```

If the SPARQL query resolved the object to an English label (via an `OPTIONAL` block), we use that. Otherwise, if the object is a URI, we extract the last path segment and replace underscores with spaces (for example, `Information_technology` becomes `Information technology`). If the object is a literal, we return it directly.

### The Shared CLI

All three scripts use the same command-line interface, provided by `run_cli`:

```python
def run_cli(answer_fn, script_name: str):
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print(f"{script_name} - QA with SPARQL + LLM")
        print("-" * 50)
        question = input("Enter your question: ")

    print(f"\nQuestion: {question}")
    print("-" * 50)

    answer, context = answer_fn(question)

    print("\nAnswer:")
    print(answer)
```

The `answer_fn` parameter is a callable supplied by each script. This is how `DBPedia.py`, `Wikidata.py`, and `DBPedia_and_Wikidata.py` each plug in their own answer pipeline while sharing the same CLI logic.

## Example 1: DBpedia (`DBPedia.py`)

The DBpedia script defines three SPARQL query templates, a relationship property mapping, an entity-type-to-ontology mapping, and an enrichment property table. Let us walk through each.

### Entity Lookup Template

The first template finds an entity by its English label and retrieves descriptive text:

```python
SPARQL_QUERY_TEMPLATE = """
SELECT DISTINCT ?s ?p ?comment WHERE {{
  ?s <http://www.w3.org/2000/01/rdf-schema#label> '{name}'@en .
  FILTER(!CONTAINS(STR(?s), 'dbpedia.org/resource/Category:')) .
  ?s ?p ?comment .
  FILTER (lang(?comment) = 'en') .
  VALUES ?p {{
    <http://dbpedia.org/ontology/abstract>
    <http://www.w3.org/2000/01/rdf-schema#comment>
    <http://dbpedia.org/ontology/description>
  }}
  {dbpedia_type}
}} ORDER BY ?p LIMIT 15
"""
```

Line by line:

- `SELECT DISTINCT ?s ?p ?comment` returns the subject URI, the predicate used, and the descriptive text.
- `?s rdfs:label '{name}'@en` finds entities whose English label matches the name we are looking for.
- `FILTER(!CONTAINS(STR(?s), 'dbpedia.org/resource/Category:'))` excludes Wikipedia category pages, which carry only the unhelpful label "Wikimedia category."
- `VALUES ?p { ... }` lists three predicates to try, in priority order.
- `{dbpedia_type}` is replaced with an optional `rdf:type` constraint.
- `ORDER BY ?p` sorts results so the abstract comes before the shorter comment and description.
- `LIMIT 15` caps the result set.

The doubled braces `{{ }}` are Python `str.format` escapes; they produce literal braces in the output. The single-brace placeholders `{name}` and `{dbpedia_type}` are replaced at runtime.

The template is instantiated by `build_sparql_query`:

```python
def build_sparql_query(name: str, dbpedia_type_uri: str) -> str:
    if dbpedia_type_uri:
        type_clause = (
            f"OPTIONAL {{ ?s "
            f"<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
            f"{dbpedia_type_uri} . }}"
        )
    else:
        type_clause = ""
    return SPARQL_QUERY_TEMPLATE.format(name=name, dbpedia_type=type_clause)
```

The type clause is wrapped in `OPTIONAL` deliberately: if we made it a hard requirement, entities whose LLM-assigned type does not exactly match the DBpedia ontology would be missed. For example, "Pepsi" might be tagged as an organization by the LLM but typed as `dbo:Beverage` in DBpedia. With `OPTIONAL`, the label match alone is sufficient.

### Enrichment Template and Properties

The enrichment template fetches a curated set of property/value pairs in one query:

```python
SPARQL_ENRICHMENT_TEMPLATE = """
SELECT DISTINCT ?propLabel ?obj ?objLabel WHERE {{
  ?s <http://www.w3.org/2000/01/rdf-schema#label> '{name}'@en .
  VALUES (?prop ?propLabel) {{
    {values}
  }}
  ?s ?prop ?obj .
  OPTIONAL {{
    ?obj <http://www.w3.org/2000/01/rdf-schema#label> ?objLabel .
    FILTER(lang(?objLabel) = 'en')
  }}
}} LIMIT 80
"""
```

This uses the two-variable `VALUES` clause described earlier. Each row pairs a property URI with a human-readable label string. The `OPTIONAL` block resolves URI-valued objects to their English label.

The `ENRICHMENT_PROPERTIES` dictionary defines which DBpedia properties to fetch for each entity type:

```python
ENRICHMENT_PROPERTIES = {
    "ORG": {
        "foundingDate": "http://dbpedia.org/ontology/foundingDate",
        "foundingYear": "http://dbpedia.org/ontology/foundingYear",
        "foundedBy": "http://dbpedia.org/ontology/foundedBy",
        "founder": "http://dbpedia.org/ontology/founder",
        "industry": "http://dbpedia.org/ontology/industry",
        "type": "http://dbpedia.org/ontology/type",
        "service": "http://dbpedia.org/ontology/service",
        "product": "http://dbpedia.org/ontology/product",
        "keyPerson": "http://dbpedia.org/ontology/keyPerson",
        "numberOfEmployees": "http://dbpedia.org/ontology/numberOfEmployees",
        "revenue": "http://dbpedia.org/ontology/revenue",
        "netIncome": "http://dbpedia.org/ontology/netIncome",
        "operatingIncome": "http://dbpedia.org/ontology/operatingIncome",
        "locationCity": "http://dbpedia.org/ontology/locationCity",
        "locationCountry": "http://dbpedia.org/ontology/locationCountry",
        "subsidiary": "http://dbpedia.org/ontology/subsidiary",
    },
    "PERSON": {
        "birthDate": "http://dbpedia.org/ontology/birthDate",
        "birthPlace": "http://dbpedia.org/ontology/birthPlace",
        "deathDate": "http://dbpedia.org/ontology/deathDate",
        "deathPlace": "http://dbpedia.org/ontology/deathPlace",
        "occupation": "http://dbpedia.org/ontology/occupation",
        "nationality": "http://dbpedia.org/ontology/nationality",
        "spouse": "http://dbpedia.org/ontology/spouse",
        "almaMater": "http://dbpedia.org/ontology/almaMater",
        "knownFor": "http://dbpedia.org/ontology/knownFor",
        "employer": "http://dbpedia.org/ontology/employer",
        "award": "http://dbpedia.org/ontology/award",
    },
    "GPE": {
        "populationTotal": "http://dbpedia.org/ontology/populationTotal",
        "areaTotal": "http://dbpedia.org/ontology/areaTotal",
        "country": "http://dbpedia.org/ontology/country",
        "capital": "http://dbpedia.org/ontology/capital",
        "leaderName": "http://dbpedia.org/ontology/leaderName",
        "type": "http://dbpedia.org/ontology/type",
        "language": "http://dbpedia.org/ontology/language",
        "currency": "http://dbpedia.org/ontology/currency",
    },
    "MISC": {},
}
```

The properties are chosen to match the kinds of information a person would typically want to know about each entity type. For an organization, that means who founded it, what industry it is in, how many employees it has, and what its revenue is. For a person, that means birth and death dates, occupation, nationality, and spouse. For a place, that means population, area, capital, and leader.

### The `enrich_entity` Function

The enrichment function builds the `VALUES` block from the property table, executes the SPARQL query, and converts the raw results into human-readable fact strings:

```python
def enrich_entity(name: str, entity_type: str) -> list[str]:
    props = ENRICHMENT_PROPERTIES.get(entity_type, {})
    if not props:
        return []

    values_block = "\n    ".join(
        f'(<{uri}> "{label}")' for label, uri in props.items()
    )
    query = SPARQL_ENRICHMENT_TEMPLATE.format(name=name, values=values_block)
    try:
        results = query_dbpedia(query)
    except Exception as e:
        print(f"[Warning] Enrichment query failed for {name}: {e}")
        return []

    facts = []
    seen = set()
    for r in results:
        label = r.get("propLabel", {}).get("value", "")
        obj = r.get("obj", {}).get("value", "")
        obj_label = r.get("objLabel", {}).get("value")
        value = resolve_value(obj, obj_label)
        if not value:
            continue
        key = (label, value)
        if key in seen:
            continue
        seen.add(key)
        facts.append(f"{label}: {value}")
    return facts
```

The `values_block` string is built by joining together `(<uri> "label")` pairs, one per property. For an organization like IBM, this produces a block of sixteen rows. The `seen` set deduplicates facts, because DBpedia sometimes has multiple labels for the same URI.

### The DBpedia Answer Pipeline

The `answer_question` function in `DBPedia.py` orchestrates the full pipeline:

```python
def answer_question(question: str) -> tuple[str, str]:
    entities = extract_entities(question)
    print(f"[DEBUG] Entities found: {entities}")

    # Step 1: Relationship detection -> direct SPARQL property lookup
    relationship = detect_relationship(question, RELATIONSHIP_PROPERTIES)
    if relationship:
        property_uri, verb = relationship
        all_names = [n for names in entities.values() for n in names]
        if not all_names:
            all_names = [w for w in question.replace("?", "").split()
                         if w[0].isupper()]
        for name in all_names:
            try:
                rel_results = query_relationship(name, property_uri)
                if rel_results:
                    parts = []
                    for r in rel_results[:5]:
                        label = r["label"]
                        comment = r.get("comment", "")
                        if comment:
                            parts.append(f"{label} ({comment})")
                        else:
                            parts.append(label)
                    answer = f"The {verb} of {name} is: {', '.join(parts)}."
                    return answer, "\n".join(parts)
            except Exception as e:
                print(f"[Warning] Relationship query failed for {name}: {e}")

    # Step 2: General entity lookup + enrichment
    context = get_entity_context(entities)
    if not context.strip():
        return "I couldn't find relevant entities in the knowledge base.", ""

    # Step 3: LLM synthesis
    answer = synthesize_answer(question, context, source_label="DBpedia")
    if answer:
        return answer, context

    # Step 4: Fallback (raw descriptions)
    # ... returns best comment per entity if LLM synthesis fails
```

If a relationship is detected, the function returns immediately with a direct answer like "The capital of France is: Paris." This early return is intentional: for relationship questions, the direct SPARQL lookup gives a precise, fast answer. Otherwise, the program builds a context string from DBpedia, asks the LLM to synthesize an answer, and falls back to raw descriptions if the LLM call fails.

### The DBpedia Relationship Property Mapping

The `RELATIONSHIP_PROPERTIES` dictionary maps English phrases to DBpedia ontology property URIs:

```python
RELATIONSHIP_PROPERTIES = {
    "capital": ("http://dbpedia.org/ontology/capital", "capital"),
    "capital of": ("http://dbpedia.org/ontology/capital", "capital"),
    "birthplace": ("http://dbpedia.org/ontology/birthPlace", "birthplace"),
    "born": ("http://dbpedia.org/ontology/birthPlace", "birthplace"),
    "born in": ("http://dbpedia.org/ontology/birthPlace", "birthplace"),
    "spouse": ("http://dbpedia.org/ontology/spouse", "spouse"),
    "married to": ("http://dbpedia.org/ontology/spouse", "spouse"),
    "founded by": ("http://dbpedia.org/ontology/foundedBy", "founder"),
    "founded": ("http://dbpedia.org/ontology/foundedBy", "founder"),
    "headquarters": ("http://dbpedia.org/ontology/locationCity",
                     "headquarters location"),
    "population": ("http://dbpedia.org/ontology/populationTotal",
                   "population"),
    "currency": ("http://dbpedia.org/ontology/currency", "currency"),
    "language": ("http://dbpedia.org/ontology/language", "language"),
    # ... more mappings
}
```

Multiple English phrases can map to the same DBpedia property. Both "capital" and "capital of" point to `dbo:capital`. This redundancy is intentional: users phrase questions in many ways.

### Entity Type Mapping for DBpedia

```python
ENTITY_TYPE_TO_DBPEDIA_URI = {
    "PERSON": "<http://dbpedia.org/ontology/Person>",
    "ORG":    "<http://dbpedia.org/ontology/Organisation>",
    "GPE":    "<http://dbpedia.org/ontology/Place>",
    "MISC":   "",
}
```

This disambiguates entities with common names. "Washington" could be a person (George Washington) or a place (Washington state). If the LLM tags it as a GPE, the type constraint steers DBpedia toward the geographic entity. For MISC entities, the empty string means the type clause is omitted entirely.

## Example 2: Wikidata (`Wikidata.py`)

Wikidata uses a different data model from DBpedia. Entities are identified by Q-numbers and properties by P-numbers. This section explains the key differences in the SPARQL templates and query patterns.

### Entity Lookup: Resolving Labels to Q-Numbers

The first step in any Wikidata query is to find the Q-number for an entity given its English label. The lookup template filters out disambiguation pages and sorts by ascending Q-number so the most prominent entity comes first:

```python
SPARQL_LOOKUP_TEMPLATE = """
SELECT ?item ?itemLabel ?description WHERE {{
  ?item rdfs:label '{name}'@en .
  ?item schema:description ?description .
  FILTER(LANG(?description) = 'en') .
  FILTER(CONTAINS(STR(?item), '/entity/Q')) .
  BIND(REPLACE(STR(?item), '.*Q', '') AS ?qnum) .
  OPTIONAL {{ ?item rdfs:label ?itemLabel .
             FILTER(LANG(?itemLabel) = 'en') . }}
}}
ORDER BY xsd:integer(?qnum)
LIMIT 5
"""
```

The `BIND` clause extracts the numeric part of the Q-number, and `ORDER BY xsd:integer(?qnum)` sorts results so that Q2283 (Microsoft) comes before Q124998 (a Microsoft brand page). The `schema:description` predicate gives us the short Wikidata description, which is the equivalent of DBpedia's `dbo:description`.

The `lookup_entity_qid` function executes this query and returns the first non-disambiguation Q-number:

```python
def lookup_entity_qid(name: str) -> str | None:
    query = SPARQL_LOOKUP_TEMPLATE.format(name=name.replace("'", "\\'"))
    try:
        results = _query_wikidata(query)
    except Exception as e:
        print(f"[Warning] Wikidata lookup failed for {name}: {e}")
        return None

    for r in results:
        item_uri = r.get("item", {}).get("value", "")
        desc = r.get("description", {}).get("value", "")
        if "disambiguation" in desc.lower():
            continue
        qid = item_uri.rsplit("/", 1)[-1]
        if qid.startswith("Q"):
            return qid
    return None
```

### Rate-Limit Handling

Wikidata enforces stricter rate limits than DBpedia. The `_query_wikidata` wrapper inserts a delay between queries to avoid HTTP 429 responses:

```python
SPARQL_DELAY = 1.0  # seconds between queries

def _query_wikidata(sparql_query: str) -> list[dict]:
    results = query_sparql(WIKIDATA_ENDPOINT, sparql_query)
    time.sleep(SPARQL_DELAY)
    return results
```

This is a simple but effective strategy. In a production system you would use exponential backoff on 429 responses, but for an educational example the fixed delay keeps the code simple.

### Enrichment: The Wikidata Meta-Model

Wikidata's property system has a meta-model that differs from DBpedia's. Each direct-claim property (`wdt:P159`, for "headquarters location") has a corresponding property entity (`wd:P159`) whose `rdfs:label` gives us the human-readable property name. The enrichment template uses this to fetch both the property name and the value in one query:

```python
SPARQL_ENRICHMENT_TEMPLATE = """
SELECT DISTINCT ?propLabel ?obj ?objLabel WHERE {{
  VALUES (?p ?propEntity) {{
    {values}
  }}
  wd:{qid} ?p ?obj .
  ?propEntity rdfs:label ?propLabel .
  FILTER(LANG(?propLabel) = 'en') .
  OPTIONAL {{
    ?obj rdfs:label ?objLabel .
    FILTER(LANG(?objLabel) = 'en') .
  }}
}}
LIMIT 80
"""
```

The `VALUES` block pairs each direct-claim property with its property entity:

```python
def enrich_entity(qid: str, entity_type: str) -> list[str]:
    props = ENRICHMENT_PROPERTIES.get(entity_type, {})
    if not props:
        return []

    values_lines = []
    for label, pid in props.items():
        values_lines.append(f"(wdt:{pid} wd:{pid})")
    values_block = "\n    ".join(values_lines)

    query = SPARQL_ENRICHMENT_TEMPLATE.format(qid=qid, values=values_block)
    # ... execute query, resolve labels, deduplicate, return facts
```

For IBM (Q37156), the `VALUES` block contains ten pairs like `(wdt:P571 wd:P571)` (inception), `(wdt:P112 wd:P112)` (founded by), and so on. The query matches all ten properties in a single round trip.

### Wikidata Enrichment Properties

The enrichment property table for Wikidata uses P-numbers instead of full URIs:

```python
ENRICHMENT_PROPERTIES = {
    "ORG": {
        "inception": "P571",
        "founded by": "P112",
        "industry": "P452",
        "headquarters location": "P159",
        "country": "P17",
        "chairperson": "P488",
        "CEO": "P169",
        "subsidiary": "P355",
        "product or material produced": "P1056",
        "net profit": "P2295",
    },
    "PERSON": {
        "date of birth": "P569",
        "place of birth": "P19",
        "date of death": "P570",
        "place of death": "P20",
        "occupation": "P106",
        "country of citizenship": "P27",
        "spouse": "P26",
        "educated at": "P69",
        "employer": "P108",
        "award received": "P166",
        "member of": "P463",
    },
    "GPE": {
        "population": "P1082",
        "area": "P2046",
        "country": "P17",
        "capital": "P36",
        "head of state": "P35",
        "head of government": "P6",
        "official language": "P37",
        "currency": "P38",
    },
    "MISC": {},
}
```

Compare this with the DBpedia enrichment table: the property names are different (DBpedia uses camelCase like `foundingDate`, Wikidata uses descriptive labels like `inception`), but the logical coverage is the same. Both tables define which properties to fetch for each entity type.

### The Wikidata Relationship Property Mapping

The relationship mapping for Wikidata uses P-number property IDs instead of full URIs:

```python
RELATIONSHIP_PROPERTIES = {
    "capital": ("P36", "capital"),
    "capital of": ("P36", "capital"),
    "birthplace": ("P19", "birthplace"),
    "born": ("P19", "birthplace"),
    "born in": ("P19", "birthplace"),
    "spouse": ("P26", "spouse"),
    "married to": ("P26", "spouse"),
    "founded by": ("P112", "founder"),
    "founded": ("P112", "founder"),
    "headquarters": ("P159", "headquarters location"),
    "population": ("P1082", "population"),
    "currency": ("P38", "currency"),
    "language": ("P37", "official language"),
    # ... more mappings
}
```

The relationship template takes a Q-number and a P-number and fetches the related entity's label:

```python
SPARQL_RELATIONSHIP_TEMPLATE = """
SELECT DISTINCT ?obj ?objLabel WHERE {{
  wd:{qid} wdt:{pid} ?obj .
  ?obj rdfs:label ?objLabel .
  FILTER(LANG(?objLabel) = 'en') .
}}
LIMIT 10
"""
```

This is simpler than the DBpedia relationship template because we already know the entity's Q-number from the lookup step, so we can use it directly in the `wd:{qid}` pattern instead of matching by label.

### The Wikidata Answer Pipeline

The Wikidata answer pipeline follows the same structure as the DBpedia one, but with an extra step: each entity name must be resolved to a Q-number before any enrichment queries can be issued:

```python
def answer_question(question: str) -> tuple[str, str]:
    entities = extract_entities(question)
    print(f"[DEBUG] Entities found: {entities}")

    # Relationship detection -> direct Wikidata property lookup
    relationship = detect_relationship(question, RELATIONSHIP_PROPERTIES)
    if relationship:
        pid, verb = relationship
        all_names = [n for names in entities.values() for n in names]
        for name in all_names:
            qid = lookup_entity_qid(name)
            if not qid:
                continue
            rel_results = query_relationship(qid, pid)
            if rel_results:
                parts = [r["label"] for r in rel_results[:5] if r["label"]]
                if parts:
                    answer = f"The {verb} of {name} is: {', '.join(parts)}."
                    return answer, "\n".join(parts)

    # General entity lookup + enrichment
    context = get_entity_context(entities)
    if not context.strip():
        return "I couldn't find relevant entities in the knowledge base.", ""

    # LLM synthesis
    answer = synthesize_answer(question, context, source_label="Wikidata")
    if answer:
        return answer, context

    return "Unable to generate an answer.", context
```

The `get_entity_context` function for Wikidata resolves each entity name to a Q-number, fetches the short description, and runs the enrichment query:

```python
def get_entity_context(entities: dict) -> str:
    context_parts = []

    for entity_type in ENTITY_TYPES:
        if entity_type not in entities:
            continue
        for name in entities[entity_type]:
            qid = lookup_entity_qid(name)
            if not qid:
                continue
            print(f"[DEBUG] {name} -> Wikidata {qid}")

            # Fetch description
            desc = ""
            # ... (lookup query to get description for this QID)

            # Fetch enrichment facts
            facts = enrich_entity(qid, entity_type)

            if desc and facts:
                context_parts.append(
                    f"{name} ({qid}): {desc}\n" + "\n".join(facts))
            elif desc:
                context_parts.append(f"{name} ({qid}): {desc}")
            elif facts:
                context_parts.append(
                    f"{name} ({qid}):\n" + "\n".join(facts))

    return "\n\n".join(context_parts)
```

The context string includes the Q-number alongside the entity name, which helps the LLM (and the reader) understand where the data came from.

## Example 3: Combined DBpedia + Wikidata (`DBPedia_and_Wikidata.py`)

The combined script demonstrates one of the most powerful features of the Semantic Web: because both DBpedia and Wikidata use RDF and SPARQL, a client application can query both and merge the results. This is called **federation**.

The script is remarkably short because it delegates all the work to the two KB-specific modules:

```python
from library import (
    extract_entities, synthesize_answer, run_cli,
)

import DBPedia
import Wikidata


def answer_question(question: str) -> tuple[str, str]:
    entities = extract_entities(question)
    print(f"[DEBUG] Entities found: {entities}")

    # --- DBpedia context ---
    print("\n--- Querying DBpedia ---")
    dbpedia_context = DBPedia.get_entity_context(entities)
    print(f"[DEBUG] DBpedia context: {len(dbpedia_context)} chars")

    # --- Wikidata context ---
    print("\n--- Querying Wikidata ---")
    wikidata_context = Wikidata.get_entity_context(entities)
    print(f"[DEBUG] Wikidata context: {len(wikidata_context)} chars")

    # --- Merge contexts ---
    parts = []
    if dbpedia_context.strip():
        parts.append("=== Knowledge from DBpedia ===\n" + dbpedia_context)
    if wikidata_context.strip():
        parts.append("=== Knowledge from Wikidata ===\n" + wikidata_context)

    combined_context = "\n\n".join(parts)

    if not combined_context.strip():
        return ("I couldn't find relevant entities in either knowledge "
                "base.", "")

    # --- LLM synthesis over the merged context ---
    answer = synthesize_answer(
        question, combined_context,
        source_label="DBpedia and Wikidata",
    )
    if answer:
        return answer, combined_context

    return combined_context, combined_context
```

The function calls `DBPedia.get_entity_context` and `Wikidata.get_entity_context` with the same entity dict, then concatenates the two context strings with source labels. The LLM sees both blocks of facts and synthesizes an answer that draws on both.

### What Each Source Contributes

When we look at the combined output, we can see what each knowledge base contributes:

- **DBpedia** provides richer product and service lists (it extracts from Wikipedia infoboxes, which list products individually), financial figures formatted as typed literals, and the short `dbo:description` text.
- **Wikidata** provides more subsidiaries (its P355 property lists child organizations comprehensively), leadership details (current and former chairpersons), and structured industry classifications that are more granular than DBpedia's.

The LLM synthesis step naturally reconciles overlapping facts. When both sources say IBM was founded on June 16, 1911, the LLM states it once. When Wikidata lists Thomas J. Watson Sr. as a founder but DBpedia does not, the LLM includes him. This is a practical demonstration of how the Semantic Web enables knowledge fusion.

### Why Not SPARQL Federation?

SPARQL supports a `SERVICE` keyword that lets a single query federate across multiple endpoints:

```sparql
SELECT ?s ?comment WHERE {
  SERVICE <http://dbpedia.org/sparql> {
    ?s rdfs:label "IBM"@en . ?s dbo:description ?comment .
  }
}
```

We do not use `SERVICE` in this program for two reasons. First, the Wikidata endpoint often blocks federated queries from external endpoints due to rate limiting. Second, our approach of querying each endpoint separately and merging the results in Python gives us more control over error handling and rate limiting. The tradeoff is that we make more HTTP requests, but the `SPARQL_DELAY` between Wikidata queries keeps us within rate limits.

## Running the Code

### Installation

The project uses `uv` for dependency management. Install dependencies with:

```bash
uv sync
```

This reads `pyproject.toml` and creates a virtual environment with `openai` and `requests`.

### Configuration

Set your Fireworks.ai API key as an environment variable:

```bash
export FIREWORKS_API_KEY="your-api-key"
```

### Running the Three Examples

**DBpedia only:**

```bash
uv run DBPedia.py "tell me about the following: IBM, Microsoft"
```

**Wikidata only:**

```bash
uv run Wikidata.py "tell me about the following: IBM, Microsoft"
```

**Combined DBpedia + Wikidata:**

```bash
uv run DBPedia_and_Wikidata.py "tell me about the following: IBM, Microsoft"
```

All three scripts also accept relationship queries:

```bash
uv run DBPedia.py "What is the capital of France"
uv run Wikidata.py "Where was Bill Gates born"
```

And all three can run in interactive mode (no arguments):

```bash
uv run DBPedia.py
```

### Multi-Turn Chat (DBpedia)

`DBPedia.py` also provides a `chat_with_context` function for a conversational interface that maintains message history:

```python
from DBPedia import chat_with_context
chat_with_context()
```

Each turn, the function extracts entities, fetches DBpedia context, and appends both to the message list. The assistant's reply is appended back, so subsequent turns have access to the full conversation history.

## Example Queries

### Relationship Queries

These queries match against the built-in property mapping tables and return direct answers:

| Query | DBpedia result | Wikidata result |
|-------|----------------|-----------------|
| "What is the capital of France" | Paris | Paris |
| "Where was Bill Gates born" | Seattle | Seattle |
| "Who is Bill Gates married to" | Melinda French Gates | Melinda French Gates |
| "Who founded IBM" | Charles Ranlett Flint, et al. | Charles Ranlett Flint, Thomas Watson Sr. |
| "What is the population of Paris" | Population figure | Population figure |
| "Where is IBM headquartered" | Armonk, New York | Armonk |

### Entity Description Queries

These queries go through the full enrichment and LLM synthesis pipeline:

```bash
uv run DBPedia.py "tell me about the following: IBM, Microsoft"
uv run Wikidata.py "tell me about the following: IBM, Microsoft"
uv run DBPedia_and_Wikidata.py "tell me about the following: IBM, Microsoft"
```

The example outputs shown at the beginning of this chapter are the results of these commands.

## Troubleshooting

**`FIREWORKS_API_KEY` not set** - The script passes `None` to the OpenAI client, which raises an authentication error on the first LLM call. Make sure you have exported the variable in the shell where you run the script.

**Wikidata returns HTTP 429 (Too Many Requests)** - Wikidata's public endpoint aggressively rate-limits queries. The `SPARQL_DELAY` constant in `Wikidata.py` inserts a pause between queries, but if you still get 429 errors, increase the delay or wait a minute before retrying.

**DBpedia SPARQL queries return empty results** - DBpedia occasionally changes which predicates are available. The query templates try multiple alternatives (`abstract`, `comment`, `description`), and the enrichment queries use stable `dbo:` properties. If all fail, wait a few minutes and retry; the public endpoint is sometimes overloaded.

**Entity not found** - Try the canonical name. For example, use "United States" instead of "America," and "Bill Gates" instead of "William Gates."

**LLM returns malformed JSON** - The entity extraction function has a regex fallback to strip markdown code fences, but if the model returns something that is not JSON at all, the exception handler returns an empty dict and the pipeline continues with no entities. Retrying usually works.

## Exercises

1. **Add support for additional relationship keywords** (e.g., "nationality", "child", "parent")

   Extend the `RELATIONSHIP_PROPERTIES` dictionaries in both `DBPedia.py` and `Wikidata.py`. Find the appropriate property URIs for DBpedia at https://dbpedia.org/ontology/ and the P-numbers for Wikidata at https://www.wikidata.org/wiki/Wikidata:List_of_properties. Example:
   ```python
   # DBPedia.py
   "nationality": ("http://dbpedia.org/ontology/nationality", "nationality"),
   # Wikidata.py
   "nationality": ("P27", "country of citizenship"),
   ```

2. **Add a new entity type: "PRODUCT"**

   Add `PRODUCT` to the entity extraction prompt in `library.py`, add a DBpedia type URI and enrichment properties in `DBPedia.py`, and add Wikidata P-numbers in `Wikidata.py`. Research DBpedia's `dbo:Product` class and Wikidata's `Q2424752` (product) class.

3. **Create a caching layer for SPARQL query results**

   Implement a simple in-memory cache using Python's `functools.lru_cache` or a dictionary keyed by query string. This is especially valuable for the Wikidata script, where rate limiting makes redundant queries expensive. Example:
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=256)
   def query_sparql_cached(endpoint: str, sparql_query: str):
       return query_sparql(endpoint, sparql_query)
   ```

4. **Modify the combined script to deduplicate overlapping facts**

   Currently, `DBPedia_and_Wikidata.py` passes both context strings to the LLM without deduplication. Implement a Python-side deduplication step that removes facts that appear in both contexts (e.g., "foundingDate: 1911-06-16"). Consider normalizing property labels across the two sources before comparing.

5. **Add support for follow-up questions that reference entities from previous answers**

   Modify the `answer_question` functions to track conversation history. Use entity coreference resolution ("it", "they") by maintaining a context dictionary of recently mentioned entities. Consider storing entity names with their types and Q-numbers for disambiguation.

6. **Handle multi-valued properties in the enrichment layer**

   Some DBpedia and Wikidata properties (like `product` and `subsidiary`) return many values. Currently each value becomes a separate fact line. Modify `enrich_entity` in both scripts to group values by property label, producing a single line like `subsidiary: Red Hat, Lotus Software, IBM Research, ...` instead of multiple `subsidiary: Red Hat` lines.

7. **Add a fourth knowledge base: DBpedia Japanese endpoint or another language**

   DBpedia has language-specific endpoints (e.g., `http://ja.dbpedia.org/sparql` for Japanese). Create a new script that queries a non-English DBpedia endpoint, and modify the entity extraction prompt to detect the question's language. This exercise explores how the Semantic Web enables multilingual knowledge access.

## Further Reading

- DBpedia: https://wiki.dbpedia.org/
- Wikidata: https://www.wikidata.org/wiki/Wikidata:Main_Page
- Wikidata SPARQL query service: https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service
- SPARQL specification: https://www.w3.org/TR/sparql11-query/
- Fireworks.ai API documentation: https://docs.fireworks.ai/
- RDF 1.1 Primer: https://www.w3.org/TR/rdf11-primer/
