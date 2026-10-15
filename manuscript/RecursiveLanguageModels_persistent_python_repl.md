# Recursive Language Models: Searching Wikidata from a Persistent Python REPL

Most LLM agents use JSON tool calls. The model asks for a tool, the harness runs it, and the harness pastes the raw result back into the chat. That works for small results. It fails when a single Wikidata entity returns hundreds of claims, and a question needs three or four hops through the graph. Each hop adds a large blob of JSON to the context, and the model must join the results in its head.

This chapter builds a different kind of harness, a *Recursive Language Model* (RLM) harness. The model does not call tools with JSON. It writes Python. The harness runs that Python in a REPL whose variables persist from turn to turn. Graph data lands in Python variables, not in the chat, and the model sees only what it prints. When the model needs to read text and judge it, it calls a small, stateless sub-LLM from inside its own code. That call from code back into an LLM is the "recursive" part of the name.

The example follows the paper *Wikidata Search Traces: A Dataset for Training Knowledge Graph Search Agents* by Chenene, Rosas-Hinostroza, Stasenko, Evenstein Sigalov, and Langlais ([arxiv.org/html/2610.06650v3](https://arxiv.org/html/2610.06650v3)). The authors release an RLM harness that keeps graph results in persistent Python state and uses sub-calls to interpret selected evidence. They report that on 100 questions the harness beats direct tool calling for both models they test. Their conclusion is that much long-horizon search failure comes from how an agent manages the evidence it retrieves, not only from the model itself.

Our version is small enough to read in one sitting: one module for the harness and one for a benchmark. It runs against any OpenAI-compatible endpoint, so you can test it with local models served by Ollama or oMLX.

## The Program at a Glance

The code lives in `source-code/RecursiveLanguageModels_persistent_python_repl`:

| File | Role |
| --- | --- |
| `rlm_wikidata_bidirectional.py` | SQLite cache, Wikidata graph primitives, the REPL harness and its loop |
| `benchmark_suite.py` | Four multi-hop tasks, a cache seeder, an answer checker, and a summary table |
| `ARCHITECTURE.txt` | ASCII diagrams that compare JSON tool calling with the RLM harness |
| `DESIGN_NOTES.md` | Notes on the benchmark and seeder design |
| `SAMPLE_OUTPUT.txt` | A full benchmark run with a small local model |
| `Makefile` | `create_db`, `run_omlx`, and `clean` targets |

The project needs Python 3.14 or newer and two packages, `openai` and `requests`. It uses [uv](https://docs.astral.sh/uv/):

```
[project]
name = "persistent-python-repl"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = [
    "openai>=3.31.0",
    "requests>=2.34.2",
]
```

## Two Agent Designs Compared

The file `ARCHITECTURE.txt` draws the two designs side by side. Here is the standard JSON tool-calling agent:

```
  +-----------------------+       JSON Tool Call        +-------------------+
  |      Primary LLM      | --------------------------> |  Wikidata API /   |
  |                       |                             |  SPARQL Endpoint  |
  |  [Message History]    | <-------------------------- |                   |
  |  - Turn 1 Raw JSON    |     Massive Raw Payload     +-------------------+
  |  - Turn 2 Raw JSON    |     (Appended directly
  |  - Turn 3 Raw JSON    |      to LLM context!)
  +-----------------------+
   * Bottleneck: Multi-hop joins happen in token space; hub entities quickly
     exhaust context limits and degrade attention.
```

And here is the RLM harness:

```
  +-----------------------+     1. Python Code Block    +-------------------+
  |      Primary LLM      | --------------------------> |  Persistent REPL  |
  |   (Reasoning Loop)    |                             |  (`repl_globals`) |
  |                       | <-------------------------- |                   |
  |  [Clean History]      |   4. Concise `stdout` only  |  Variables:       |
  |  - Compact Code       |      (Truncated <= 10k)     |  `films = [...]`  |
  |  - Printed Summaries  |                             |  `matches = ...`  |
  +-----------------------+                             +---------+---------+
                                                                  |
                         2. Graph Primitives & Sub-Calls          | 3. Grounding
              +---------------------------------------------------+    Check
              |                         |                         |
              v                         v                         v
    +-------------------+     +-------------------+     +-------------------+
    |   SQLite Cache    |     | Stateless Sub-LLM |     | `visited_entities`|
    | (`wikidata_api`,  |     |  (`llm_query` /   |     |   Set Verifier    |
    |  `sparql`, `llm`) |     |    `llm_map`)     |     |   on `FINAL()`    |
    +---------+---------+     +-------------------+     +-------------------+
              |
              | (Cache Miss)
              v
    +-------------------+
    | Live Wikidata API |
    | & SPARQL Endpoint |
    +-------------------+
```

The key change: a join between two result sets is now one line of Python. Suppose the model must find people who won both the Turing Award and the Nobel Memorial Prize in Economic Sciences. With JSON tools, it must read two long lists in its context and spot the shared IDs. In the REPL it writes:

```python
turing = set(edges("Q185667", "P166")["P166"])
nobel = set(edges("Q47170", "P166")["P166"])
both = turing & nobel
print(both)
```

Python computes `T \cap N`$ exactly, and the context grows by one short line of output, not two lists of a hundred IDs.

## The SQLite Cache

Every Wikidata API response, every SPARQL result, and every sub-LLM answer goes through a SQLite cache. The cache makes runs repeatable, lets you rerun a benchmark without the network, and spares the public Wikidata servers.

The class also defines `FinalAnswerException`, the signal that ends a run. We cover it with `FINAL` below.

```python
"""
rlm_wikidata_bidirectional.py
Recursive Language Model (RLM) Knowledge Graph Search Harness
with bidirectional traversal (Action API + SPARQL reverse edges)
and deterministic SQLite caching.
"""

import ast
import hashlib
import io
import json
import re
import sqlite3
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import requests
from openai import OpenAI


class FinalAnswerException(Exception):
    """Signal raised inside the REPL when the agent calls FINAL(answer)."""
    def __init__(self, answer: Any):
        self.answer = answer


class SQLiteCache:
    """
    Thread-safe persistent SQLite cache for Wikidata Action API,
    WDQS SPARQL queries, and stateless LLM sub-calls.
    """
    def __init__(self, db_path: str = "wikidata_rlm_cache.sqlite"):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            self.conn.commit()

    @staticmethod
    def make_key(namespace: str, data: Any) -> str:
        canonical = json.dumps(data, sort_keys=True, ensure_ascii=True)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    def get(self, namespace: str, key_data: Any) -> Optional[Any]:
        cache_key = self.make_key(namespace, key_data)
        with self._lock:
            cur = self.conn.execute(
                "SELECT payload FROM cache WHERE cache_key = ?", (cache_key,)
            )
            row = cur.fetchone()
        if row is not None:
            return json.loads(row[0])
        return None

    def set(self, namespace: str, key_data: Any, value: Any):
        cache_key = self.make_key(namespace, key_data)
        payload = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO cache (cache_key, namespace, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (cache_key, namespace, payload, time.time())
            )
            self.conn.commit()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT namespace, COUNT(*) FROM cache GROUP BY namespace"
            )
            return dict(cur.fetchall())

    def close(self):
        with self._lock:
            self.conn.close()
```

The key for each entry is a namespace plus a SHA-256 hash of the request, serialized as JSON with `sort_keys=True`. Two requests with the same parameters in a different order hash to the same key. The namespace (`wikidata_api`, `wikidata_sparql`, or `llm_subcall`) keeps the three kinds of data apart and lets `stats()` report a count for each.

The `check_same_thread=False` flag plus an explicit lock lets `llm_map` call the cache from worker threads. WAL journal mode lets readers and a writer work at the same time.

## The Wikidata Environment

`WikidataEnvironment` holds the functions the model may call. It talks to two Wikidata services:

- The **Action API** (`www.wikidata.org/w/api.php`) for search, labels, and the *outgoing* claims of an entity.
- The **SPARQL endpoint** (`query.wikidata.org/sparql`) for *incoming* edges: which entities point at this one?

The Action API can only answer "what does Q25188 point to?" It cannot answer "which films point to Q25188 as their director?" without a SPARQL query. Supporting both directions is why the module name ends in `bidirectional`.

### Rate Limiting and Retries

Wikidata asks clients to be polite, and the public servers return HTTP 429 when you are not. All network calls go through one class method that enforces a minimum gap between requests across all threads and backs off on 429 and 5xx errors:

```python
class WikidataEnvironment:
    """
    Provides bidirectional graph navigation primitives (outgoing `claims` via Action API
    and incoming `edges` via SPARQL), caches responses in SQLite, and enforces grounding.
    """
    API_URL = "https://www.wikidata.org/w/api.php"
    SPARQL_URL = "https://query.wikidata.org/sparql"
    HEADERS = {
        "User-Agent": "RLMBookExample/1.2 (Educational Python Agent)",
        "Accept": "application/sparql-results+json, application/json"
    }

    # Rate limiting: minimum seconds between network requests (shared across
    # all instances and threads), plus retry/backoff on 429 / 5xx responses.
    MIN_REQUEST_INTERVAL = 2.0
    MAX_RETRIES = 6
    _throttle_lock = threading.Lock()
    _last_request_time = 0.0

    @classmethod
    def _throttled_request(cls, url: str, params: Dict[str, Any], timeout: int) -> requests.Response:
        backoff = 5.0
        for attempt in range(cls.MAX_RETRIES + 1):
            with cls._throttle_lock:
                wait = cls.MIN_REQUEST_INTERVAL - (time.monotonic() - cls._last_request_time)
                if wait > 0:
                    time.sleep(wait)
                try:
                    resp = requests.get(url, params=params, headers=cls.HEADERS, timeout=timeout)
                finally:
                    cls._last_request_time = time.monotonic()

            if resp.status_code in (429, 500, 502, 503, 504) and attempt < cls.MAX_RETRIES:
                retry_after = resp.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else backoff
                except ValueError:
                    delay = backoff
                print(f"    [rate-limit] HTTP {resp.status_code}; sleeping {delay:.0f}s "
                      f"(retry {attempt + 1}/{cls.MAX_RETRIES})", file=sys.stderr)
                time.sleep(delay)
                backoff = min(backoff * 2, 120.0)
                continue

            resp.raise_for_status()
            return resp
        raise RuntimeError("unreachable")
```

The lock and `_last_request_time` live on the class, not the instance, so every environment in the process shares one request budget. The method honors a `Retry-After` header when the server sends one and otherwise doubles the wait, capped at two minutes. Wikidata requires a descriptive `User-Agent`; change it to name your own project and contact address.

### Cached Fetches

The constructor sets up counters and the `visited_entities` set. `_get` and `_sparql` check the cache first, refuse to touch the network in offline mode, and store each fresh response:

```python
    def __init__(self, client: OpenAI, model: str, cache: SQLiteCache, offline_only: bool = False):
        self.client = client
        self.model = model
        self.cache = cache
        self.offline_only = offline_only

        self.visited_entities: Set[str] = set()
        self.call_log: List[str] = []
        self.api_cache_hits = 0
        self.api_network_calls = 0
        self.subcall_count = 0
        self.subcall_cache_hits = 0

    def _get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["format"] = "json"
        cached = self.cache.get("wikidata_api", params)
        if cached is not None:
            self.api_cache_hits += 1
            return cached

        if self.offline_only:
            raise RuntimeError(f"Offline mode enabled and cache miss for params: {params}")

        self.api_network_calls += 1
        resp = self._throttled_request(self.API_URL, params, timeout=15)
        data = resp.json()
        self.cache.set("wikidata_api", params, data)
        return data

    def _sparql(self, query: str) -> Dict[str, Any]:
        cached = self.cache.get("wikidata_sparql", {"query": query})
        if cached is not None:
            self.api_cache_hits += 1
            return cached

        if self.offline_only:
            raise RuntimeError(f"Offline mode enabled and SPARQL cache miss for query: {query}")

        self.api_network_calls += 1
        resp = self._throttled_request(
            self.SPARQL_URL,
            {"query": query, "format": "json"},
            timeout=25
        )
        data = resp.json()
        self.cache.set("wikidata_sparql", {"query": query}, data)
        return data
```

### Search and Metadata Primitives

Four small functions find entities and properties by name and fetch labels:

```python
    def search_entity(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """Search Wikidata entities by label/alias."""
        self.call_log.append(f"search_entity({query!r})")
        data = self._get({
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "limit": limit,
            "type": "item"
        })
        results = []
        for item in data.get("search", []):
            qid = item["id"]
            self.visited_entities.add(qid)
            results.append({
                "id": qid,
                "label": item.get("label", ""),
                "description": item.get("description", "")
            })
        return results

    def search_property(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """Search Wikidata properties (P-IDs) by label."""
        self.call_log.append(f"search_property({query!r})")
        data = self._get({
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "limit": limit,
            "type": "property"
        })
        return [
            {
                "id": item["id"],
                "label": item.get("label", ""),
                "description": item.get("description", "")
            }
            for item in data.get("search", [])
        ]

    def describe(self, entity_id: str) -> Dict[str, str]:
        """Return the English label and description for a Q-ID or P-ID."""
        self.call_log.append(f"describe({entity_id!r})")
        self.visited_entities.add(entity_id)
        data = self._get({
            "action": "wbgetentities",
            "ids": entity_id,
            "props": "labels|descriptions",
            "languages": "en"
        })
        ent = data.get("entities", {}).get(entity_id, {})
        return {
            "id": entity_id,
            "label": ent.get("labels", {}).get("en", {}).get("value", entity_id),
            "description": ent.get("descriptions", {}).get("en", {}).get("value", "")
        }

    def label(self, entity_id: str) -> str:
        """Return just the English label for an entity or property ID."""
        return self.describe(entity_id)["label"]
```

Note that every function which sees a Q-ID adds it to `visited_entities`. That set drives the grounding check at the end of a run.

### Forward and Reverse Traversal

`claims` follows outgoing edges. Raw Wikidata statements nest the value deep inside a `mainsnak` structure, with a different shape for each data type. `claims` flattens all of that to a plain dict that maps a property ID to a list of values:

```python
    def claims(self, entity_id: str, property_id: str = None) -> Dict[str, List[Any]]:
        """
        Fetch OUTGOING statements (claims) from `entity_id`.
        Optionally filter by `property_id` (e.g., 'P31').
        """
        self.call_log.append(f"claims({entity_id!r}, {property_id!r})")
        self.visited_entities.add(entity_id)
        data = self._get({
            "action": "wbgetentities",
            "ids": entity_id,
            "props": "claims"
        })
        raw_claims = data.get("entities", {}).get(entity_id, {}).get("claims", {})
        if property_id:
            raw_claims = {property_id: raw_claims.get(property_id, [])}

        parsed: Dict[str, List[Any]] = {}
        for pid, stmt_list in raw_claims.items():
            values = []
            for stmt in stmt_list:
                mainsnak = stmt.get("mainsnak", {})
                datavalue = mainsnak.get("datavalue", {})
                dtype = datavalue.get("type")
                val = datavalue.get("value")

                if dtype == "wikibase-entityid" and isinstance(val, dict):
                    target_id = val.get("id")
                    if target_id:
                        self.visited_entities.add(target_id)
                        values.append(target_id)
                elif dtype == "time" and isinstance(val, dict):
                    values.append(val.get("time"))
                elif dtype == "quantity" and isinstance(val, dict):
                    values.append(val.get("amount"))
                elif dtype == "monolingualtext" and isinstance(val, dict):
                    values.append(val.get("text"))
                elif val is not None:
                    values.append(val)
            if values:
                parsed[pid] = values
        return parsed
```

Entity values become bare Q-IDs, times become strings such as `+2016-11-11T00:00:00Z`, and quantities become their amount. Since `claims(qid)` and `claims(qid, pid)` send the same Action API request, the second call is always a cache hit.

`edges` follows incoming edges with SPARQL. With a property it asks "which subjects have `property_id` pointing to `entity_id`?" Without one it returns every incoming edge, grouped by property:

```python
    def edges(
        self,
        entity_id: str,
        property_id: str = None,
        limit: int = 100
    ) -> Dict[str, List[str]]:
        """
        Fetch INCOMING links (reverse edges) where other entities point to `entity_id`.
        Returns a dict mapping {property_id: [subject_QIDs]}.
        If `property_id` is provided (e.g., 'P57'), filters only incoming links for that property.
        """
        if not re.match(r"^Q\d+$", entity_id):
            raise ValueError(f"Invalid entity_id {entity_id!r}; expected Q-ID.")
        if property_id is not None and not re.match(r"^P\d+$", property_id):
            raise ValueError(f"Invalid property_id {property_id!r}; expected P-ID.")

        self.call_log.append(f"edges({entity_id!r}, {property_id!r}, limit={limit})")
        self.visited_entities.add(entity_id)

        if property_id:
            sparql_query = f"""
            SELECT ?subject WHERE {{
              ?subject wdt:{property_id} wd:{entity_id} .
              FILTER(STRSTARTS(STR(?subject), "http://www.wikidata.org/entity/Q"))
            }}
            LIMIT {int(limit)}
            """
            data = self._sparql(sparql_query)
            subjects = []
            for row in data.get("results", {}).get("bindings", []):
                uri = row["subject"]["value"]
                qid = uri.rsplit("/", 1)[-1]
                self.visited_entities.add(qid)
                subjects.append(qid)
            return {property_id: subjects} if subjects else {}
        else:
            sparql_query = f"""
            SELECT ?prop ?subject WHERE {{
              ?subject ?p wd:{entity_id} .
              ?prop wikibase:directClaim ?p .
              FILTER(STRSTARTS(STR(?subject), "http://www.wikidata.org/entity/Q"))
            }}
            LIMIT {int(limit)}
            """
            data = self._sparql(sparql_query)
            grouped: Dict[str, List[str]] = {}
            for row in data.get("results", {}).get("bindings", []):
                pid = row["prop"]["value"].rsplit("/", 1)[-1]
                qid = row["subject"]["value"].rsplit("/", 1)[-1]
                self.visited_entities.add(qid)
                grouped.setdefault(pid, []).append(qid)
            return grouped

    def has(self, entity_id: str, property_id: str, target_id: str) -> bool:
        """Check whether `entity_id` has an outgoing claim `property_id` pointing to `target_id`."""
        prop_claims = self.claims(entity_id, property_id).get(property_id, [])
        return target_id in prop_claims
```

The model writes the arguments to `edges`, and those arguments go straight into a SPARQL string. The two regular expressions at the top make sure the IDs are plain `Q` and `P` numbers, so model-written code cannot inject extra SPARQL. The `wikibase:directClaim` pattern in the second query maps a direct-claim predicate such as `wdt:P57` back to its property entity, so we can report the property ID. The `FILTER` drops subjects that are not items, such as lexemes.

### Recursive Sub-Calls

`llm_query` is the recursive part of the harness. The model passes some text and an instruction, and a fresh LLM call with no chat history answers it. The model can use this to classify film genres from descriptions, pick the right entity from a list of ten, or pull a year out of a block of text, without pulling that text into its own context:

```python
    def llm_query(self, text: str, instruction: str) -> str:
        """
        Delegate reading/filtering of retrieved evidence to a stateless sub-call.
        Caches results by (model, instruction, text).
        """
        self.subcall_count += 1
        key_payload = {"model": self.model, "instruction": instruction, "text": text}
        cached = self.cache.get("llm_subcall", key_payload)
        if cached is not None:
            self.subcall_cache_hits += 1
            return cached

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise analytical assistant. Follow the instruction strictly using only the provided text."
                },
                {
                    "role": "user",
                    "content": f"Instruction: {instruction}\n\nEvidence:\n{text}"
                }
            ],
            temperature=0.0
        )
        answer = resp.choices[0].message.content.strip()
        self.cache.set("llm_subcall", key_payload, answer)
        return answer

    def llm_map(self, items: List[str], instruction: str, max_workers: int = 4) -> List[str]:
        """Run `llm_query` concurrently across a list of text items."""
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(lambda item: self.llm_query(str(item), instruction), items))
```

The sub-call uses temperature 0 and caches on model, instruction, and text, so a rerun returns the same answer. `llm_map` fans one instruction out over a list with a thread pool. Here the sub-LLM is the same model as the main agent. You could point it at a smaller, cheaper model.

### Grounding and the Namespace

An agent can make up a plausible Q-ID from memory. `FINAL` blocks that. It finds every Q-ID in the answer and rejects the answer if any of them never came back from a graph call during this run:

```python
    def FINAL(self, answer: Any):
        """
        Terminate the search and return `answer`.
        Verifies grounding if the answer is a Q-ID or contains Q-IDs in a list/dict.
        """
        qids_to_check = re.findall(r"\bQ\d+\b", str(answer))
        unvisited = [q for q in qids_to_check if q not in self.visited_entities]
        if unvisited:
            raise ValueError(
                f"Grounding Error: Entity ID(s) {unvisited} were never visited or retrieved "
                f"from the graph during this session. Verify them using describe(), claims(), or edges() first."
            )
        raise FinalAnswerException(answer)

    def bind_namespace(self) -> Dict[str, Any]:
        """Return the persistent dictionary of globals exposed to the agent's REPL."""
        return {
            "search_entity": self.search_entity,
            "search_property": self.search_property,
            "describe": self.describe,
            "label": self.label,
            "claims": self.claims,
            "edges": self.edges,
            "has": self.has,
            "llm_query": self.llm_query,
            "llm_map": self.llm_map,
            "FINAL": self.FINAL,
        }
```

A grounding failure raises a normal `ValueError`. The harness catches it like any other exception and shows the traceback to the model, which can then verify the ID and try again. A good answer raises `FinalAnswerException`, which unwinds out of the model's code and ends the loop.

`bind_namespace` returns the dict that becomes the REPL's globals. The model's code can see these ten functions and nothing else from the harness. Variables the model creates go into the same dict, which is how they persist across turns.

W> The REPL runs model-written code with `exec` in your own Python process. The namespace limits what the model *sees*, not what it *can do*: code can still `import os`. Run this example on a machine and account you trust, or move the REPL into a container or subprocess before you use it with untrusted models or prompts.

## The System Prompt

The prompt lists the functions with their return shapes and gives three rules of strategy:

```python
SYSTEM_PROMPT = """You are a Recursive Language Model (RLM) agent exploring Wikidata through a persistent Python REPL.
Reply with ONE Python code block (```python ... ```) per turn.
Variables you define persist across turns. Do not re-fetch data you have already stored in variables.

Available functions in your global namespace:
- search_entity(query: str, limit: int = 5) -> list[dict]  # [{'id': 'Q...', 'label': '...', 'description': '...'}]
- search_property(query: str, limit: int = 5) -> list[dict] # [{'id': 'P...', 'label': '...', 'description': '...'}]
- describe(entity_id: str) -> dict                         # {'id': 'Q...', 'label': '...', 'description': '...'}
- label(entity_id: str) -> str                             # English label for Q-ID or P-ID
- claims(entity_id: str, property_id: str = None) -> dict  # OUTGOING statements: {pid: [values_or_target_QIDs]}
- edges(entity_id: str, property_id: str = None, limit: int = 100) -> dict # INCOMING reverse links: {pid: [subject_QIDs]}
- has(entity_id: str, property_id: str, target_id: str) -> bool
- llm_query(text: str, instruction: str) -> str            # Stateless LLM sub-call for semantic filtering
- llm_map(items: list[str], instruction: str) -> list[str] # Parallel stateless sub-calls
- FINAL(answer: Any)                                       # Call when certain to finish and return the answer

Strategy:
1. Use `claims(qid)` for outgoing edges and `edges(qid, pid)` for incoming reverse relationships (e.g. finding works by an author/director or items located in a place).
2. Keep printed output concise; inspect slices or summaries rather than dumping huge dicts.
3. The question has an answer. Do not stop until you find and verify it, then call FINAL(answer).
"""
```

The prompt shows the return shape of each function as a comment. Models write better code when they know the shape of the data before they call a function.

## The Harness Loop

`RLMHarness` owns the turn loop. Its constructor takes the limits that bound a run: the number of turns, the size of REPL output shown to the model, and the token budget for each reply.

```python
class RLMHarness:
    """Manages the persistent Python namespace, SQLite cache, and multi-turn loop."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        cache: SQLiteCache,
        max_turns: int = 12,
        max_output_chars: int = 10000,
        offline_only: bool = False,
        max_tokens: int = 2048
    ):
        self.client = client
        self.model = model
        self.cache = cache
        self.max_turns = max_turns
        self.max_output_chars = max_output_chars
        self.offline_only = offline_only
        self.max_tokens = max_tokens
```

### Cleaning Up Model Output

Small local models rarely reply with one clean code block. Some wrap their reply in `<think>` tags. Some get cut off at the token limit before they close the fence. Some fall back to the native tool-call format they learned in training. Some get stuck and repeat one line over and over. The next three methods turn all of these into runnable Python:

```python
    @staticmethod
    def _collapse_repeats(code: str) -> tuple:
        """Collapse runs of identical consecutive non-blank lines (degenerate LLM loops). Returns (code, n_removed)."""
        out: List[str] = []
        removed = 0
        for line in code.splitlines():
            if out and line.strip() and line == out[-1]:
                removed += 1
                continue
            out.append(line)
        return "\n".join(out), removed

    @staticmethod
    def extract_code(response_text: str) -> str:
        text = response_text or ""
        # Drop reasoning blocks emitted by "thinking" models.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)  # unopened </think>

        match = re.search(r"```(?:python|py)?\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Unclosed fence (e.g. reply truncated at max_tokens).
        match = re.search(r"```(?:python|py)?[ \t]*\n(.*)", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: native tool-call syntax, e.g. <tool_call>...</tool_call>
        if "<tool_call>" in text or "</tool_call>" in text:
            bodies = re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text, re.DOTALL)
            lines = []
            for body in bodies or [text]:
                body = re.sub(r"</?tool_call>", "", body).strip()
                lines.append(RLMHarness._tool_call_to_python(body))
            return "\n".join(l for l in lines if l)

        return text.strip()

    @staticmethod
    def _tool_call_to_python(body: str) -> str:
        """Convert a JSON tool call ({"name":..., "arguments":{...}}) to a Python call; pass raw code through."""
        try:
            obj = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return body
        if isinstance(obj, dict) and "name" in obj:
            args = obj.get("arguments") or obj.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            kwargs = ", ".join(f"{k}={v!r}" for k, v in args.items())
            call = f"{obj['name']}({kwargs})"
            return call if obj["name"] == "FINAL" else f"print({call})"
        return body
```

`extract_code` tries the clean case first, then an unclosed fence, then tool-call tags. If nothing matches, it treats the whole reply as code and lets the Python parser judge it. `_tool_call_to_python` turns `{"name": "claims", "arguments": {"entity_id": "Q42"}}` into `print(claims(entity_id='Q42'))`. It wraps the call in `print` so the model sees the result, except for `FINAL`, which returns nothing.

### Notebook-Style Echo

Models often write a bare expression such as `claims("Q42")` and expect to see the result, as in a Jupyter cell. Plain `exec` throws that value away. `_exec_with_echo` parses the code, pops the last statement if it is an expression, runs the rest, and then evaluates and prints the popped expression:

```python
    @staticmethod
    def _exec_with_echo(code: str, repl_globals: Dict[str, Any]):
        """Exec `code`; if the final statement is an expression, print its value (like a notebook cell)."""
        tree = ast.parse(code, mode="exec")
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(tree.body.pop().value)
        exec(compile(tree, "<repl>", "exec"), repl_globals)
        if last_expr is not None:
            value = eval(compile(last_expr, "<repl>", "eval"), repl_globals)
            if value is not None:
                print(repr(value))
```

Both `exec` and `eval` receive the same `repl_globals` dict, so names defined in one turn exist in the next.

### The Turn Loop

`run` ties the parts together. Each turn it asks the model for code, cleans the code, runs it with `stdout` and `stderr` captured, and sends the captured text back as the next user message:

```python
    def run(self, question: str) -> Dict[str, Any]:
        env = WikidataEnvironment(
            self.client, self.model, cache=self.cache, offline_only=self.offline_only
        )
        repl_globals = env.bind_namespace()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"}
        ]

        for turn in range(1, self.max_turns + 1):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
            choice = resp.choices[0]
            assistant_msg = choice.message.content or ""
            truncated = choice.finish_reason == "length"

            code = self.extract_code(assistant_msg)
            code, repeats = self._collapse_repeats(code)
            if truncated:
                # The last line may have been cut off mid-generation; drop it if it breaks parsing.
                try:
                    ast.parse(code)
                except SyntaxError:
                    code = "\n".join(code.splitlines()[:-1]).rstrip()

            # Store a cleaned, canonical code block in history (not the raw runaway text),
            # which also reinforces the expected ```python``` format.
            messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            print(f"\n--- [Turn {turn} Code] ---\n{code}")

            notes = []
            if repeats:
                notes.append(f"[Harness] Removed {repeats} duplicated consecutive line(s) from your reply. "
                             "Do not repeat the same call; inspect the result and move on.")
            if truncated:
                notes.append(f"[Harness] Your reply hit the {self.max_tokens}-token limit and was truncated. "
                             "Keep replies short: one small code block per turn.")

            stdout_buf = io.StringIO()
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = stdout_buf

            try:
                self._exec_with_echo(code, repl_globals)
                output = stdout_buf.getvalue()
                if not output.strip():
                    output = "(Executed successfully with no printed output. Use print() to inspect variables.)"
            except FinalAnswerException as fin:
                sys.stdout, sys.stderr = old_stdout, old_stderr
                print(f"\n=== FINAL ANSWER: {fin.answer} ===")
                return {
                    "answer": fin.answer,
                    "turns": turn,
                    "graph_calls": len(env.call_log),
                    "api_cache_hits": env.api_cache_hits,
                    "api_network_calls": env.api_network_calls,
                    "subcalls": env.subcall_count,
                    "subcall_cache_hits": env.subcall_cache_hits,
                    "visited_entities": len(env.visited_entities),
                }
            except SyntaxError:
                output = stdout_buf.getvalue() + "\n" + traceback.format_exc() + (
                    "\nHint: Reply with plain Python inside a single ```python ... ``` block. "
                    "Do not use <tool_call> tags or JSON; call the functions directly, e.g.\n"
                    "```python\nresults = search_entity(\"Guggenheim Museum Bilbao\")\nprint(results)\n```"
                )
            except Exception:
                output = stdout_buf.getvalue() + "\n" + traceback.format_exc()
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

            if len(output) > self.max_output_chars:
                output = output[:self.max_output_chars] + "\n...[Output truncated at 10,000 chars]"

            if notes:
                output = output.rstrip() + "\n\n" + "\n".join(notes)

            print(f"--- [Turn {turn} REPL Output] ---\n{output.strip()}")
            messages.append({"role": "user", "content": f"REPL Output:\n{output}"})

        raise RuntimeError(f"Agent exceeded max_turns ({self.max_turns}) without calling FINAL().")
```

A few choices here matter more than they look:

- **The history holds the cleaned code, not the raw reply.** If the model rambled, repeated itself, or used tool-call tags, the next turn sees a tidy `python` block instead. Each turn then shows the model the format we want.
- **Errors are feedback, not failures.** A traceback goes back to the model as REPL output. A `SyntaxError` also gets a hint with a correct example, since small models often fall back to the wrong format.
- **Output has a hard cap.** The model sees at most 10,000 characters per turn. That cap bounds context growth, the problem this whole design exists to solve. Data the model does not print stays in Python variables, where the next turn can still use it.
- **The harness tells the model what it fixed.** Notes about removed repeats and truncated replies ride along with the output so the model can change its behavior.

### Running One Question

The module's `__main__` block runs one question against a local Ollama server:

```python
if __name__ == "__main__":
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="local-key"
    )

    cache = SQLiteCache("wikidata_rlm_cache.sqlite")
    try:
        harness = RLMHarness(
            client=client,
            model="qwen2.5-coder:32b",
            cache=cache
        )

        # Multi-hop query requiring BOTH reverse edges (`edges`) and forward claims (`claims`):
        # 1. Reverse edge: Find science fiction films directed by (P57) Denis Villeneuve (Q25188)
        # 2. Forward claim: Check composer (P86) and publication date (P577)
        sample_question = (
            "Which science fiction film directed by Denis Villeneuve and released in 2016 "
            "had its musical score composed by Jóhann Jóhannsson? Return a dict with its Q-ID and label."
        )
        result = harness.run(sample_question)
        print("\nTrajectory Summary:", json.dumps(result, indent=2))
        print("Total SQLite Cache Entries:", cache.stats())
    finally:
        cache.close()
```

To run it, pull the model in Ollama and run the module with uv:

```
ollama pull qwen2.5-coder:32b
uv run rlm_wikidata_bidirectional.py
```

## The Benchmark Suite

`benchmark_suite.py` runs four questions, each built to test one kind of graph reasoning:

| Task | Pattern | What the model must do |
| --- | --- | --- |
| `task_1_forward_2hop` | `claims` then `claims` | Museum to architect to birthplace |
| `task_2_reverse_edge_join` | `edges` plus `claims` | Films by a director, filtered by composer and year |
| `task_3_set_intersection` | `edges` and `edges` | People who won two given awards |
| `task_4_notable_work_filter` | `claims` plus `describe` | Pick one book from an author's notable works |

Each task is a dataclass with the question, the expected Q-ID, keywords that must appear in the answer, and a *seed function*:

```python
@dataclass
class BenchmarkTask:
    task_id: str
    category: str
    question: str
    expected_qid: str
    expected_keywords: List[str]
    seed_fn: Callable[[WikidataEnvironment], None]
```

A seed function walks the expected path through the graph with no LLM involved, so every lookup lands in the cache. Here is the seeder for the set intersection task:

```python
def seed_turing_nobel(env: WikidataEnvironment):
    env.search_entity("Turing Award")
    env.search_entity("Nobel Memorial Prize in Economic Sciences")
    env.search_entity("Nobel Prize in Economics")
    env.search_property("award received")
    turing_winners = set(env.edges("Q185667", "P166", limit=100).get("P166", []))
    nobel_winners = set(env.edges("Q47170", "P166", limit=100).get("P166", []))
    for qid in turing_winners & nobel_winners:
        env.describe(qid)
        env.claims(qid, "P166")
```

After seeding, you can run models in `--offline` mode with no network access, so every model gets the same data. The answer check is a loose string match on the serialized answer, so `FINAL` may return a string, dict, or list:

```python
def verify_answer(answer: Any, task: BenchmarkTask) -> bool:
    """Check whether the agent's FINAL(answer) contains the target Q-ID and keywords."""
    serialized = json.dumps(answer, ensure_ascii=False) if isinstance(answer, (dict, list)) else str(answer)
    serialized_lower = serialized.lower()
    has_qid = task.expected_qid.lower() in serialized_lower
    has_keywords = all(kw.lower() in serialized_lower for kw in task.expected_keywords)
    return has_qid and has_keywords
```

`run_benchmark` runs each task through one `RLMHarness`, catches failures, and prints a Markdown table of turns, graph calls, cache hits, network calls, sub-calls, and time. The command line flags pick the endpoint and model:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--base-url` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `--api-key` | `local-key` | Key for the endpoint |
| `--model` | `qwen2.5-coder:32b` | Model name |
| `--db` | `wikidata_rlm_cache.sqlite` | Cache file |
| `--seed` | off | Seed the cache, then run |
| `--seed-only` | off | Seed the cache and exit |
| `--offline` | off | Use only the cache; no Wikidata requests |

The `Makefile` wraps the common cases:

```
clean:
	rm -rf *~ *.sqlite __pycache__ .venv

create_db:
	uv run benchmark_suite.py --seed-only

run_omlx:
	uv run benchmark_suite.py --base-url http://127.0.0.1:8000/v1 --model Laguna-XS-2.1-6bit
```

## A Sample Run with a Small Local Model

`SAMPLE_OUTPUT.txt` holds a full run of `make run_omlx`, which serves the small `Laguna-XS-2.1-6bit` model through oMLX. The run is worth a close read because it shows both what the design buys you and where a small model still struggles.

### What Went Well

Task 4 is the design working as intended. The model fetched the fifteen notable works, then stored the list in a variable and printed one line per work:

```
--- [Turn 4 REPL Output] ---
Q116546: Earthsea series - fantasy fiction series by Ursula K. Le Guin, including 5 novels 1968 to 2001 among other publications
Q2008211: The Left Hand of Darkness - 1969 novel by Ursula K. Le Guin
Q467720: The Dispossessed - 1974 novel by Ursula K. Le Guin
...
```

It saw that the list holds the Earthsea *series*, not the first novel, so it followed the series' `P527` ("has part") claims one more hop and found the book:

```
--- [Turn 6 REPL Output] ---
P527 (has part) items:
  Q1771810: A Wizard of Earthsea - 1968 novel by Ursula K. Le Guin
  Q2070380: The Tombs of Atuan - 1971 novel by Ursula K. Le Guin
...
--- [Turn 8 Code] ---
FINAL({"qid": "Q1771810", "label": "A Wizard of Earthsea", "earliest_publication_year": 1968})
```

In Task 2 the model first used the wrong property for composer (`P571`, which is "inception"). Its own code printed `No composer found with P571`, so it called `search_property("composer")`, found `P86`, and confirmed the answer:

```
=== Arrival (Q20382729) Details ===
Label: Arrival
Director: Denis Villeneuve (Q548823)
Composer: Jóhann Jóhannsson (Q428347)
Release date: ['+2016-11-11T00:00:00Z', '+2016-11-10T00:00:00Z', '+2016-11-24T00:00:00Z']
```

Both of these answers are correct.

### What Went Wrong

The final table reports zero of four solved:

```
| Task ID | Passed | Turns | Graph Calls | Cache Hits | Net Calls | Sub-Calls | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `task_1_forward_2hop` | **FAIL** | 10 | 0 | 0 | 0 | 0 | 23.09s |
| `task_2_reverse_edge_join` | **FAIL** | 9 | 92 | 5 | 87 | 0 | 575.74s |
| `task_3_set_intersection` | **FAIL** | 10 | 0 | 0 | 0 | 0 | 47.22s |
| `task_4_notable_work_filter` | **FAIL** | 8 | 28 | 19 | 9 | 0 | 42.5s |
```

There are four separate lessons in that table.

**1. The benchmark's expected IDs do not match live Wikidata.** The model returned `Q20382729` for *Arrival* and `Q1771810` for *A Wizard of Earthsea*. Both IDs came from the live graph, and both passed the grounding check. The benchmark expects `Q20168408` and `Q320435`, so `verify_answer` scored two correct answers as failures. The seeders have the same problem: they seed `Q170040` for the Guggenheim Museum Bilbao, `Q6722` for Frank Gehry, and `Q25188` for Denis Villeneuve, while live search in this run returned `Q179199`, `Q180374`, and `Q548823`. An offline run would therefore miss the cache on the paths the model actually takes. The paper avoids this class of bug by building its questions on a frozen Wikidata snapshot. The lesson for your own benchmarks: derive expected IDs from the same data source the agent queries, and check them with a script, not from memory.

**2. Small models fight the output format.** Many turns failed with a `SyntaxError` because the model ran several calls together on one line, as in `describe("Q185667")describe("Q47170")`, or left a stray fence or a `<|tool_call_begin|>` token in its code. The hint in the error message fixed the problem each time, but each fix cost a turn. Task 3 spent five of its ten turns on syntax errors and ran out of turns.

**3. Printed loops can make many network calls.** In Task 2 the model looped over all of Denis Villeneuve's claims and called `label(pid)` for each property. That one turn made dozens of throttled requests and drew repeated HTTP 429 responses, then hit a read timeout. The task took almost ten minutes, with 87 network calls. A second run would be fast, since every one of those calls is now in the cache.

**4. The model never used `llm_query`.** The Sub-Calls column is zero for all tasks. These four questions are simple enough that plain graph calls suffice, so this is not a failure. Questions that need judgment over long text, such as "which of these 40 films is science fiction, based on its description?", give the sub-call a real job.

Also note that the zeros in the Graph Calls column for Tasks 1 and 3 come from the error path in `run_benchmark`, which records zeros when `harness.run` raises. They do not mean the model made no calls.

Task 1 shows one more pattern. By turn 9 the model had found Toronto (`Q172`) as Frank Gehry's birthplace, but it went on to look up the `place of birth` property instead of calling `FINAL`, and ran out of turns. A larger turn budget, or a prompt that tells the model to call `FINAL` as soon as it has a verified answer, would likely fix this.

## Wrap Up

The RLM harness changes where an agent's working memory lives. With JSON tool calls, every result piles up in the context, and the model must join and filter in token space. With a persistent REPL, results live in Python variables, joins and filters are exact Python operations, and the model reads only short printed summaries. Sub-calls let the model hand off reading and judging tasks to a fresh LLM call that sees only the text it needs.

The rest of the code supports that core idea:

1. **A content-keyed SQLite cache** makes runs repeatable, makes reruns fast, and keeps load off public servers.
2. **Bidirectional primitives** give the model both outgoing claims (Action API) and incoming edges (SPARQL), which multi-hop questions need.
3. **A grounding check in `FINAL`** rejects Q-IDs the model did not retrieve during the run.
4. **Defensive parsing** turns the messy output of small models into runnable code and tells the model what went wrong.

The sample run also shows the limits. A small local model can follow the method and find correct answers, but it wastes turns on format errors, and a benchmark is only as good as its ground truth.

## Exercises for the Reader

1. **Fix the benchmark ground truth.** Write a script that resolves each expected entity by name with `search_entity`, prints the live Q-IDs, and updates `BENCHMARK_TASKS` and the seed functions. Rerun `make create_db` and then the benchmark with `--offline`.
2. **Sandbox the REPL.** Move `_exec_with_echo` into a subprocess that holds `repl_globals` and talks to the harness over a pipe. Expose the graph primitives as stubs that send requests back to the parent process. What does this cost in speed?
3. **Add a turn-budget note.** Tell the model in each REPL output how many turns remain. Does Task 1 now call `FINAL` in time?
4. **Use a smaller sub-model.** Add a `sub_model` argument to `WikidataEnvironment` so `llm_query` can use a cheaper model than the main agent. Write a question that needs `llm_map`, for example one that filters a director's films by genre from their descriptions, and compare results.
5. **Compare with JSON tool calling.** Build a baseline agent that exposes the same ten functions as JSON tools and appends raw results to the chat. Run both agents on the four tasks with the same model and record turns, prompt tokens, and pass rate.
6. **Batch label lookups.** The Action API `wbgetentities` call accepts up to 50 IDs separated by `|`. Add a `labels(ids)` primitive that fetches many labels in one request, and document it in the system prompt. How much faster is Task 2 with a cold cache?
