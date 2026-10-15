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

    # --- Search & Metadata Primitives ---

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

    # --- Forward & Reverse Graph Traversal Primitives ---

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

    # --- Cached Recursive Sub-Call Primitives ---

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

    # --- Termination & Grounding ---

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

