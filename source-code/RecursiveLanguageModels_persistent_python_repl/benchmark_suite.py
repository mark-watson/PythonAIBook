"""
benchmark_suite.py
Pre-seeds the SQLite cache with canonical multi-hop Wikidata paths and evaluates
an RLM agent across 4 representative Knowledge Graph reasoning tasks.
"""

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List
from openai import OpenAI

from rlm_wikidata_bidirectional import (
    RLMHarness,
    SQLiteCache,
    WikidataEnvironment,
)


@dataclass
class BenchmarkTask:
    task_id: str
    category: str
    question: str
    expected_qid: str
    expected_keywords: List[str]
    seed_fn: Callable[[WikidataEnvironment], None]


# --- Deterministic Seeder Functions ---
# These pre-populate the SQLite cache with the exact entities, properties,
# forward claims, and reverse SPARQL edges needed to solve each task.

def seed_guggenheim(env: WikidataEnvironment):
    env.search_entity("Guggenheim Museum Bilbao")
    env.search_property("architect")
    env.search_property("place of birth")
    env.claims("Q170040")            # Guggenheim Museum Bilbao claims
    env.claims("Q170040", "P84")     # architect -> Q6722 (Frank Gehry)
    env.describe("Q6722")
    env.claims("Q6722")              # Frank Gehry claims
    env.claims("Q6722", "P19")       # place of birth -> Q172 (Toronto)
    env.describe("Q172")


def seed_villeneuve(env: WikidataEnvironment):
    env.search_entity("Denis Villeneuve")
    env.search_entity("Jóhann Jóhannsson")
    env.search_property("director")
    env.search_property("composer")
    env.search_property("publication date")
    films = env.edges("Q25188", "P57", limit=100).get("P57", [])
    for qid in films[:25]:
        env.describe(qid)
        env.claims(qid, "P86")       # composer
        env.claims(qid, "P577")      # publication date
    env.claims("Q20168408")          # Arrival (full claims)


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


def seed_leguin(env: WikidataEnvironment):
    env.search_entity("Ursula K. Le Guin")
    env.search_property("notable work")
    env.search_property("publication date")
    works = env.claims("Q181659", "P800").get("P800", [])
    env.claims("Q181659")
    for qid in works:
        env.describe(qid)
        env.claims(qid, "P577")


BENCHMARK_TASKS: List[BenchmarkTask] = [
    BenchmarkTask(
        task_id="task_1_forward_2hop",
        category="Forward Multi-Hop (claims -> claims)",
        question=(
            "Find the Wikidata Q-ID and English label of the birth city of the "
            "architect who designed the Guggenheim Museum Bilbao."
        ),
        expected_qid="Q172",
        expected_keywords=["Toronto"],
        seed_fn=seed_guggenheim,
    ),
    BenchmarkTask(
        task_id="task_2_reverse_edge_join",
        category="Bidirectional Join (edges + claims)",
        question=(
            "Which science fiction film directed by Denis Villeneuve and released in 2016 "
            "had its musical score composed by Jóhann Jóhannsson? Return both its Q-ID and English label."
        ),
        expected_qid="Q20168408",
        expected_keywords=["Arrival"],
        seed_fn=seed_villeneuve,
    ),
    BenchmarkTask(
        task_id="task_3_set_intersection",
        category="Programmatic Set Intersection (edges & edges)",
        question=(
            "Identify the computer scientist and economist who won both the Turing Award (Q185667) "
            "and the Nobel Memorial Prize in Economic Sciences (Q47170). Return their Q-ID and English label."
        ),
        expected_qid="Q20955",
        expected_keywords=["Herbert", "Simon"],
        seed_fn=seed_turing_nobel,
    ),
    BenchmarkTask(
        task_id="task_4_notable_work_filter",
        category="One-to-Many Filtering (claims + describe)",
        question=(
            "Among the notable works (P800) listed on Ursula K. Le Guin's Wikidata entity (Q181659), "
            "which novel is the first book set in the Earthsea archipelago? "
            "Return its Q-ID, English label, and earliest publication year."
        ),
        expected_qid="Q320435",
        expected_keywords=["Wizard of Earthsea", "1968"],
        seed_fn=seed_leguin,
    ),
]


def seed_all_tasks(client: OpenAI, model: str, cache: SQLiteCache):
    """Pre-populate the SQLite cache with all canonical graph lookups."""
    print("Seeding SQLite cache with canonical multi-hop graph paths...")
    env = WikidataEnvironment(client=client, model=model, cache=cache, offline_only=False)
    for task in BENCHMARK_TASKS:
        print(f"  -> Seeding {task.task_id} ({task.category})...")
        task.seed_fn(env)
    print(
        f"Seeding complete. Network calls: {env.api_network_calls}, "
        f"Cache hits: {env.api_cache_hits}, Total DB stats: {cache.stats()}\n"
    )


def verify_answer(answer: Any, task: BenchmarkTask) -> bool:
    """Check whether the agent's FINAL(answer) contains the target Q-ID and keywords."""
    serialized = json.dumps(answer, ensure_ascii=False) if isinstance(answer, (dict, list)) else str(answer)
    serialized_lower = serialized.lower()
    has_qid = task.expected_qid.lower() in serialized_lower
    has_keywords = all(kw.lower() in serialized_lower for kw in task.expected_keywords)
    return has_qid and has_keywords


def run_benchmark(
    client: OpenAI,
    model: str,
    cache: SQLiteCache,
    max_turns: int = 10,
    offline_only: bool = False,
):
    harness = RLMHarness(
        client=client,
        model=model,
        cache=cache,
        max_turns=max_turns,
        offline_only=offline_only,
    )

    summary_rows = []
    for task in BENCHMARK_TASKS:
        print("=" * 78)
        print(f"TASK: {task.task_id} | {task.category}")
        print(f"QUESTION: {task.question}")
        print("=" * 78)

        t0 = time.time()
        try:
            res = harness.run(task.question)
            elapsed = round(time.time() - t0, 2)
            passed = verify_answer(res["answer"], task)
            summary_rows.append({
                "task_id": task.task_id,
                "passed": passed,
                "turns": res["turns"],
                "graph_calls": res["graph_calls"],
                "cache_hits": res["api_cache_hits"],
                "net_calls": res["api_network_calls"],
                "subcalls": res["subcalls"],
                "elapsed_s": elapsed,
                "answer": res["answer"],
            })
        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            print(f"\n[!] Task {task.task_id} failed with error: {exc}")
            summary_rows.append({
                "task_id": task.task_id,
                "passed": False,
                "turns": max_turns,
                "graph_calls": 0,
                "cache_hits": 0,
                "net_calls": 0,
                "subcalls": 0,
                "elapsed_s": elapsed,
                "answer": f"ERROR: {exc}",
            })

    # Print Markdown Summary Table
    print("\n\n### Benchmark Evaluation Summary\n")
    print("| Task ID | Passed | Turns | Graph Calls | Cache Hits | Net Calls | Sub-Calls | Time (s) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    passed_count = 0
    for row in summary_rows:
        mark = "PASS" if row["passed"] else "FAIL"
        if row["passed"]:
            passed_count += 1
        print(
            f"| `{row['task_id']}` | **{mark}** | {row['turns']} | "
            f"{row['graph_calls']} | {row['cache_hits']} | {row['net_calls']} | "
            f"{row['subcalls']} | {row['elapsed_s']}s |"
        )
    print(f"\n**Overall Score:** {passed_count}/{len(BENCHMARK_TASKS)} tasks solved.")
    print(f"**SQLite Cache Totals:** `{cache.stats()}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLM Wikidata Benchmark & Cache Seeder")
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible endpoint (e.g., http://127.0.0.1:8000/v1 for oMLX or http://localhost:11434/v1 for Ollama)"
    )
    parser.add_argument(
        "--api-key",
        default="local-key",
        help="API key for the endpoint"
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:32b",
        help="Model identifier"
    )
    parser.add_argument(
        "--db",
        default="wikidata_rlm_cache.sqlite",
        help="Path to SQLite cache file"
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Pre-seed the SQLite cache with canonical graph lookups before running"
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Only seed the SQLite cache and exit without calling the LLM"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Disallow live Wikidata HTTP/SPARQL requests (strictly use SQLite cache)"
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    cache = SQLiteCache(args.db)

    try:
        if args.seed or args.seed_only:
            seed_all_tasks(client, args.model, cache)
        if not args.seed_only:
            run_benchmark(
                client=client,
                model=args.model,
                cache=cache,
                offline_only=args.offline
            )
    finally:
        cache.close()

