# Design of the Benchmark & Cache Seeder Suite

Deterministic Reference Seeder (--seed-only): Before an LLM even runs, seed_benchmark_cache() executes the canonical multi-hop graph
paths (search_entity, search_property, claims, edges, describe, and has) for four distinct reasoning archetypes:

- Forward Multi-Hop (forward_2hop): Guggenheim Museum Bilbao (Q170040) $\rightarrow$ architect (P84) $\rightarrow$ Frank Gehry (Q6722) $\rightarrow$ place of birth (P19) $\rightarrow$ Toronto (Q172).
- Reverse Edge Intersection (reverse_edge_join): Incoming P57 (director) edges to Denis Villeneuve (Q25188) intersected with outgoing P86 (composer) claims pointing to Jóhann Jóhannsson (Q1389457) and publication date (P577) in 2016 $\rightarrow$ Arrival (Q20168408).
- Set Intersection (set_intersection): Incoming P166 (award received) edges for both the Turing Award (Q185667) and the Nobel Prize in Economics (Q47170) $\rightarrow$ Herbert A. Simon (Q20955).
- Semantic Filtering (semantic_subcall): Outgoing P800 (notable work) claims from Ursula K. Le Guin (Q181659) filtered by description/claims to identify the Earthsea novel and its publication year (1968, Q320435).

Flexible Answer Normalization (verify_answer): Just as the paper evaluates execution against ground-truth target entities, the evaluator accepts strings, dicts, or lists from FINAL(answer) and verifies that both the target Q-ID and expected label/value are present and grounded.

CLI Flags for Local Endpoints: Readers can pass --base-url (e.g., [http://127.0.0.1:8000/v1](http://127.0.0.1:8000/v1) for oMLX or http://localhost:11434/v1 for Ollama), --model, and --offline to benchmark different local models against the pre-cached SQLite database.
