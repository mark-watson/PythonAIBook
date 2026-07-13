# okf_explorer.py  — Open Knowledge Format (OKF) Bundle Explorer
#
# Demonstrates the OKF ideas from:
#   https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing/
#   https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md
#
# This script:
#   1. Loads an OKF knowledge bundle from disk (a directory tree of
#      Markdown files with YAML frontmatter).
#   2. Parses every concept document into a structured Concept object.
#   3. Builds a simple in-memory index (search by type, tag, text).
#   4. Uses Ollama (gemma4:e2b-it-qat) as the LLM "consumption agent" —
#      it receives the relevant concept bodies as context and answers
#      natural-language questions about the data assets.
#
# Run: uv run okf_explorer.py
#
# Requirements: ollama (already in pyproject.toml) + model pulled locally:
#   ollama pull gemma4:e2b-it-qat

import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import ollama

# ---------------------------------------------------------------------------
# OKF data model
# ---------------------------------------------------------------------------

RESERVED_FILENAMES = {"index.md", "log.md"}
MODEL = "gemma4:e2b-it-qat"
BUNDLE_DIR = Path(__file__).parent / "bundle"


@dataclass
class Concept:
    """A single OKF concept document."""

    concept_id: str  # relative path without .md suffix
    path: Path  # absolute path to the file
    frontmatter: dict[str, str | list[str]]  # parsed YAML frontmatter
    body: str  # markdown body (everything after frontmatter)

    # Convenience accessors from frontmatter
    @property
    def type(self) -> str:
        value = self.frontmatter.get("type", "Unknown")
        assert isinstance(value, str)
        return value

    @property
    def title(self) -> str:
        value = self.frontmatter.get("title", self.concept_id)
        assert isinstance(value, str)
        return value

    @property
    def description(self) -> str:
        value = self.frontmatter.get("description", "")
        assert isinstance(value, str)
        return value

    @property
    def tags(self) -> list[str]:
        value = self.frontmatter.get("tags", [])
        assert isinstance(value, list)
        return value

    def as_context_block(self) -> str:
        """Return a compact text representation suitable for LLM context."""
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


# ---------------------------------------------------------------------------
# OKF parser
# ---------------------------------------------------------------------------


def _parse_simple_yaml(yaml_text: str) -> dict[str, str | list[str]]:
    """
    Minimal YAML parser for OKF frontmatter.

    Handles the subset used in this bundle:
      - key: scalar value
      - key: [item1, item2, ...]   (inline lists)
    Does NOT require PyYAML as a dependency so the example stays lightweight.
    Upgrade to `import yaml; yaml.safe_load(...)` for production use.
    """
    result: dict[str, str | list[str]] = {}
    for line in yaml_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        # Inline list: [a, b, c]
        if rest.startswith("[") and rest.endswith("]"):
            items = [i.strip().strip('"').strip("'") for i in rest[1:-1].split(",")]
            result[key] = [i for i in items if i]
        else:
            result[key] = rest.strip('"').strip("'")
    return result


def parse_concept(path: Path, bundle_root: Path) -> Concept | None:
    """
    Parse a single OKF concept document.

    Returns None for reserved filenames (index.md, log.md).
    """
    if path.name in RESERVED_FILENAMES:
        return None

    text = path.read_text(encoding="utf-8")

    # Extract YAML frontmatter delimited by ---
    frontmatter: dict[str, str | list[str]] = {}
    body = text
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if fm_match:
        frontmatter = _parse_simple_yaml(fm_match.group(1))
        body = text[fm_match.end() :]

    # Concept ID = path relative to bundle root, without .md suffix
    concept_id = str(path.relative_to(bundle_root).with_suffix(""))
    # Normalise Windows separators
    concept_id = concept_id.replace("\\", "/")

    return Concept(
        concept_id=concept_id,
        path=path,
        frontmatter=frontmatter,
        body=body,
    )


# ---------------------------------------------------------------------------
# OKF bundle loader
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeBundle:
    """An in-memory representation of an OKF knowledge bundle."""

    root: Path
    concepts: list[Concept] = field(default_factory=list)

    @classmethod
    def load(cls, root: Path) -> "KnowledgeBundle":
        """Recursively walk `root` and parse every concept document."""
        bundle = cls(root=root)
        for md_path in sorted(root.rglob("*.md")):
            concept = parse_concept(md_path, root)
            if concept is not None:
                bundle.concepts.append(concept)
        return bundle

    # ------------------------------------------------------------------
    # Search / index helpers
    # ------------------------------------------------------------------

    def by_type(self, concept_type: str) -> list[Concept]:
        """Return all concepts of a given type (case-insensitive)."""
        t = concept_type.lower()
        return [c for c in self.concepts if c.type.lower() == t]

    def by_tag(self, tag: str) -> list[Concept]:
        """Return all concepts that carry a given tag (case-insensitive)."""
        t = tag.lower()
        return [c for c in self.concepts if t in [x.lower() for x in c.tags]]

    def search(self, query: str) -> list[Concept]:
        """
        Simple keyword search across title, description, and body text.
        Returns concepts sorted by hit count (descending).
        """
        keywords = [w.lower() for w in query.split() if len(w) > 2]
        scored: list[tuple[int, Concept]] = []
        for concept in self.concepts:
            haystack = (
                concept.title + " " + concept.description + " " + concept.body
            ).lower()
            score = sum(haystack.count(kw) for kw in keywords)
            if score > 0:
                scored.append((score, concept))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]

    def summary(self) -> str:
        """One-line summary of the bundle contents."""
        type_counts: dict[str, int] = {}
        for c in self.concepts:
            type_counts[c.type] = type_counts.get(c.type, 0) + 1
        counts = ", ".join(f"{v}× {k}" for k, v in sorted(type_counts.items()))
        return f"{len(self.concepts)} concepts ({counts})"


# ---------------------------------------------------------------------------
# LLM consumption agent
# ---------------------------------------------------------------------------


class OKFAgent:
    """
    A simple 'consumption agent' that uses an Ollama LLM to answer
    questions about the knowledge bundle.

    Following the OKF spec's vision of agents that can read and traverse
    the bundle to surface curated insight.
    """

    SYSTEM_PROMPT = textwrap.dedent("""\
        You are a data knowledge assistant. You have been given excerpts from
        an Open Knowledge Format (OKF) knowledge bundle — a collection of
        structured documentation about data tables, metrics, and operational
        playbooks for a retail analytics platform.

        Answer the user's question using ONLY the provided knowledge context.
        Be concise, accurate, and cite the concept ID (e.g. tables/sales_events)
        when referring to a specific asset. If the answer is not in the context,
        say so clearly rather than guessing.
    """)

    def __init__(self, bundle: KnowledgeBundle, model: str = MODEL):
        self.bundle = bundle
        self.model = model

    def _build_context(self, query: str, top_k: int = 4) -> str:
        """Select the most relevant concepts and format them as context."""
        relevant = self.bundle.search(query)[:top_k]
        if not relevant:
            relevant = self.bundle.concepts[:top_k]  # fallback: first N
        blocks = [c.as_context_block() for c in relevant]
        return "\n\n---\n\n".join(blocks)

    def ask(self, question: str) -> str:
        """Send a question to the LLM with relevant OKF context."""
        context = self._build_context(question)
        user_message = f"""## Knowledge Context

{context}

---

## Question

{question}"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        content = response.message.content
        assert content is not None
        return content


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------


def print_section(title: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def main() -> None:
    # 1. Load the OKF bundle ------------------------------------------------
    print_section("Loading OKF Knowledge Bundle")
    bundle = KnowledgeBundle.load(BUNDLE_DIR)
    print(f"Bundle root : {BUNDLE_DIR}")
    print(f"Contents    : {bundle.summary()}")

    # 2. Show all concept IDs and types ------------------------------------
    print_section("All Concepts in Bundle")
    for c in bundle.concepts:
        print(f"  [{c.type:<18}]  {c.concept_id}")
        if c.description:
            print(f"  {'':20}  {c.description[:72]}")

    # 3. Demonstrate index / search ----------------------------------------
    print_section("Search: 'revenue'")
    for c in bundle.search("revenue")[:3]:
        print(f"  {c.concept_id}  —  {c.description[:60]}")

    print_section("Filter by type: 'Metric'")
    for c in bundle.by_type("Metric"):
        print(f"  {c.concept_id}  —  {c.title}")

    print_section("Filter by tag: 'KPI'")
    for c in bundle.by_tag("KPI"):
        print(f"  {c.concept_id}  —  {c.title}")

    # 4. LLM Q&A over the knowledge bundle ---------------------------------
    print_section(f"LLM Q&A  (model: {MODEL})")

    agent = OKFAgent(bundle)

    questions = [
        "How is daily revenue calculated and what tables does it use?",
        "What should I do if daily revenue drops suddenly?",
        "What percentage of sales events have a customer ID? "
        "And what does that tell us about LTV calculations?",
        "How do I join sales_events to products correctly for historical reports?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        print("-" * 60)
        answer = agent.ask(q)
        # Wrap long lines for readability
        for line in answer.splitlines():
            if line:
                print(textwrap.fill(line, width=72, subsequent_indent="    "))
            else:
                print()


if __name__ == "__main__":
    if not BUNDLE_DIR.exists():
        print(
            f"ERROR: Bundle directory not found: {BUNDLE_DIR}\n"
            "Make sure you run this script from the openknowledge_format/ directory.",
            file=sys.stderr,
        )
        sys.exit(1)
    main()
