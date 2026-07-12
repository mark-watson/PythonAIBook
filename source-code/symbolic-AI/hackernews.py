# hackernews.py - Extract named entities from Hacker News and assert as Prolog facts
#
# This example combines three technologies:
#   1. Firebase Hacker News API - fetches the most recent stories
#   2. spaCy NLP model - extracts named entities (people, organizations)
#   3. Swi-Prolog - stores extracted entities as Prolog facts and queries them
#
# For each story, the script fetches the linked web page, runs NER (Named
# Entity Recognition) on the text, and asserts person/2 and organization/2
# facts into a Prolog knowledge base keyed by the source URL.
#
# Requirements:
#   brew install swi-prolog
#   uv pip install swiplserver spacy beautifulsoup4
#   python -m spacy download en_core_web_sm

import json
import re
from pprint import pprint
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from swiplserver import PrologMQI

# Load the spaCy English NLP model (auto-downloads if missing)
import spacy

try:
    spacy_model = spacy.load("en_core_web_sm")
except Exception:
    from os import system

    system("python -m spacy download en_core_web_sm")
    spacy_model = spacy.load("en_core_web_sm")

LEN = 100  # larger amount of text is more expensive for OpenAI APIs


def get_new_stories(anAgent: dict[str, str] | None = None):
    """Fetch the IDs of the most recent Hacker News stories (returns top 3)."""
    if anAgent is None:
        anAgent = {"User-Agent": "PythonAiBook/1.0"}
    req = Request(
        "https://hacker-news.firebaseio.com/v0/newstories.json", headers=anAgent
    )
    httpResponse = urlopen(req)
    data = httpResponse.read()
    ids = json.loads(data)
    # just return the most recent 3 stories:
    return ids[0:3]


def get_story_data(id: int, anAgent: dict[str, str] | None = None):
    """Fetch the JSON metadata for a single Hacker News story by ID."""
    if anAgent is None:
        anAgent = {"User-Agent": "PythonAiBook/1.0"}
    req = Request(
        f"https://hacker-news.firebaseio.com/v0/item/{id}.json", headers=anAgent
    )
    httpResponse = urlopen(req)
    return json.loads(httpResponse.read())


def get_story_text(a_uri: str, anAgent: dict[str, str] | None = None):
    """Fetch a web page and extract its visible text using BeautifulSoup."""
    if anAgent is None:
        anAgent = {"User-Agent": "PythonAiBook/1.0"}
    req = Request(a_uri, headers=anAgent)
    httpResponse = urlopen(req)
    soup = BeautifulSoup(httpResponse.read(), "html.parser")
    return soup.get_text()


def find_entities_in_text(some_text: str):
    """Run spaCy NER on text and return [entity_text, entity_label] pairs."""

    def clean(s: str) -> str:
        return s.replace("\n", " ").strip()

    doc = spacy_model(some_text)
    return map(list, [[clean(entity.text), entity.label_] for entity in doc.ents])


# Regex to strip characters that are invalid in Prolog atoms
regex = re.compile("[^a-z_]")


def safe_prolog_text(s: str) -> str:
    """Convert a string into a valid Prolog atom (lowercase, underscores only)."""
    s = s.lower().replace(" ", "_").replace("&", "and").replace("-", "_")
    return regex.sub("", s)


# --- Main pipeline: fetch stories, extract entities, assert into Prolog ---

ids = get_new_stories()

for id in ids:
    story_json_data = get_story_data(id)
    if story_json_data is not None and "url" in story_json_data:
        print(f"Processing {story_json_data['url']}\n")

        # Fetch the story's web page (skip if blocked by the server)
        try:
            story_text = get_story_text(story_json_data["url"])
        except Exception as e:
            print(f"  Skipping (could not fetch: {e})\n")
            continue

        # Extract named entities and partition into organizations and people
        entities = list(find_entities_in_text(story_text))
        organizations = set(
            [
                safe_prolog_text(name)
                for [name, entity_type] in entities
                if entity_type == "ORG"
            ]
        )
        people = set(
            [
                safe_prolog_text(name)
                for [name, entity_type] in entities
                if entity_type == "PERSON"
            ]
        )

        # Assert each entity as a Prolog fact and query the knowledge base
        with PrologMQI() as mqi:
            with mqi.create_thread() as prolog_thread:
                # Assert person(Name, URI) facts
                for person in people:
                    s = f"assertz(person({person}, \
                                  '{story_json_data['url']}'))."
                    try:
                        prolog_thread.query(s)
                    except Exception:
                        print(f"Error with term: {s}")

                # Assert organization(Name, URI) facts
                for organization in organizations:
                    s = f"assertz(organization({organization}, '{story_json_data['url']}'))."
                    try:
                        prolog_thread.query(s)
                    except Exception:
                        print(f"Error with term: {s}")

                # Query and display all asserted organizations and people
                try:
                    result = prolog_thread.query("organization(Organization, URI).")
                    pprint(result)
                except Exception:
                    print("No results for organizations.")
                try:
                    result = prolog_thread.query("person(Person, URI).")
                    pprint(result)
                except Exception:
                    print("No results for people.")
