"""Import-only smoke tests.

`DBPedia.py`, `Wikidata.py`, and `DBPedia_and_Wikidata.py` guard their
CLI entry points behind `if __name__ == "__main__":`, so importing them
here does not fire SPARQL queries or Fireworks LLM calls.

`library.py` runs no top-level side effects (module-level code just
defines the Fireworks client from env vars).
"""


def test_library_imports() -> None:
    import library

    assert callable(library.llm_complete)
    assert callable(library.extract_entities)
    assert callable(library.synthesize_answer)


def test_dbpedia_imports() -> None:
    import DBPedia

    assert hasattr(DBPedia, "__file__")


def test_wikidata_imports() -> None:
    import Wikidata

    assert hasattr(Wikidata, "__file__")


def test_dbpedia_and_wikidata_imports() -> None:
    import DBPedia_and_Wikidata

    assert hasattr(DBPedia_and_Wikidata, "__file__")
