"""Import-only smoke test for the OKF explorer.

`okf_explorer.py` guards its real work behind `if __name__ == "__main__":`,
so importing it here does not touch Ollama or the bundle contents.
"""


def test_okf_explorer_imports() -> None:
    import okf_explorer

    assert hasattr(okf_explorer, "__file__")
