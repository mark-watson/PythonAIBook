"""Import smoke tests.

Both scripts are import-safe: `classification.py` does its work inside
`main()` behind an `if __name__ == "__main__":` guard, and `load_data.py`
only defines a function. Importing them catches syntax and import-time
errors without reading the CSV files or fitting a model.
"""

import classification
import load_data


def test_classification_imports() -> None:
    assert callable(classification.main)


def test_load_data_is_callable() -> None:
    assert callable(load_data.load_data)
