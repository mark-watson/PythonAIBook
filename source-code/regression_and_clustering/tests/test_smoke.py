"""Import-only smoke tests.

Both scripts guard their real work under `if __name__ == "__main__":`, so
importing them here does not download data or fit models.
"""


def test_regression_imports() -> None:
    import regression

    assert hasattr(regression, "__file__")


def test_clustering_imports() -> None:
    import clustering

    assert hasattr(clustering, "__file__")
