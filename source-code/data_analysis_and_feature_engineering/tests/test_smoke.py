"""Import-only smoke tests.

Both modules run their real work under `if __name__ == '__main__':`, so
importing them here does not trigger the California Housing download.
This test just proves the modules parse and their dependencies resolve.
"""


def test_eda_imports() -> None:
    import eda

    assert callable(eda.main)


def test_feature_engineering_imports() -> None:
    import feature_engineering

    assert callable(feature_engineering.load_and_engineer)
    assert callable(feature_engineering.compare_models)
