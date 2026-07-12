"""Import-only smoke tests.

Both scripts guard their real work under `if __name__ == "__main__":`, so
importing them here does *not* download Stable Diffusion weights or call the
Gemini API. This just proves both modules parse and their deps resolve.
"""


def test_image_generation_imports() -> None:
    import image_generation

    assert callable(image_generation.main)


def test_gemini_image_generation_imports() -> None:
    import gemini_image_generation

    assert callable(gemini_image_generation.main)
