"""Import-only smoke test.

`game.py` guards its real work behind `if __name__ == "__main__":`, so
importing it does not call the OpenAI API or start the game loop.
"""


def test_game_imports() -> None:
    import game

    assert hasattr(game, "__file__")
