import pytest

from chess_engine import Board, BoardState


def perft(board: Board, depth: int) -> int:
    """Recursively count leaf positions at the given depth.

    Also verifies the incremental Zobrist hash matches a fresh recomputation
    after every make/unmake, catching hash corruption regressions.
    """
    if depth == 0:
        return 1

    nodes = 0
    for move in board.get_legal_moves():
        state = BoardState(board)
        board.make_move(move)

        assert board.zobrist_hash == board.compute_zobrist_hash(), (
            f"hash mismatch after {move}"
        )

        nodes += perft(board, depth - 1)
        board.unmake_move(move, state)

        assert board.zobrist_hash == state.zobrist_hash, (
            f"hash not restored after unmaking {move}"
        )

    return nodes


@pytest.mark.parametrize(
    ("depth", "expected"),
    [(1, 20), (2, 400), (3, 8902)],
)
def test_perft_starting_position(depth: int, expected: int) -> None:
    board = Board()
    assert perft(board, depth) == expected
