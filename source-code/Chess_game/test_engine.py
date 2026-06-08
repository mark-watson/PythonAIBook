import time
from chess_engine import Board, BoardState, ZOBRIST_PIECES, ZOBRIST_SIDE, ZOBRIST_CASTLING, ZOBRIST_EP

def perft(board, depth):
    """Recursively generates all positions at the given depth and counts them.
    Used for verifying the correctness and speed of the move generator.
    """
    if depth == 0:
        return 1

    nodes = 0
    moves = board.get_legal_moves()

    for move in moves:
        state = BoardState(board)
        board.make_move(move)
        
        # Verify Zobrist incremental hash matches fresh computation
        if board.zobrist_hash != board.compute_zobrist_hash():
            raise AssertionError(f"Hash mismatch after move {move}! Incremental: {board.zobrist_hash}, Computed: {board.compute_zobrist_hash()}")
            
        nodes += perft(board, depth - 1)
        board.unmake_move(move, state)
        
        # Verify Zobrist hash is restored perfectly
        if board.zobrist_hash != state.zobrist_hash:
            raise AssertionError(f"Hash restore mismatch after unmake move {move}! Board: {board.zobrist_hash}, Saved: {state.zobrist_hash}")

    return nodes

def run_tests():
    board = Board()
    print("Running move generator verification tests (Perft)...")
    
    expected_results = {
        1: 20,
        2: 400,
        3: 8902
    }

    all_passed = True
    for depth, expected in expected_results.items():
        start_time = time.time()
        result = perft(board, depth)
        elapsed = time.time() - start_time
        
        status = "\033[1;32mPASS\033[0m" if result == expected else f"\033[1;31mFAIL (got {result})\033[0m"
        nps = int(result / elapsed) if elapsed > 0 else 0
        print(f"  Depth {depth}: Expected {expected:6,}, got {result:6,}, Time: {elapsed:5.3f}s ({nps:,} nodes/sec) -> {status}")
        
        if result != expected:
            all_passed = False

    if all_passed:
        print("\n\033[1;32mAll move generator tests passed successfully!\033[0m")
    else:
        print("\n\033[1;31mMove generator tests failed! Please check logic.\033[0m")

if __name__ == '__main__':
    run_tests()
