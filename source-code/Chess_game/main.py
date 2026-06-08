import time
import sys
from chess_engine import (
    Board, WHITE, BLACK, QUEEN, ROOK, BISHOP, KNIGHT, PAWN, KING,
    NAME_TO_SQUARE, SQUARE_NAMES, EMPTY, COLOR_MASK, TYPE_MASK,
    WK, WQ, BK, BQ
)
from chess_bot import get_best_move, evaluate, nodes_visited

def print_pretty_board(board):
    """Prints the board with clean borders, ANSI colors, and game status on the side."""
    print("\033[1;36m  +-----------------+  Game Info & Status\033[0m")
    for rank in range(7, -1, -1):
        row_str = f"\033[1;36m{rank + 1} |\033[0m "
        for file in range(8):
            sq = rank * 8 + file
            piece = board.board[sq]
            if piece == EMPTY:
                # Dim gray dot for empty square
                row_str += "\033[90m. \033[0m"
            else:
                color = piece & COLOR_MASK
                ptype = piece & TYPE_MASK
                char = {
                    PAWN: 'P', KNIGHT: 'N', BISHOP: 'B',
                    ROOK: 'R', QUEEN: 'Q', KING: 'K'
                }[ptype]
                if color == WHITE:
                    # Bright bold white for White pieces
                    row_str += f"\033[1;37m{char} \033[0m"
                else:
                    # Magenta for Black pieces
                    row_str += f"\033[1;35m{char.lower()} \033[0m"
        row_str += "\033[1;36m|\033[0m"
        
        # Side information panel
        if rank == 7:
            active_color = "\033[1;37mWhite\033[0m" if board.turn == WHITE else "\033[1;35mBlack\033[0m"
            row_str += f"   Active Turn: {active_color}"
        elif rank == 6:
            row_str += f"   Move Count:  {board.fullmove_number}"
        elif rank == 5:
            row_str += f"   50-Move Rule: {board.halfmove_clock}/100"
        elif rank == 4:
            in_chk = board.is_in_check()
            check_status = "\033[1;31mIN CHECK!\033[0m" if in_chk else "No check"
            row_str += f"   Check Status: {check_status}"
        elif rank == 3:
            ep = SQUARE_NAMES[board.en_passant_square] if board.en_passant_square else "-"
            row_str += f"   En Passant:  {ep}"
        elif rank == 2:
            rights = ""
            if board.castling_rights & WK: rights += "K"
            if board.castling_rights & WQ: rights += "Q"
            if board.castling_rights & BK: rights += "k"
            if board.castling_rights & BQ: rights += "q"
            row_str += f"   Castling:    {rights if rights else '-'}"
        elif rank == 1:
            eval_score = evaluate(board) / 100.0
            row_str += f"   Eval Score:  {eval_score:+.2f}"
            
        print(row_str)
    print("\033[1;36m  +-----------------+\033[0m")
    print("\033[1;36m    a b c d e f g h\033[0m")
    print()


def parse_user_move(board, move_str):
    """Parses a UCI move string (e.g. 'e2e4' or 'e7e8q') and validates against legal moves."""
    move_str = move_str.strip().lower()
    if len(move_str) not in (4, 5):
        return None
    from_name = move_str[:2]
    to_name = move_str[2:4]
    promo_char = move_str[4] if len(move_str) == 5 else ""

    if from_name not in NAME_TO_SQUARE or to_name not in NAME_TO_SQUARE:
        return None

    from_sq = NAME_TO_SQUARE[from_name]
    to_sq = NAME_TO_SQUARE[to_name]

    promo_piece = 0
    if promo_char:
        if promo_char == 'q': promo_piece = QUEEN
        elif promo_char == 'r': promo_piece = ROOK
        elif promo_char == 'b': promo_piece = BISHOP
        elif promo_char == 'n': promo_piece = KNIGHT
        else: return None

    # Match input against list of legal moves
    for move in board.get_legal_moves():
        if move.from_square == from_sq and move.to_square == to_sq:
            if promo_piece and move.promotion == promo_piece:
                return move
            elif not promo_piece and not move.promotion:
                return move
    return None


def print_help():
    print("\033[1;33mCommands available:\033[0m")
    print("  \033[1;32m<move>\033[0m  - Play a move in UCI format (e.g. e2e4, g1f3, e7e8q for queen promotion)")
    print("  \033[1;32mfen\033[0m     - Print current board FEN string")
    print("  \033[1;32msetfen\033[0m  - Load a custom FEN position")
    print("  \033[1;32mlegal\033[0m   - Print all legal moves in this position")
    print("  \033[1;32mreset\033[0m   - Reset board to starting layout")
    print("  \033[1;32mhelp\033[0m    - Show this commands help guide")
    print("  \033[1;32mexit\033[0m    - Terminate game and exit")
    print()


def play_game():
    board = Board()
    print("\033[1;35m===================================================\033[0m")
    print("\033[1;35m       Welcome to the Antigravity Chess Engine       \033[0m")
    print("\033[1;35m===================================================\033[0m")
    print()
    
    # 1. Select game mode
    while True:
        print("Select Mode:")
        print("  1. Play as White (against Bot)")
        print("  2. Play as Black (against Bot)")
        print("  3. Watch Bot vs Bot")
        choice = input("Enter choice (1-3): ").strip()
        if choice in ('1', '2', '3'):
            break
        print("Invalid choice, select 1, 2, or 3.")

    # 2. Select difficulty/depth
    depth = 3
    if choice in ('1', '2', '3'):
        while True:
            d_input = input("Enter bot search depth (Recommended: 3 or 4, Max: 5) [Default 3]: ").strip()
            if not d_input:
                depth = 3
                break
            try:
                depth = int(d_input)
                if 1 <= depth <= 6:
                    break
                print("Please enter a depth between 1 and 6.")
            except ValueError:
                print("Please enter a valid integer.")

    mode = int(choice)
    print("\nStarting game...")
    print_help()
    print_pretty_board(board)

    while True:
        legal_moves = board.get_legal_moves()

        # Check for game end
        if not legal_moves:
            if board.is_in_check():
                winner = "Black" if board.turn == WHITE else "White"
                print(f"\033[1;31mCHECKMATE! {winner} wins!\033[0m\n")
            else:
                print("\033[1;33mSTALEMATE! Game drawn.\033[0m\n")
            break

        if board.halfmove_clock >= 100:
            print("\033[1;33mDRAW by 50-move rule.\033[0m\n")
            break

        # Bot turn
        is_bot_turn = (
            (mode == 1 and board.turn == BLACK) or
            (mode == 2 and board.turn == WHITE) or
            (mode == 3)
        )

        if is_bot_turn:
            print(f"Bot searching depth {depth}...")
            start_time = time.time()
            bot_move, score = get_best_move(board, depth)
            elapsed = time.time() - start_time
            
            if bot_move is None:
                # No legal moves available
                if board.is_in_check():
                    winner = "White" if board.turn == BLACK else "Black"
                    print(f"\033[1;31mCHECKMATE! {winner} wins!\033[0m\n")
                else:
                    print("\033[1;33mSTALEMATE! Game drawn.\033[0m\n")
                break
                
            print(f"Bot played: \033[1;32m{bot_move}\033[0m (eval: {score/100.0:+.2f}, nodes: {nodes_visited}, time: {elapsed:.2f}s)")
            board.make_move(bot_move)
            print_pretty_board(board)
            if mode == 3:
                time.sleep(0.5)  # Pause so the user can easily spectate
            continue

        # Human turn
        usr_input = input("\033[1;32mYour move (or command):\033[0m ").strip()
        if not usr_input:
            continue

        if usr_input == 'exit':
            print("Thanks for playing!")
            break
        elif usr_input == 'help':
            print_help()
            continue
        elif usr_input == 'reset':
            board.reset_board()
            print_pretty_board(board)
            continue
        elif usr_input == 'fen':
            print(f"FEN: {board.to_fen()}\n")
            continue
        elif usr_input == 'legal':
            print(f"Legal moves: {', '.join(str(m) for m in legal_moves)}\n")
            continue
        elif usr_input == 'setfen':
            fen_input = input("Enter FEN string: ").strip()
            try:
                board.from_fen(fen_input)
                print_pretty_board(board)
            except Exception as e:
                print(f"Error loading FEN: {e}")
            continue

        # Try to parse as move
        move = parse_user_move(board, usr_input)
        if move is None:
            print("\033[1;31mInvalid move or command.\033[0m Type 'help' for guidance.\n")
            continue

        board.make_move(move)
        print_pretty_board(board)

if __name__ == '__main__':
    play_game()
