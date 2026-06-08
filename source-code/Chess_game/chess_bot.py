from chess_engine import (
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    WHITE, BLACK, TYPE_MASK, COLOR_MASK, BoardState,
    EMPTY
)

# Material value constants for evaluation
PIECE_VALUES = {
    PAWN: 100,
    KNIGHT: 320,
    BISHOP: 330,
    ROOK: 500,
    QUEEN: 900,
    KING: 20000
}

# Piece-Square Tables (PSTs) from White's perspective.
# Positive values encourage pieces to occupy strong active squares.
PAWN_PST = [
      0,  0,  0,  0,  0,  0,  0,  0,
     50, 50, 50, 50, 50, 50, 50, 50,
     10, 10, 20, 30, 30, 20, 10, 10,
      5,  5, 10, 25, 25, 10,  5,  5,
      0,  0,  0, 20, 20,  0,  0,  0,
      5, -5,-10,  0,  0,-10, -5,  5,
      5, 10, 10,-20,-20, 10, 10,  5,
      0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_PST = [
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  5,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

# King tables for middlegame safety vs. endgame centralization
KING_MIDDLE_PST = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
]

KING_END_PST = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
]

PST_TABLES = {
    PAWN: PAWN_PST,
    KNIGHT: KNIGHT_PST,
    BISHOP: BISHOP_PST,
    ROOK: ROOK_PST,
    QUEEN: QUEEN_PST
}

max_depth = 3
nodes_visited = 0

def is_endgame(board):
    """Determines if the board state has transitioned to the endgame.
    Based on the remaining major and minor pieces (excluding pawns & kings).
    """
    material = 0
    for sq in board.pieces[WHITE]:
        ptype = board.board[sq] & TYPE_MASK
        if ptype not in (PAWN, KING):
            material += PIECE_VALUES[ptype]
    for sq in board.pieces[BLACK]:
        ptype = board.board[sq] & TYPE_MASK
        if ptype not in (PAWN, KING):
            material += PIECE_VALUES[ptype]
    return material <= 3000


def evaluate(board):
    """Computes the board evaluation score.
    Positive scores favor White, negative favor Black.
    """
    score = 0
    endgame = is_endgame(board)

    # 1. White pieces evaluation
    for sq in board.pieces[WHITE]:
        piece = board.board[sq]
        ptype = piece & TYPE_MASK
        score += PIECE_VALUES[ptype]
        if ptype == KING:
            score += KING_END_PST[sq] if endgame else KING_MIDDLE_PST[sq]
        else:
            score += PST_TABLES[ptype][sq]

    # 2. Black pieces evaluation
    for sq in board.pieces[BLACK]:
        piece = board.board[sq]
        ptype = piece & TYPE_MASK
        score -= PIECE_VALUES[ptype]
        mirrored_sq = sq ^ 56  # Vertical flip for Black's perspective
        if ptype == KING:
            score -= KING_END_PST[mirrored_sq] if endgame else KING_MIDDLE_PST[mirrored_sq]
        else:
            score -= PST_TABLES[ptype][mirrored_sq]

    # Return relative evaluation depending on active turn
    return score if board.turn == WHITE else -score


def move_value(board, move, tt_move=None):
    """Heuristic function for ordering moves.
    High score is sorted first (captures, promotions, active squares).
    """
    if tt_move is not None and move == tt_move:
        return 1000000

    score = 0
    # Capture (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
    if move.piece_captured:
        victim_val = PIECE_VALUES[move.piece_captured & TYPE_MASK]
        attacker_val = PIECE_VALUES[move.piece_moved & TYPE_MASK]
        score += 10000 + victim_val - (attacker_val // 100)

    # Promotion
    if move.promotion:
        score += 8000 + PIECE_VALUES[move.promotion]

    # Castling
    if move.is_castling:
        score += 1000

    # Positional progress based on PST
    ptype = move.piece_moved & TYPE_MASK
    if ptype != KING:
        pst = PST_TABLES[ptype]
        if board.turn == WHITE:
            score += pst[move.to_square] - pst[move.from_square]
        else:
            score += pst[move.to_square ^ 56] - pst[move.from_square ^ 56]

    return score


# Transposition Table Constants
TT_EXACT = 0
TT_ALPHA = 1  # Upper bound (fail-low)
TT_BETA = 2   # Lower bound (fail-high)

# Global Transposition Table
transposition_table = {}


def search(board, depth, alpha, beta):
    """Negamax search with Alpha-Beta pruning & Transposition Table."""
    global nodes_visited
    nodes_visited += 1

    # Draw detection (50-move rule)
    if board.halfmove_clock >= 100:
        return 0

    original_alpha = alpha

    # 1. Transposition Table Lookup
    tt_entry = transposition_table.get(board.zobrist_hash)
    if tt_entry is not None and tt_entry['depth'] >= depth:
        tt_flag = tt_entry['flag']
        tt_score = tt_entry['score']
        
        # Adjust mate score relative to the current search depth
        score_val = tt_score
        if score_val > 29000:
            score_val -= (max_depth - depth)
        elif score_val < -29000:
            score_val += (max_depth - depth)

        if tt_flag == TT_EXACT:
            return score_val
        elif tt_flag == TT_ALPHA:
            if score_val <= alpha:
                return score_val
            beta = min(beta, score_val)
        elif tt_flag == TT_BETA:
            if score_val >= beta:
                return score_val
            alpha = max(alpha, score_val)

        if alpha >= beta:
            return score_val

    legal_moves = board.get_legal_moves()

    if not legal_moves:
        if board.is_in_check():
            # Checkmate: negative value adjusted by depth to prefer quicker mate
            return -30000 + (max_depth - depth)
        else:
            # Stalemate
            return 0

    if depth == 0:
        return quiescence_search(board, alpha, beta)

    # Move ordering: prioritize the best move from the transposition table
    tt_move = tt_entry['best_move'] if tt_entry is not None else None
    legal_moves.sort(key=lambda m: move_value(board, m, tt_move), reverse=True)

    best_score = -float('inf')
    best_move = None

    for move in legal_moves:
        state = BoardState(board)
        board.make_move(move)
        score = -search(board, depth - 1, -beta, -alpha)
        board.unmake_move(move, state)

        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    # 2. Store in Transposition Table
    if best_score <= original_alpha:
        tt_flag = TT_ALPHA
    elif best_score >= beta:
        tt_flag = TT_BETA
    else:
        tt_flag = TT_EXACT

    stored_score = best_score
    if stored_score > 29000:
        stored_score += (max_depth - depth)
    elif stored_score < -29000:
        stored_score -= (max_depth - depth)

    # Avoid overwriting deeper search result with shallow one
    if tt_entry is None or depth >= tt_entry['depth']:
        transposition_table[board.zobrist_hash] = {
            'depth': depth,
            'score': stored_score,
            'flag': tt_flag,
            'best_move': best_move
        }

    return best_score


def quiescence_search(board, alpha, beta):
    """Performs quiescence search evaluating captures to avoid the horizon effect."""
    global nodes_visited
    nodes_visited += 1

    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    opponent = BLACK if board.turn == WHITE else WHITE
    color = board.turn

    # Filter for capture/promotion pseudo-legal moves
    moves = board.get_pseudo_legal_moves()
    captures = []
    for move in moves:
        if move.piece_captured or move.promotion:
            # Validate legality of capture in-place
            state = BoardState(board)
            board.make_move(move)
            if not board.is_square_attacked(board.king_square[color], opponent):
                captures.append(move)
            board.unmake_move(move, state)

    # Order captures
    captures.sort(key=lambda m: move_value(board, m), reverse=True)

    for move in captures:
        state = BoardState(board)
        board.make_move(move)
        score = -quiescence_search(board, -beta, -alpha)
        board.unmake_move(move, state)

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def get_best_move(board, depth=3):
    """Finds the best move at the given depth using iterative deepening Negamax search."""
    global max_depth, nodes_visited
    
    # Cap TT size to avoid unbounded memory growth
    if len(transposition_table) > 500000:
        transposition_table.clear()

    best_move = None
    best_score = -float('inf')

    # Iterative Deepening Loop: search from depth 1 to target depth
    for d in range(1, depth + 1):
        max_depth = d
        nodes_visited = 0

        legal_moves = board.get_legal_moves()
        if not legal_moves:
            break

        # Order moves using the TT best move from previous depth searches
        tt_entry = transposition_table.get(board.zobrist_hash)
        prev_best = tt_entry['best_move'] if tt_entry is not None else None
        legal_moves.sort(key=lambda m: move_value(board, m, prev_best), reverse=True)

        current_best_move = None
        current_best_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        for move in legal_moves:
            state = BoardState(board)
            board.make_move(move)
            score = -search(board, d - 1, -beta, -alpha)
            board.unmake_move(move, state)

            if score > current_best_score:
                current_best_score = score
                current_best_move = move
            if score > alpha:
                alpha = score

        # Update root search results upon completion of this depth iteration
        best_move = current_best_move
        best_score = current_best_score

        # Save root search to transposition table
        transposition_table[board.zobrist_hash] = {
            'depth': d,
            'score': best_score,
            'flag': TT_EXACT,
            'best_move': best_move
        }

    return best_move, best_score

