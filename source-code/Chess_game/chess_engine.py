import sys
import random

# Piece representation constants
EMPTY = 0
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

WHITE = 8
BLACK = 16

COLOR_MASK = WHITE | BLACK
TYPE_MASK = 7

# Castling rights bitwise constants
WK = 1  # White King-side
WQ = 2  # White Queen-side
BK = 4  # Black King-side
BQ = 8  # Black Queen-side

# Square coordinate mapping
SQUARE_NAMES = [
    f"{chr(ord('a') + col)}{row + 1}"
    for row in range(8)
    for col in range(8)
]
NAME_TO_SQUARE = {name: i for i, name in enumerate(SQUARE_NAMES)}

# Precomputed lookup tables
_knight_moves_list = [[] for _ in range(64)]
_king_moves_list = [[] for _ in range(64)]
_rook_rays_list = [[] for _ in range(64)]
_bishop_rays_list = [[] for _ in range(64)]
_queen_rays_list = [[] for _ in range(64)]

def _precompute_moves():
    knight_offsets = [
        (2, 1), (2, -1), (-2, 1), (-2, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2)
    ]
    king_offsets = [
        (1, 1), (1, 0), (1, -1),
        (0, 1),          (0, -1),
        (-1, 1), (-1, 0), (-1, -1)
    ]
    rook_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    bishop_dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    for s in range(64):
        r, f = s // 8, s % 8

        # Knight
        for dr, df in knight_offsets:
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                _knight_moves_list[s].append(nr * 8 + nf)

        # King
        for dr, df in king_offsets:
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                _king_moves_list[s].append(nr * 8 + nf)

        # Rook Rays
        for dr, df in rook_dirs:
            ray = []
            nr, nf = r + dr, f + df
            while 0 <= nr < 8 and 0 <= nf < 8:
                ray.append(nr * 8 + nf)
                nr += dr
                nf += df
            if ray:
                _rook_rays_list[s].append(ray)

        # Bishop Rays
        for dr, df in bishop_dirs:
            ray = []
            nr, nf = r + dr, f + df
            while 0 <= nr < 8 and 0 <= nf < 8:
                ray.append(nr * 8 + nf)
                nr += dr
                nf += df
            if ray:
                _bishop_rays_list[s].append(ray)

        # Queen Rays
        _queen_rays_list[s] = _rook_rays_list[s] + _bishop_rays_list[s]

_precompute_moves()

# Freeze tables as tuples for optimal iteration speed in Python
KNIGHT_MOVES = tuple(tuple(moves) for moves in _knight_moves_list)
KING_MOVES = tuple(tuple(moves) for moves in _king_moves_list)
ROOK_RAYS = tuple(tuple(tuple(ray) for ray in rays) for rays in _rook_rays_list)
BISHOP_RAYS = tuple(tuple(tuple(ray) for ray in rays) for rays in _bishop_rays_list)
QUEEN_RAYS = tuple(tuple(tuple(ray) for ray in rays) for rays in _queen_rays_list)

# Zobrist hashing keys initialized with a stable seed for reproducibility
_rng = random.Random(1337)
ZOBRIST_PIECES = [[_rng.getrandbits(64) for _ in range(32)] for _ in range(64)]
ZOBRIST_SIDE = _rng.getrandbits(64)
ZOBRIST_CASTLING = [_rng.getrandbits(64) for _ in range(16)]
ZOBRIST_EP = [_rng.getrandbits(64) for _ in range(64)]


class Move:
    __slots__ = (
        'from_square',
        'to_square',
        'piece_moved',
        'piece_captured',
        'promotion',
        'is_en_passant',
        'is_castling',
        'is_double_push'
    )

    def __init__(self, from_square, to_square, piece_moved, piece_captured=0, promotion=0,
                 is_en_passant=False, is_castling=False, is_double_push=False):
        self.from_square = from_square
        self.to_square = to_square
        self.piece_moved = piece_moved
        self.piece_captured = piece_captured
        self.promotion = promotion
        self.is_en_passant = is_en_passant
        self.is_castling = is_castling
        self.is_double_push = is_double_push

    def uci_string(self):
        """Returns the move in UCI format (e.g. e2e4, e7e8q)"""
        promo_char = ""
        if self.promotion:
            promo_char = {QUEEN: 'q', ROOK: 'r', BISHOP: 'b', KNIGHT: 'n'}[self.promotion]
        return f"{SQUARE_NAMES[self.from_square]}{SQUARE_NAMES[self.to_square]}{promo_char}"

    def __str__(self):
        return self.uci_string()

    def __repr__(self):
        return f"Move({self.uci_string()}, moved={self.piece_moved}, captured={self.piece_captured})"

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return (self.from_square == other.from_square and
                self.to_square == other.to_square and
                self.promotion == other.promotion)


class BoardState:
    """Stores board metadata for undoing moves without cloning the entire board state."""
    __slots__ = ('en_passant_square', 'castling_rights', 'halfmove_clock', 'zobrist_hash')

    def __init__(self, board):
        self.en_passant_square = board.en_passant_square
        self.castling_rights = board.castling_rights
        self.halfmove_clock = board.halfmove_clock
        self.zobrist_hash = board.zobrist_hash


class Board:
    def __init__(self):
        self.board = [EMPTY] * 64
        self.pieces = {WHITE: set(), BLACK: set()}
        self.king_square = {WHITE: None, BLACK: None}
        self.turn = WHITE
        self.castling_rights = WK | WQ | BK | BQ
        self.en_passant_square = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.history = []
        self.zobrist_hash = 0

        self.reset_board()

    def reset_board(self):
        self.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    def from_fen(self, fen):
        """Sets the board state from a FEN string."""
        parts = fen.split()
        placement = parts[0]
        self.turn = WHITE if parts[1] == 'w' else BLACK

        # Castling rights
        self.castling_rights = 0
        if parts[2] != '-':
            if 'K' in parts[2]: self.castling_rights |= WK
            if 'Q' in parts[2]: self.castling_rights |= WQ
            if 'k' in parts[2]: self.castling_rights |= BK
            if 'q' in parts[2]: self.castling_rights |= BQ

        # En passant
        if parts[3] == '-':
            self.en_passant_square = None
        else:
            self.en_passant_square = NAME_TO_SQUARE[parts[3]]

        self.halfmove_clock = int(parts[4])
        self.fullmove_number = int(parts[5])

        # Board grid & Piece lists
        self.board = [EMPTY] * 64
        self.pieces[WHITE].clear()
        self.pieces[BLACK].clear()
        self.king_square[WHITE] = None
        self.king_square[BLACK] = None

        rows = placement.split('/')
        for row_idx, row in enumerate(rows):
            rank = 7 - row_idx
            file = 0
            for char in row:
                if char.isdigit():
                    file += int(char)
                else:
                    color = WHITE if char.isupper() else BLACK
                    piece_char = char.lower()
                    ptype = {
                        'p': PAWN, 'n': KNIGHT, 'b': BISHOP,
                        'r': ROOK, 'q': QUEEN, 'k': KING
                    }[piece_char]
                    sq = rank * 8 + file
                    piece = color | ptype
                    self.board[sq] = piece
                    self.pieces[color].add(sq)
                    if ptype == KING:
                        self.king_square[color] = sq
                    file += 1

        self.zobrist_hash = self.compute_zobrist_hash()

    def compute_zobrist_hash(self):
        """Computes the Zobrist hash of the current board state from scratch."""
        h = 0
        for sq in range(64):
            piece = self.board[sq]
            if piece != EMPTY:
                h ^= ZOBRIST_PIECES[sq][piece]
        if self.turn == BLACK:
            h ^= ZOBRIST_SIDE
        h ^= ZOBRIST_CASTLING[self.castling_rights]
        if self.en_passant_square is not None:
            h ^= ZOBRIST_EP[self.en_passant_square]
        return h


    def to_fen(self):
        """Generates the FEN representation of the current board state."""
        rows = []
        for rank in range(7, -1, -1):
            empty_count = 0
            row_str = ""
            for file in range(8):
                sq = rank * 8 + file
                piece = self.board[sq]
                if piece == EMPTY:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    color = piece & COLOR_MASK
                    ptype = piece & TYPE_MASK
                    char = {
                        PAWN: 'p', KNIGHT: 'n', BISHOP: 'b',
                        ROOK: 'r', QUEEN: 'q', KING: 'k'
                    }[ptype]
                    row_str += char.upper() if color == WHITE else char
            if empty_count > 0:
                row_str += str(empty_count)
            rows.append(row_str)

        fen_placement = "/".join(rows)
        turn_str = 'w' if self.turn == WHITE else 'b'

        castling_str = ""
        if self.castling_rights & WK: castling_str += 'K'
        if self.castling_rights & WQ: castling_str += 'Q'
        if self.castling_rights & BK: castling_str += 'k'
        if self.castling_rights & BQ: castling_str += 'q'
        if not castling_str: castling_str = "-"

        ep_str = "-" if self.en_passant_square is None else SQUARE_NAMES[self.en_passant_square]

        return f"{fen_placement} {turn_str} {castling_str} {ep_str} {self.halfmove_clock} {self.fullmove_number}"

    def is_square_attacked(self, square, attacker_color):
        """Returns True if the square is attacked by any piece of attacker_color."""
        opponent = attacker_color
        # 1. Pawn attacks
        if opponent == WHITE:
            # Pawns attack upwards, so checker is downwards
            if square % 8 > 0 and square - 7 >= 0 and self.board[square - 7] == (WHITE | PAWN):
                return True
            if square % 8 < 7 and square - 9 >= 0 and self.board[square - 9] == (WHITE | PAWN):
                return True
        else:
            # Pawns attack downwards, so checker is upwards
            if square % 8 < 7 and square + 7 < 64 and self.board[square + 7] == (BLACK | PAWN):
                return True
            if square % 8 > 0 and square + 9 < 64 and self.board[square + 9] == (BLACK | PAWN):
                return True

        # 2. Knight attacks
        for target in KNIGHT_MOVES[square]:
            if self.board[target] == (opponent | KNIGHT):
                return True

        # 3. King attacks
        for target in KING_MOVES[square]:
            if self.board[target] == (opponent | KING):
                return True

        # 4. Rook / Queen attacks (horizontal & vertical)
        for ray in ROOK_RAYS[square]:
            for target in ray:
                target_piece = self.board[target]
                if target_piece != EMPTY:
                    if (target_piece & COLOR_MASK) == opponent and (target_piece & TYPE_MASK) in (ROOK, QUEEN):
                        return True
                    break

        # 5. Bishop / Queen attacks (diagonals)
        for ray in BISHOP_RAYS[square]:
            for target in ray:
                target_piece = self.board[target]
                if target_piece != EMPTY:
                    if (target_piece & COLOR_MASK) == opponent and (target_piece & TYPE_MASK) in (BISHOP, QUEEN):
                        return True
                    break

        return False

    def is_in_check(self, color=None):
        if color is None:
            color = self.turn
        opponent = BLACK if color == WHITE else WHITE
        return self.is_square_attacked(self.king_square[color], opponent)

    def get_pseudo_legal_moves(self):
        """Generates all pseudo-legal moves for the player whose turn it is."""
        moves = []
        color = self.turn
        opponent = BLACK if color == WHITE else WHITE

        for s in self.pieces[color]:
            piece = self.board[s]
            ptype = piece & TYPE_MASK

            if ptype == PAWN:
                if color == WHITE:
                    # Single Push
                    target = s + 8
                    if target < 64 and self.board[target] == EMPTY:
                        if target >= 56:  # Rank 8 promotion
                            for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                                moves.append(Move(s, target, piece, promotion=promo))
                        else:
                            moves.append(Move(s, target, piece))
                            # Double Push from starting rank (Rank 2)
                            if 8 <= s <= 15:
                                double_target = s + 16
                                if self.board[double_target] == EMPTY:
                                    moves.append(Move(s, double_target, piece, is_double_push=True))
                    # Diagonal Attacks
                    file_idx = s % 8
                    if file_idx > 0:
                        target = s + 7
                        target_piece = self.board[target]
                        if target_piece != EMPTY and (target_piece & COLOR_MASK) == opponent:
                            if target >= 56:
                                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                                    moves.append(Move(s, target, piece, target_piece, promotion=promo))
                            else:
                                moves.append(Move(s, target, piece, target_piece))
                        elif target == self.en_passant_square:
                            moves.append(Move(s, target, piece, opponent | PAWN, is_en_passant=True))
                    if file_idx < 7:
                        target = s + 9
                        target_piece = self.board[target]
                        if target_piece != EMPTY and (target_piece & COLOR_MASK) == opponent:
                            if target >= 56:
                                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                                    moves.append(Move(s, target, piece, target_piece, promotion=promo))
                            else:
                                moves.append(Move(s, target, piece, target_piece))
                        elif target == self.en_passant_square:
                            moves.append(Move(s, target, piece, opponent | PAWN, is_en_passant=True))
                else:  # BLACK
                    # Single Push
                    target = s - 8
                    if target >= 0 and self.board[target] == EMPTY:
                        if target <= 7:  # Rank 1 promotion
                            for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                                moves.append(Move(s, target, piece, promotion=promo))
                        else:
                            moves.append(Move(s, target, piece))
                            # Double Push from starting rank (Rank 7)
                            if 48 <= s <= 55:
                                double_target = s - 16
                                if self.board[double_target] == EMPTY:
                                    moves.append(Move(s, double_target, piece, is_double_push=True))
                    # Diagonal Attacks
                    file_idx = s % 8
                    if file_idx > 0:
                        target = s - 9
                        target_piece = self.board[target]
                        if target_piece != EMPTY and (target_piece & COLOR_MASK) == opponent:
                            if target <= 7:
                                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                                    moves.append(Move(s, target, piece, target_piece, promotion=promo))
                            else:
                                moves.append(Move(s, target, piece, target_piece))
                        elif target == self.en_passant_square:
                            moves.append(Move(s, target, piece, opponent | PAWN, is_en_passant=True))
                    if file_idx < 7:
                        target = s - 7
                        target_piece = self.board[target]
                        if target_piece != EMPTY and (target_piece & COLOR_MASK) == opponent:
                            if target <= 7:
                                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                                    moves.append(Move(s, target, piece, target_piece, promotion=promo))
                            else:
                                moves.append(Move(s, target, piece, target_piece))
                        elif target == self.en_passant_square:
                            moves.append(Move(s, target, piece, opponent | PAWN, is_en_passant=True))

            elif ptype == KNIGHT:
                for target in KNIGHT_MOVES[s]:
                    target_piece = self.board[target]
                    if target_piece == EMPTY:
                        moves.append(Move(s, target, piece))
                    elif (target_piece & COLOR_MASK) == opponent:
                        moves.append(Move(s, target, piece, target_piece))

            elif ptype == KING:
                for target in KING_MOVES[s]:
                    target_piece = self.board[target]
                    if target_piece == EMPTY:
                        moves.append(Move(s, target, piece))
                    elif (target_piece & COLOR_MASK) == opponent:
                        moves.append(Move(s, target, piece, target_piece))

                # Castling moves
                if color == WHITE:
                    if s == 4:
                        # King-side WK
                        if (self.castling_rights & WK) and self.board[5] == EMPTY and self.board[6] == EMPTY:
                            if not self.is_square_attacked(4, BLACK) and not self.is_square_attacked(5, BLACK) and not self.is_square_attacked(6, BLACK):
                                moves.append(Move(4, 6, piece, is_castling=True))
                        # Queen-side WQ
                        if (self.castling_rights & WQ) and self.board[3] == EMPTY and self.board[2] == EMPTY and self.board[1] == EMPTY:
                            if not self.is_square_attacked(4, BLACK) and not self.is_square_attacked(3, BLACK) and not self.is_square_attacked(2, BLACK):
                                moves.append(Move(4, 2, piece, is_castling=True))
                else:  # BLACK
                    if s == 60:
                        # King-side BK
                        if (self.castling_rights & BK) and self.board[61] == EMPTY and self.board[62] == EMPTY:
                            if not self.is_square_attacked(60, WHITE) and not self.is_square_attacked(61, WHITE) and not self.is_square_attacked(62, WHITE):
                                moves.append(Move(60, 62, piece, is_castling=True))
                        # Queen-side BQ
                        if (self.castling_rights & BQ) and self.board[59] == EMPTY and self.board[58] == EMPTY and self.board[57] == EMPTY:
                            if not self.is_square_attacked(60, WHITE) and not self.is_square_attacked(59, WHITE) and not self.is_square_attacked(58, WHITE):
                                moves.append(Move(60, 58, piece, is_castling=True))

            else:  # Sliding pieces (Bishop, Rook, Queen)
                rays = []
                if ptype == BISHOP or ptype == QUEEN:
                    rays.extend(BISHOP_RAYS[s])
                if ptype == ROOK or ptype == QUEEN:
                    rays.extend(ROOK_RAYS[s])

                for ray in rays:
                    for target in ray:
                        target_piece = self.board[target]
                        if target_piece == EMPTY:
                            moves.append(Move(s, target, piece))
                        else:
                            if (target_piece & COLOR_MASK) == opponent:
                                moves.append(Move(s, target, piece, target_piece))
                            break

        return moves

    def get_legal_moves(self):
        """Generates all fully legal moves (checks if move would put/leave king in check)."""
        pseudo_moves = self.get_pseudo_legal_moves()
        legal_moves = []
        color = self.turn
        opponent = BLACK if color == WHITE else WHITE

        for move in pseudo_moves:
            state = BoardState(self)
            self.make_move(move)
            king_sq = self.king_square[color]
            if not self.is_square_attacked(king_sq, opponent):
                legal_moves.append(move)
            self.unmake_move(move, state)
        return legal_moves

    def make_move(self, move):
        color = self.turn
        opponent = BLACK if color == WHITE else WHITE

        # Incremental Zobrist updates: XOR out old volatile properties & moved piece
        self.zobrist_hash ^= ZOBRIST_SIDE
        self.zobrist_hash ^= ZOBRIST_CASTLING[self.castling_rights]
        if self.en_passant_square is not None:
            self.zobrist_hash ^= ZOBRIST_EP[self.en_passant_square]
        self.zobrist_hash ^= ZOBRIST_PIECES[move.from_square][move.piece_moved]

        # 1. Update Board and Piece lists
        self.board[move.from_square] = EMPTY
        self.pieces[color].remove(move.from_square)

        # Handle En Passant Capture
        if move.is_en_passant:
            cap_sq = move.to_square - 8 if color == WHITE else move.to_square + 8
            self.board[cap_sq] = EMPTY
            self.pieces[opponent].remove(cap_sq)
            self.zobrist_hash ^= ZOBRIST_PIECES[cap_sq][move.piece_captured]
        # Handle Regular Capture
        elif move.piece_captured:
            self.pieces[opponent].remove(move.to_square)
            self.zobrist_hash ^= ZOBRIST_PIECES[move.to_square][move.piece_captured]

        # Handle Promotion
        if move.promotion:
            promo_piece = color | move.promotion
            self.board[move.to_square] = promo_piece
            self.pieces[color].add(move.to_square)
            self.zobrist_hash ^= ZOBRIST_PIECES[move.to_square][promo_piece]
        else:
            self.board[move.to_square] = move.piece_moved
            self.pieces[color].add(move.to_square)
            self.zobrist_hash ^= ZOBRIST_PIECES[move.to_square][move.piece_moved]

        # Update King Square
        if (move.piece_moved & TYPE_MASK) == KING:
            self.king_square[color] = move.to_square

        # Handle Castling Rook Movement
        if move.is_castling:
            if move.to_square == 6:  # WK
                self.board[7] = EMPTY
                self.board[5] = WHITE | ROOK
                self.pieces[WHITE].remove(7)
                self.pieces[WHITE].add(5)
                self.zobrist_hash ^= ZOBRIST_PIECES[7][WHITE | ROOK]
                self.zobrist_hash ^= ZOBRIST_PIECES[5][WHITE | ROOK]
            elif move.to_square == 2:  # WQ
                self.board[0] = EMPTY
                self.board[3] = WHITE | ROOK
                self.pieces[WHITE].remove(0)
                self.pieces[WHITE].add(3)
                self.zobrist_hash ^= ZOBRIST_PIECES[0][WHITE | ROOK]
                self.zobrist_hash ^= ZOBRIST_PIECES[3][WHITE | ROOK]
            elif move.to_square == 62:  # BK
                self.board[63] = EMPTY
                self.board[61] = BLACK | ROOK
                self.pieces[BLACK].remove(63)
                self.pieces[BLACK].add(61)
                self.zobrist_hash ^= ZOBRIST_PIECES[63][BLACK | ROOK]
                self.zobrist_hash ^= ZOBRIST_PIECES[61][BLACK | ROOK]
            elif move.to_square == 58:  # BQ
                self.board[56] = EMPTY
                self.board[59] = BLACK | ROOK
                self.pieces[BLACK].remove(56)
                self.pieces[BLACK].add(59)
                self.zobrist_hash ^= ZOBRIST_PIECES[56][BLACK | ROOK]
                self.zobrist_hash ^= ZOBRIST_PIECES[59][BLACK | ROOK]

        # 2. Update Castling Rights
        # King movement clears castling rights
        if (move.piece_moved & TYPE_MASK) == KING:
            if color == WHITE:
                self.castling_rights &= ~(WK | WQ)
            else:
                self.castling_rights &= ~(BK | BQ)

        # Rook movement clears specific castling rights
        elif (move.piece_moved & TYPE_MASK) == ROOK:
            if color == WHITE:
                if move.from_square == 7:
                    self.castling_rights &= ~WK
                elif move.from_square == 0:
                    self.castling_rights &= ~WQ
            else:
                if move.from_square == 63:
                    self.castling_rights &= ~BK
                elif move.from_square == 56:
                    self.castling_rights &= ~BQ

        # Capture of Rook clears opponent's castling rights
        if move.piece_captured and (move.piece_captured & TYPE_MASK) == ROOK:
            if opponent == WHITE:
                if move.to_square == 7:
                    self.castling_rights &= ~WK
                elif move.to_square == 0:
                    self.castling_rights &= ~WQ
            else:
                if move.to_square == 63:
                    self.castling_rights &= ~BK
                elif move.to_square == 56:
                    self.castling_rights &= ~BQ

        # 3. Update En Passant Target Square
        if move.is_double_push:
            self.en_passant_square = (move.from_square + move.to_square) // 2
        else:
            self.en_passant_square = None

        # 4. Update Halfmove Clock
        if (move.piece_moved & TYPE_MASK) == PAWN or move.piece_captured:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # 5. Update Fullmove Number
        if color == BLACK:
            self.fullmove_number += 1

        # 6. Switch Turns
        self.turn = opponent

        # XOR back in new volatile properties
        self.zobrist_hash ^= ZOBRIST_CASTLING[self.castling_rights]
        if self.en_passant_square is not None:
            self.zobrist_hash ^= ZOBRIST_EP[self.en_passant_square]

    def unmake_move(self, move, state):
        color = self.turn  # Currently it's the other player's turn, so self.turn is opponent
        opponent = color
        color = BLACK if opponent == WHITE else WHITE  # Color that made the original move

        # 1. Restore Turn and Metadata
        self.turn = color
        self.en_passant_square = state.en_passant_square
        self.castling_rights = state.castling_rights
        self.halfmove_clock = state.halfmove_clock
        self.zobrist_hash = state.zobrist_hash

        if color == BLACK:
            self.fullmove_number -= 1

        # 2. Revert Board and Piece lists
        self.board[move.to_square] = EMPTY
        self.pieces[color].remove(move.to_square)

        # Restore Moved Piece
        self.board[move.from_square] = move.piece_moved
        self.pieces[color].add(move.from_square)

        # Restore Captured Piece
        if move.is_en_passant:
            cap_sq = move.to_square - 8 if color == WHITE else move.to_square + 8
            self.board[cap_sq] = move.piece_captured
            self.pieces[opponent].add(cap_sq)
        elif move.piece_captured:
            self.board[move.to_square] = move.piece_captured
            self.pieces[opponent].add(move.to_square)

        # Restore King Square
        if (move.piece_moved & TYPE_MASK) == KING:
            self.king_square[color] = move.from_square

        # Restore Castling Rook Movement
        if move.is_castling:
            if move.to_square == 6:  # WK
                self.board[5] = EMPTY
                self.board[7] = WHITE | ROOK
                self.pieces[WHITE].remove(5)
                self.pieces[WHITE].add(7)
            elif move.to_square == 2:  # WQ
                self.board[3] = EMPTY
                self.board[0] = WHITE | ROOK
                self.pieces[WHITE].remove(3)
                self.pieces[WHITE].add(0)
            elif move.to_square == 62:  # BK
                self.board[61] = EMPTY
                self.board[63] = BLACK | ROOK
                self.pieces[BLACK].remove(61)
                self.pieces[BLACK].add(63)
            elif move.to_square == 58:  # BQ
                self.board[59] = EMPTY
                self.board[56] = BLACK | ROOK
                self.pieces[BLACK].remove(59)
                self.pieces[BLACK].add(56)


    def print_board(self):
        """Prints a text representation of the board."""
        print("  +-----------------+")
        for rank in range(7, -1, -1):
            row_str = f"{rank + 1} | "
            for file in range(8):
                sq = rank * 8 + file
                piece = self.board[sq]
                if piece == EMPTY:
                    char = "."
                else:
                    color = piece & COLOR_MASK
                    ptype = piece & TYPE_MASK
                    char = {
                        PAWN: 'p', KNIGHT: 'n', BISHOP: 'b',
                        ROOK: 'r', QUEEN: 'q', KING: 'k'
                    }[ptype]
                    if color == WHITE:
                        char = char.upper()
                row_str += char + " "
            row_str += "|"
            print(row_str)
        print("  +-----------------+")
        print("    a b c d e f g h")
