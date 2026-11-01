#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pywebview>=5"]
# ///
"""
AI Checkers Game - Human vs AI (American checkers / English draughts rules)

Engine:
  - Full rules: kings, multi-jump captures, mandatory capture, promotion ends
    the move, 40-move (80 ply) no-progress draw and threefold repetition.
  - Search: iterative deepening negamax with alpha-beta + principal variation
    search, Zobrist-hashed transposition table, killer moves, history
    heuristic, capture quiescence (never evaluate while a jump is pending),
    forced-move extension, repetition detection, and a per-move time limit.
  - Evaluation heuristics (after Samuel's checkers features and the Chinook
    papers): material with king premium, phase-scaled advancement, back-rank /
    bridge defence, center control, king centralization, runaway checkers,
    trade-down when ahead, endgame mop-up and double-corner defence.

UI:
  - Native WebKit window via pywebview.  Run with:  uv run checkers.py
  - See CHECKERS.md for directions and design notes.
"""

import argparse
import random
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Board representation
# ---------------------------------------------------------------------------
# A board is a flat list of 64 ints, index = row * 8 + col. Only dark squares
# ((row + col) odd) are ever occupied. Red (human by default) starts on rows
# 0-2 and moves toward row 7; Black (AI) starts on rows 5-7 and moves to row 0.

EMPTY = 0
RED_MAN, RED_KING = 1, 2
BLACK_MAN, BLACK_KING = -1, -2
RED, BLACK = 1, -1

MAN_VALUE = 100
KING_VALUE = 150
WIN_SCORE = 100000
DRAW_PLIES = 80  # 40 moves each without a capture or a man move

DARK = [r * 8 + c for r in range(8) for c in range(8) if (r + c) % 2 == 1]
DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MAN_DIRS = {RED: (2, 3), BLACK: (0, 1)}  # indices into DIRS
KING_DIRS = (0, 1, 2, 3)


def _build_tables():
    step = [[-1] * 4 for _ in range(64)]
    jump = [[-1] * 4 for _ in range(64)]
    for sq in DARK:
        r, c = divmod(sq, 8)
        for d, (dr, dc) in enumerate(DIRS):
            if 0 <= r + dr < 8 and 0 <= c + dc < 8:
                step[sq][d] = (r + dr) * 8 + c + dc
            if 0 <= r + 2 * dr < 8 and 0 <= c + 2 * dc < 8:
                jump[sq][d] = (r + 2 * dr) * 8 + c + 2 * dc
    return step, jump


STEP, JUMP = _build_tables()


def initial_board():
    board = [EMPTY] * 64
    for sq in DARK:
        r = sq // 8
        if r <= 2:
            board[sq] = RED_MAN
        elif r >= 5:
            board[sq] = BLACK_MAN
    return board


def side_of(piece):
    return RED if piece > 0 else BLACK if piece < 0 else 0


def is_king(piece):
    return piece in (RED_KING, BLACK_KING)


def promotion_row(side):
    return 7 if side == RED else 0


# ---------------------------------------------------------------------------
# Move generation
# ---------------------------------------------------------------------------
# A move is a tuple (path, captured): path is a tuple of squares visited
# (start, landing1, landing2, ...), captured is a tuple of jumped squares.

def _jumps_from(board, start, piece, side):
    """All maximal jump sequences for the piece on `start`."""
    dirs = KING_DIRS if is_king(piece) else MAN_DIRS[side]
    results = []
    board[start] = EMPTY  # the moving piece's origin is vacated during the jump

    def dfs(sq, path, captured):
        extended = False
        for d in dirs:
            mid, land = STEP[sq][d], JUMP[sq][d]
            if land < 0 or board[land] != EMPTY or mid in captured:
                continue
            if board[mid] * side >= 0:  # empty or own piece
                continue
            new_path = path + (land,)
            new_captured = captured + (mid,)
            extended = True
            # A man reaching the king row is crowned and the move ends.
            if not is_king(piece) and land // 8 == promotion_row(side):
                results.append((new_path, new_captured))
            else:
                dfs(land, new_path, new_captured)
        if not extended and captured:
            results.append((path, captured))

    dfs(start, (start,), ())
    board[start] = piece
    return results


def generate_moves(board, side):
    """Legal moves for `side`. Captures are mandatory, so if any exist only
    captures are returned."""
    captures = []
    for sq in DARK:
        piece = board[sq]
        if piece * side > 0:
            dirs = KING_DIRS if is_king(piece) else MAN_DIRS[side]
            for d in dirs:
                mid, land = STEP[sq][d], JUMP[sq][d]
                if land >= 0 and board[land] == EMPTY and board[mid] * side < 0:
                    captures.extend(_jumps_from(board, sq, piece, side))
                    break
    if captures:
        return captures
    moves = []
    for sq in DARK:
        piece = board[sq]
        if piece * side > 0:
            dirs = KING_DIRS if is_king(piece) else MAN_DIRS[side]
            for d in dirs:
                to = STEP[sq][d]
                if to >= 0 and board[to] == EMPTY:
                    moves.append(((sq, to), ()))
    return moves


def make_move(board, move):
    """Apply `move` in place. Returns undo info for unmake_move."""
    path, captured = move
    src, dst = path[0], path[-1]
    piece = board[src]
    board[src] = EMPTY
    promoted = piece
    if not is_king(piece) and dst // 8 == promotion_row(side_of(piece)):
        promoted = piece * 2
    board[dst] = promoted
    taken = [board[sq] for sq in captured]
    for sq in captured:
        board[sq] = EMPTY
    return piece, promoted, taken


def unmake_move(board, move, undo):
    path, captured = move
    piece, _, taken = undo
    board[path[-1]] = EMPTY
    for sq, p in zip(captured, taken):
        board[sq] = p
    board[path[0]] = piece


def apply_move(board, move):
    """Return a new board with `move` applied."""
    new_board = board[:]
    make_move(new_board, move)
    return new_board


def is_irreversible(board, move):
    """Captures and man moves can never be undone (they reset draw counters)."""
    return bool(move[1]) or not is_king(board[move[0][0]])


# ---------------------------------------------------------------------------
# Zobrist hashing
# ---------------------------------------------------------------------------

_rng = random.Random(20261101)
_PIECE_INDEX = {RED_MAN: 0, RED_KING: 1, BLACK_MAN: 2, BLACK_KING: 3}
ZOBRIST = [[_rng.getrandbits(64) for _ in range(4)] for _ in range(64)]
ZOBRIST_BLACK_TO_MOVE = _rng.getrandbits(64)


def zobrist_hash(board, side):
    h = ZOBRIST_BLACK_TO_MOVE if side == BLACK else 0
    for sq in DARK:
        p = board[sq]
        if p:
            h ^= ZOBRIST[sq][_PIECE_INDEX[p]]
    return h


def hash_after(h, move, undo):
    """Incrementally update hash `h` for a move (also flips side to move)."""
    path, captured = move
    piece, promoted, taken = undo
    h ^= ZOBRIST[path[0]][_PIECE_INDEX[piece]]
    h ^= ZOBRIST[path[-1]][_PIECE_INDEX[promoted]]
    for sq, p in zip(captured, taken):
        h ^= ZOBRIST[sq][_PIECE_INDEX[p]]
    return h ^ ZOBRIST_BLACK_TO_MOVE


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Bonus by rows advanced (0 = still on own back row, 6 = one step from crown).
ADVANCE_BONUS = [0, 0, 1, 2, 4, 7, 11, 0]
CENTER = {r * 8 + c for r in (3, 4) for c in range(2, 6) if (r + c) % 2 == 1}
NEAR_CENTER = {r * 8 + c for r in (2, 3, 4, 5) for c in range(1, 7) if (r + c) % 2 == 1}
# Back-rank "bridge" guard squares, the two squares that stop most early
# kingings. Point-symmetric between the two sides.
BRIDGE = {RED: {0 * 8 + 1, 0 * 8 + 5}, BLACK: {7 * 8 + 6, 7 * 8 + 2}}
DOUBLE_CORNERS = {0 * 8 + 1, 1 * 8 + 0, 6 * 8 + 7, 7 * 8 + 6}
KING_CENTRALITY = [0] * 64
for _sq in DARK:
    _r, _c = divmod(_sq, 8)
    KING_CENTRALITY[_sq] = int(8 - 2 * (abs(_r - 3.5) + abs(_c - 3.5)))


def _is_runaway(board, sq, side):
    """True if no enemy piece stands in the forward cone of the man on `sq`,
    so nothing can stop it from being crowned (approximation)."""
    r, c = divmod(sq, 8)
    target = promotion_row(side)
    dr = 1 if side == RED else -1
    dist = 0
    row = r
    while row != target:
        row += dr
        dist += 1
        for col in range(max(0, c - dist - 1), min(8, c + dist + 2)):
            if board[row * 8 + col] * side < 0:
                return False
    return True


def evaluate(board, side):
    """Static evaluation from the point of view of `side` (higher = better)."""
    material = {RED: 0, BLACK: 0}
    positional = {RED: 0, BLACK: 0}
    kings = {RED: [], BLACK: []}
    men = {RED: [], BLACK: []}
    for sq in DARK:
        p = board[sq]
        if p == EMPTY:
            continue
        s = RED if p > 0 else BLACK
        if is_king(p):
            material[s] += KING_VALUE
            kings[s].append(sq)
        else:
            material[s] += MAN_VALUE
            men[s].append(sq)

    total = len(men[RED]) + len(men[BLACK]) + len(kings[RED]) + len(kings[BLACK])
    endgame = total <= 10
    # Advancement matters more as the board empties.
    adv_scale = 1.0 + (24 - total) / 12.0

    for s in (RED, BLACK):
        enemy = -s
        pos = 0
        enemy_kings = len(kings[enemy])
        for sq in men[s]:
            r = sq // 8
            adv = r if s == RED else 7 - r
            pos += ADVANCE_BONUS[adv] * adv_scale
            if adv == 0 and total > 8:
                # Back-rank defence keeps the opponent from crowning cheaply.
                pos += 6 + (4 if sq in BRIDGE[s] else 0)
            if sq in CENTER:
                pos += 5
            elif sq in NEAR_CENTER:
                pos += 2
            if adv >= 4 and _is_runaway(board, sq, s):
                bonus = 40 - 8 * (7 - adv)
                pos += bonus // 2 if enemy_kings else bonus
        for sq in kings[s]:
            pos += KING_CENTRALITY[sq]
        positional[s] = pos

    score_red = material[RED] - material[BLACK] + positional[RED] - positional[BLACK]

    # Trade down when ahead: the same lead is worth more with fewer pieces.
    lead = material[RED] - material[BLACK]
    if abs(lead) >= MAN_VALUE:
        score_red += (1 if lead > 0 else -1) * (24 - total) * 3

    # Endgame: the winning side hunts with its kings, the losing side
    # retreats its kings to the double corners where they are hardest to trap.
    if endgame and abs(lead) >= MAN_VALUE // 2:
        winner = RED if lead > 0 else BLACK
        loser = -winner
        targets = men[loser] + kings[loser]
        hunt = 0
        for k in kings[winner]:
            kr, kc = divmod(k, 8)
            hunt -= min((max(abs(kr - t // 8), abs(kc - t % 8)) for t in targets), default=0)
        defend = sum(10 for k in kings[loser] if k in DOUBLE_CORNERS)
        score_red += winner * (hunt * 3) - winner * defend

    score_red = int(score_red)
    return score_red if side == RED else -score_red


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

EXACT, LOWER, UPPER = 0, 1, 2


class SearchTimeout(Exception):
    pass


class Engine:
    """Iterative-deepening alpha-beta searcher with a transposition table."""

    def __init__(self, time_limit=2.0, max_depth=40):
        self.time_limit = time_limit
        self.max_depth = max_depth
        self.tt = {}
        self.history = {}
        self.killers = []
        self.nodes = 0
        self.deadline = 0.0
        self.stop_event = None

    def _check_time(self):
        if (self.nodes & 1023) == 0:
            if time.perf_counter() > self.deadline or (
                    self.stop_event is not None and self.stop_event.is_set()):
                raise SearchTimeout()

    def _order(self, moves, tt_move, ply):
        killers = self.killers[ply] if ply < len(self.killers) else ()

        def key(m):
            if m == tt_move:
                return 10 ** 9
            if m[1]:
                return 10 ** 8 + len(m[1])
            if m in killers:
                return 10 ** 7
            return self.history.get(m, 0)

        moves.sort(key=key, reverse=True)
        return moves

    def _negamax(self, board, side, depth, alpha, beta, ply, h, reps, quiet):
        """quiet = plies since the last irreversible move (for draw rules)."""
        self.nodes += 1
        self._check_time()

        # Repetition (any repeat within the search counts as a draw) and
        # the 40-move rule.
        if ply > 0 and (reps.get(h, 0) > 0 or quiet >= DRAW_PLIES):
            return 0

        alpha_orig = alpha
        entry = self.tt.get(h)
        tt_move = None
        if entry is not None:
            e_depth, e_val, e_flag, tt_move = entry
            if e_depth >= depth and ply > 0:
                if e_val > WIN_SCORE - 1000:
                    e_val -= ply
                elif e_val < -WIN_SCORE + 1000:
                    e_val += ply
                if e_flag == EXACT:
                    return e_val
                if e_flag == LOWER and e_val >= beta:
                    return e_val
                if e_flag == UPPER and e_val <= alpha:
                    return e_val

        moves = generate_moves(board, side)
        if not moves:
            return -WIN_SCORE + ply  # no legal moves = loss

        is_capture = bool(moves[0][1])
        if depth <= 0 and not is_capture:
            return evaluate(board, side)
        # Quiescence: positions with a pending jump are never evaluated
        # statically; forced (single) moves are free extensions.
        if len(moves) == 1 and ply < 60:
            next_depth = depth
        elif depth <= 0:
            next_depth = 0
        else:
            next_depth = depth - 1

        self._order(moves, tt_move, ply)
        best_val = -WIN_SCORE * 2
        best_move = moves[0]
        reps[h] = reps.get(h, 0) + 1
        try:
            for i, move in enumerate(moves):
                irreversible = is_irreversible(board, move)
                undo = make_move(board, move)
                child_h = hash_after(h, move, undo)
                child_reps = {} if irreversible else reps
                child_quiet = 0 if irreversible else quiet + 1
                try:
                    if i == 0:
                        val = -self._negamax(board, -side, next_depth, -beta, -alpha,
                                             ply + 1, child_h, child_reps, child_quiet)
                    else:
                        # Principal variation search: null window first.
                        val = -self._negamax(board, -side, next_depth, -alpha - 1, -alpha,
                                             ply + 1, child_h, child_reps, child_quiet)
                        if alpha < val < beta:
                            val = -self._negamax(board, -side, next_depth, -beta, -alpha,
                                                 ply + 1, child_h, child_reps, child_quiet)
                finally:
                    unmake_move(board, move, undo)
                if val > best_val:
                    best_val = val
                    best_move = move
                if val > alpha:
                    alpha = val
                if alpha >= beta:
                    if not move[1]:
                        while len(self.killers) <= ply:
                            self.killers.append([])
                        k = self.killers[ply]
                        if move not in k:
                            k.insert(0, move)
                            del k[2:]
                        self.history[move] = self.history.get(move, 0) + depth * depth
                    break
        finally:
            reps[h] -= 1
            if reps[h] == 0:
                del reps[h]

        if best_val <= alpha_orig:
            flag = UPPER
        elif best_val >= beta:
            flag = LOWER
        else:
            flag = EXACT
        store_val = best_val
        if store_val > WIN_SCORE - 1000:
            store_val += ply
        elif store_val < -WIN_SCORE + 1000:
            store_val -= ply
        self.tt[h] = (depth, store_val, flag, best_move)
        return best_val

    def principal_variation(self, board, side, max_len=12):
        board = board[:]
        pv = []
        seen = set()
        h = zobrist_hash(board, side)
        while len(pv) < max_len and h not in seen:
            seen.add(h)
            entry = self.tt.get(h)
            if entry is None or entry[3] not in generate_moves(board, side):
                break
            move = entry[3]
            pv.append(move)
            undo = make_move(board, move)
            h = hash_after(h, move, undo)
            side = -side
        return pv

    def search(self, board, side, game_hashes=(), quiet=0, stop_event=None, on_iteration=None):
        """Find the best move for `side`.

        game_hashes: hashes of positions since the last irreversible move
        (excluding the current one), used for repetition detection.
        Returns (move, info) where info has depth, score, nodes, time, pv.
        """
        board = board[:]
        moves = generate_moves(board, side)
        info = {"depth": 0, "score": 0, "nodes": 0, "time": 0.0, "pv": []}
        if not moves:
            return None, info
        if len(moves) == 1:
            info["pv"] = moves
            return moves[0], info

        if len(self.tt) > 2_000_000:
            self.tt.clear()
        self.history = {k: v // 4 for k, v in self.history.items() if v >= 4}
        self.killers = []
        self.nodes = 0
        self.stop_event = stop_event
        start = time.perf_counter()
        self.deadline = start + self.time_limit
        h = zobrist_hash(board, side)
        reps = {}
        for gh in game_hashes:
            reps[gh] = reps.get(gh, 0) + 1

        best_move = moves[0]
        for depth in range(1, self.max_depth + 1):
            try:
                score = self._negamax(board, side, depth, -WIN_SCORE * 2, WIN_SCORE * 2,
                                      0, h, reps, quiet)
            except SearchTimeout:
                break
            entry = self.tt.get(h)
            if entry is not None:
                best_move = entry[3]
            info.update(depth=depth, score=score, nodes=self.nodes,
                        time=time.perf_counter() - start)
            if on_iteration:
                on_iteration(dict(info))
            if abs(score) > WIN_SCORE - 1000:
                break  # forced win/loss found
            # Don't start an iteration we almost certainly can't finish.
            if time.perf_counter() - start > self.time_limit * 0.55:
                break
        info["nodes"] = self.nodes
        info["time"] = time.perf_counter() - start
        info["pv"] = self.principal_variation(board, side)
        if not info["pv"] or info["pv"][0] != best_move:
            info["pv"] = [best_move]
        return best_move, info


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

class Game:
    def __init__(self, human=RED, time_limit=2.0):
        self.human = human
        self.ai = -human
        self.engine = Engine(time_limit=time_limit)
        self.reset()

    def reset(self):
        self.board = initial_board()
        self.side = RED  # Red moves first
        self.quiet = 0
        self.rep_hashes = []      # hashes since last irreversible move
        self.undo_stack = []
        self.move_log = []        # (side, move)
        self.last_info = None

    def legal_moves(self):
        return generate_moves(self.board, self.side)

    def play(self, move):
        self.undo_stack.append((self.board[:], self.side, self.quiet,
                                list(self.rep_hashes), len(self.move_log)))
        if is_irreversible(self.board, move):
            self.quiet = 0
            self.rep_hashes = []
        else:
            self.quiet += 1
            self.rep_hashes.append(zobrist_hash(self.board, self.side))
        make_move(self.board, move)
        self.move_log.append((self.side, move))
        self.side = -self.side

    def undo(self):
        """Undo back to the last position where it was the human's turn."""
        while self.undo_stack:
            self.board, self.side, self.quiet, self.rep_hashes, n = self.undo_stack.pop()
            del self.move_log[n:]
            if self.side == self.human:
                return True
        return False

    def result(self):
        """None while the game is running, else (winner_or_0, reason)."""
        if not self.legal_moves():
            who = "Red" if self.side == RED else "Black"
            return -self.side, f"{who} has no legal moves"
        if self.quiet >= DRAW_PLIES:
            return 0, "40 moves without a capture or man move"
        h = zobrist_hash(self.board, self.side)
        if self.rep_hashes.count(h) >= 2:
            return 0, "threefold repetition"
        return None

    def ai_move(self, stop_event=None, on_iteration=None):
        move, info = self.engine.search(self.board, self.side, self.rep_hashes,
                                        self.quiet, stop_event, on_iteration)
        self.last_info = info
        if move is not None:
            self.play(move)
        return move, info


# ---------------------------------------------------------------------------
# Move notation
# ---------------------------------------------------------------------------

def square_name(sq):
    return f"{sq // 8 + 1},{sq % 8 + 1}"


def move_str(move):
    sep = " x " if move[1] else " - "
    return sep.join(square_name(sq) for sq in move[0])


# ---------------------------------------------------------------------------
# WebKit UI (pywebview -> WKWebView on macOS)
# ---------------------------------------------------------------------------
# Mirrors the webkit-cl design: Python plays the role of the Lisp host, the
# js_api object plays the role of register-handler (JS calls
# `await pywebview.api.<name>(...)` instead of `window.webkit_cl.invoke`), and
# results come back to JS as JSON automatically.

GUI_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Checkers</title>
<style>
  :root {
    --bg: #14161a; --panel: #1d2026; --text: #eceff4; --muted: #8b93a1;
    --light: #e9d8b4; --dark: #8a5a3b; --sel: #ffd166; --hint: #06d6a0;
    --last: rgba(255, 209, 102, .35);
    --red: #d64545; --red-edge: #8f2222; --black: #2b2d33; --black-edge: #0c0d0f;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    background: var(--bg); color: var(--text); height: 100vh;
    display: flex; align-items: center; justify-content: center; gap: 28px;
    user-select: none; -webkit-user-select: none; padding: 20px;
  }
  #board {
    display: grid; grid-template-columns: repeat(8, 1fr);
    width: min(80vh, 62vw); aspect-ratio: 1; border: 6px solid #3a2a1e;
    border-radius: 6px; box-shadow: 0 12px 40px rgba(0,0,0,.5);
  }
  .sq { position: relative; display: flex; align-items: center; justify-content: center; }
  .sq.l { background: var(--light); }
  .sq.d { background: var(--dark); }
  .sq.last::before { content: ""; position: absolute; inset: 0; background: var(--last); }
  .sq.target { cursor: pointer; }
  .sq.target::after {
    content: ""; position: absolute; width: 30%; height: 30%; border-radius: 50%;
    background: var(--hint); opacity: .85;
  }
  .piece {
    position: relative; width: 78%; height: 78%; border-radius: 50%; z-index: 1;
    display: flex; align-items: center; justify-content: center;
    font-size: min(4vh, 3vw); transition: transform .12s;
  }
  .piece.red { background: radial-gradient(circle at 35% 30%, #f07070, var(--red));
               border: 3px solid var(--red-edge); }
  .piece.black { background: radial-gradient(circle at 35% 30%, #50535c, var(--black));
                 border: 3px solid var(--black-edge); }
  .piece.king::after { content: "♛"; color: #ffd166; text-shadow: 0 1px 2px #000; }
  .piece.movable { cursor: pointer; }
  .piece.movable:hover { transform: scale(1.06); }
  .piece.selected { box-shadow: 0 0 0 4px var(--sel); }
  #panel {
    width: 300px; background: var(--panel); border-radius: 10px; padding: 18px;
    display: flex; flex-direction: column; gap: 14px; max-height: min(80vh, 640px);
  }
  h1 { font-size: 20px; }
  #status { font-size: 15px; min-height: 40px; }
  #status.thinking::after { content: " ⏳"; }
  .row { display: flex; gap: 8px; }
  button {
    flex: 1; padding: 8px 10px; border: 0; border-radius: 6px; cursor: pointer;
    background: #343944; color: var(--text); font-size: 13px;
  }
  button:hover { background: #434a57; }
  button:disabled { opacity: .4; cursor: default; }
  label { font-size: 13px; color: var(--muted); display: flex; flex-direction: column; gap: 4px; }
  #stats { font: 12px ui-monospace, Menlo, monospace; color: var(--muted); white-space: pre-wrap; }
  #log { flex: 1; overflow-y: auto; font: 12px ui-monospace, Menlo, monospace;
         background: #15171b; border-radius: 6px; padding: 8px; min-height: 80px; }
  #log div:nth-child(odd) { color: #f3a3a3; }
  #log div:nth-child(even) { color: #b8c0cc; }
</style>
</head>
<body>
  <div id="board"></div>
  <div id="panel">
    <h1>Checkers</h1>
    <div id="status">Loading…</div>
    <div class="row">
      <button id="new">New game</button>
      <button id="undo">Undo</button>
      <button id="hint">Hint</button>
    </div>
    <label>Play as
      <select id="color">
        <option value="1">Red (moves first)</option>
        <option value="-1">Black</option>
      </select>
    </label>
    <label>AI think time: <span id="tval">2.0</span>s
      <input id="time" type="range" min="0.2" max="10" step="0.2" value="2">
    </label>
    <div id="stats"></div>
    <div id="log"></div>
  </div>
<script>
let state = null;
let partial = [];   // squares clicked so far for the move being built
let hintMove = null;
let busy = false;
const $ = id => document.getElementById(id);

function render() {
  const b = $("board");
  b.innerHTML = "";
  const flip = state.human === -1;
  const last = new Set(state.last_move || []);
  const candidates = state.your_turn && !busy ? state.moves.filter(m =>
      partial.every((sq, i) => m[i] === sq)) : [];
  const movable = new Set(candidates.map(m => m[0]));
  const targets = new Set(partial.length ? candidates.filter(m => m.length > partial.length)
                                                    .map(m => m[partial.length]) : []);
  const hintSet = new Set(hintMove || []);
  for (let i = 0; i < 64; i++) {
    const sq = flip ? 63 - i : i;
    const r = Math.floor(sq / 8), c = sq % 8;
    const cell = document.createElement("div");
    cell.className = "sq " + ((r + c) % 2 ? "d" : "l");
    if (last.has(sq) || hintSet.has(sq)) cell.classList.add("last");
    const p = state.board[sq];
    const shownAt = partial.length ? partial[partial.length - 1] : -1;
    const piece = (partial.length && sq === partial[0]) ? 0 :
                  (sq === shownAt && partial.length > 1) ? state.board[partial[0]] : p;
    if (piece) {
      const el = document.createElement("div");
      el.className = "piece " + (piece > 0 ? "red" : "black") + (Math.abs(piece) === 2 ? " king" : "");
      if (movable.has(sq) && partial.length === 0) el.classList.add("movable");
      if (partial.length && sq === shownAt) el.classList.add("selected");
      cell.appendChild(el);
    }
    if (targets.has(sq)) cell.classList.add("target");
    cell.onclick = () => clickSquare(sq, movable, targets);
    b.appendChild(cell);
  }
  $("log").innerHTML = state.log.map((m, i) =>
      `<div>${Math.floor(i / 2) + 1}${i % 2 ? "…" : "."} ${m}</div>`).join("");
  $("log").scrollTop = 1e9;
  $("undo").disabled = busy || !state.can_undo;
  $("hint").disabled = busy || !state.your_turn;
  if (state.info) {
    const s = state.info;
    $("stats").textContent =
      `AI depth ${s.depth}   eval ${s.score > 0 ? "+" : ""}${s.score}\n` +
      `${s.nodes.toLocaleString()} nodes in ${s.time.toFixed(2)}s\n` +
      (s.pv ? `PV: ${s.pv}` : "");
  }
}

function setStatus(text, thinking) {
  $("status").textContent = text;
  $("status").classList.toggle("thinking", !!thinking);
}

function statusFromState() {
  if (state.result) setStatus(state.result);
  else if (state.your_turn)
    setStatus(state.must_capture ? "Your move — a capture is mandatory." : "Your move.");
  else setStatus("AI is thinking", true);
}

async function clickSquare(sq, movable, targets) {
  if (busy || !state.your_turn) return;
  if (partial.length === 0) {
    if (movable.has(sq)) partial = [sq];
  } else if (targets.has(sq)) {
    partial.push(sq);
    const done = state.moves.find(m => m.length === partial.length &&
                                       m.every((s, i) => s === partial[i]));
    if (done) {
      const path = partial;
      partial = []; hintMove = null;
      busy = true;
      state = await pywebview.api.human_move(path);
      busy = false;
      render(); statusFromState();
      await maybeAiMove();
      return;
    }
  } else if (partial.length === 1) {
    // Clicking another movable piece switches selection; anything else deselects.
    const starts = new Set(state.moves.map(m => m[0]));
    partial = starts.has(sq) && sq !== partial[0] ? [sq] : [];
  }
  render();
}

async function maybeAiMove() {
  if (state.result || state.your_turn) return;
  busy = true; render(); setStatus("AI is thinking", true);
  state = await pywebview.api.ai_move();
  busy = false; render(); statusFromState();
}

async function refresh(newState) {
  state = newState; partial = []; hintMove = null;
  render(); statusFromState();
  await maybeAiMove();
}

window.addEventListener("pywebviewready", async () => {
  $("new").onclick = async () =>
    refresh(await pywebview.api.new_game(parseInt($("color").value)));
  $("color").onchange = $("new").onclick;
  $("undo").onclick = async () => refresh(await pywebview.api.undo());
  $("hint").onclick = async () => {
    busy = true; render(); setStatus("Looking for a hint", true);
    hintMove = await pywebview.api.hint();
    busy = false; render(); statusFromState();
  };
  $("time").oninput = () => {
    $("tval").textContent = parseFloat($("time").value).toFixed(1);
    pywebview.api.set_time(parseFloat($("time").value));
  };
  const s = await pywebview.api.state();
  $("time").value = s.time_limit; $("tval").textContent = s.time_limit.toFixed(1);
  refresh(s);
});
</script>
</body>
</html>
"""


class GuiApi:
    """Methods exposed to JavaScript as pywebview.api.<name>. pywebview runs
    each call on a worker thread, so a long AI search doesn't block the UI."""

    def __init__(self, time_limit):
        self._game = Game(human=RED, time_limit=time_limit)
        self._lock = threading.Lock()

    def _state(self):
        g = self._game
        res = g.result()
        if res is None:
            result = None
        else:
            winner, reason = res
            if winner == 0:
                result = f"Draw — {reason}."
            elif winner == g.human:
                result = f"You win! ({reason})"
            else:
                result = f"AI wins. ({reason})"
        legal = g.legal_moves() if res is None else []
        info = None
        if g.last_info:
            i = g.last_info
            # score is from the AI's point of view (positive = AI is better)
            info = {"depth": i["depth"], "score": i["score"], "nodes": i["nodes"],
                    "time": i["time"], "pv": "  ".join(move_str(m) for m in i["pv"][:6])}
        return {
            "board": g.board,
            "human": g.human,
            "your_turn": res is None and g.side == g.human,
            "moves": [list(m[0]) for m in legal] if g.side == g.human else [],
            "must_capture": bool(legal and legal[0][1]),
            "last_move": list(g.move_log[-1][1][0]) if g.move_log else [],
            "log": [move_str(m) for _, m in g.move_log],
            "can_undo": any(s == g.human for _, s, *_ in g.undo_stack),
            "result": result,
            "info": info,
            "time_limit": g.engine.time_limit,
        }

    def state(self):
        with self._lock:
            return self._state()

    def new_game(self, human=RED):
        with self._lock:
            time_limit = self._game.engine.time_limit
            self._game = Game(human=RED if human == RED else BLACK, time_limit=time_limit)
            return self._state()

    def set_time(self, seconds):
        with self._lock:
            self._game.engine.time_limit = max(0.1, float(seconds))

    def human_move(self, path):
        with self._lock:
            g = self._game
            if g.result() is None and g.side == g.human:
                for m in g.legal_moves():
                    if list(m[0]) == list(path):
                        g.play(m)
                        break
            return self._state()

    def ai_move(self):
        with self._lock:
            g = self._game
            if g.result() is None and g.side == g.ai:
                g.ai_move()
            return self._state()

    def hint(self):
        with self._lock:
            g = self._game
            if g.result() is not None or g.side != g.human:
                return None
            engine = Engine(time_limit=min(2.0, g.engine.time_limit))
            move, _ = engine.search(g.board, g.side, g.rep_hashes, g.quiet)
            return list(move[0]) if move else None

    def undo(self):
        with self._lock:
            self._game.undo()
            self._game.last_info = None
            return self._state()


def run_gui(time_limit):
    try:
        import webview
    except ImportError:
        sys.exit("Checkers needs pywebview. Run it with:  uv run checkers.py")
    webview.create_window("Checkers", html=GUI_HTML, js_api=GuiApi(time_limit),
                          width=1040, height=760, min_size=(760, 560),
                          background_color="#14161a")
    webview.start()


def main():
    parser = argparse.ArgumentParser(description="Checkers: human vs AI")
    parser.add_argument("--time", type=float, default=2.0,
                        help="AI think time per move in seconds (default 2)")
    args = parser.parse_args()
    run_gui(args.time)


if __name__ == "__main__":
    main()
