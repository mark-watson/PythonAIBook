# Checkers: Human vs AI

A single-file checkers game (`checkers.py`): a native macOS WebKit window
on top of an alpha-beta search engine. It plays American checkers
(English draughts) rules.

## Running

```bash
uv run checkers.py              # AI thinks 2 seconds per move
uv run checkers.py --time 5     # stronger: 5 seconds per move
```

You don't need to install anything first. `checkers.py` carries
[PEP 723](https://peps.python.org/pep-0723/) inline script metadata:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["pywebview>=5"]
# ///
```

`uv run` reads it, builds a cached private environment with `pywebview`
(and the PyObjC WebKit bindings it uses on macOS), and launches the game.
The first run takes a few seconds; later runs start immediately.

## How to play

- **You play Red by default.** Red starts at the top and moves first. Use
  the *Play as* menu to take Black instead; the AI then plays the opening move.
- **Moving:** click one of your pieces. Every piece that can legally move
  shows a hand cursor. Green dots mark where it can land; click one.
- **Multi-jumps:** click each landing square in turn. Only legal
  continuations are offered, and the move is sent once the jump sequence
  is complete.
- **Changing your mind:** click another movable piece to switch, or click
  anywhere else to deselect.
- **Buttons:**
  - **New game:** starts over (also triggered by changing *Play as*).
  - **Undo:** takes back your last move together with the AI's reply.
  - **Hint:** highlights the move the engine would play for you.
- **AI think time slider:** 0.2 to 10 seconds per move. Takes effect on the
  AI's next move.
- **Side panel:** shows the move list and, after each AI move, its search
  depth, evaluation (positive means the AI thinks it is ahead; 100 = one
  man), nodes searched, and the line it expects (principal variation).

Squares are written `row,col` from 1 to 8, with row 1 at the top of the
board when you play Red. `-` marks a step and `x` marks a jump, e.g.
`3,2 x 5,4 x 7,6`.

### Rules implemented

- Men move one square diagonally forward; kings move diagonally either way.
- **Captures are mandatory.** A jump must continue while further jumps are
  available, and you may choose any capture sequence (not necessarily the
  longest).
- A man reaching the far row is crowned, and **crowning ends the move**,
  even in the middle of a jump sequence.
- You lose when you have no legal move, either because you have no pieces
  left or because all of them are blocked.
- **Draws:** 40 moves per side without a capture or a man move, or the same
  position occurring three times.

## Design

`checkers.py` has five layers, from the bottom up:

```
┌────────────────────────────────────────────┐
│  GUI_HTML  (HTML/CSS/JS in a WKWebView)    │  rendering, clicks, move building
├──────────── pywebview js_api bridge ───────┤  await pywebview.api.<method>()
│  GuiApi    (bridge handlers, JSON state)   │  locking, game ↔ UI translation
├────────────────────────────────────────────┤
│  Game      (history, undo, draw rules)     │
├────────────────────────────────────────────┤
│  Engine    (search)  +  evaluate()         │
├────────────────────────────────────────────┤
│  Board / move generation / Zobrist hashing │
└────────────────────────────────────────────┘
```

### Board and move generation

- **Board:** a flat list of 64 ints (`index = row * 8 + col`): `1`/`2` for
  a Red man/king, `-1`/`-2` for a Black man/king, `0` for empty. A side is
  `+1` (Red) or `-1` (Black), so `piece * side > 0` means "my piece".
- **Neighbour tables:** `STEP[sq][dir]` and `JUMP[sq][dir]` are computed at
  import, so move generation is table lookups, not bounds checks.
- **Moves:** a move is `(path, captured)`. `path` is the tuple of squares
  visited (start, landing, landing, ...) and `captured` is the tuple of
  jumped squares. Jump sequences are found by depth-first search from each
  piece. The origin square is cleared first so a king can loop through it,
  and a piece already captured in the sequence can't be jumped again.
- **Search speed:** `make_move` / `unmake_move` change the board in place
  and return undo data, so the search never copies boards.
- **Verification:** move generation matches the published perft counts for
  the starting position: 7, 49, 302, 1469, 7361, 36768, 179740 at depths 1–7.

### Search (`Engine`)

- **Iterative deepening:** the engine searches depth 1, 2, 3, ... until the
  time limit. Each iteration seeds move ordering for the next, and the move
  from the last completed depth is played. It won't start a new iteration
  after 55% of the time budget is used, and it stops early on a forced
  win or loss.
- **Negamax with alpha-beta and principal variation search:** the first
  (expected best) move is searched with a full window; the rest get a
  cheap null-window probe and are re-searched only if they might beat it.
- **Transposition table:** results are stored in a dict keyed by 64-bit
  Zobrist hashes, which are updated incrementally on every move. Each entry
  holds depth, score, bound type (exact/lower/upper) and best move. Win
  scores are adjusted by distance from the root so the engine prefers the
  fastest win.
- **Move ordering:** in priority order:
  1. the move from the transposition table
  2. captures, longest first
  3. two killer moves per ply
  4. the history heuristic (bonus of depth² for each quiet move that
     caused a cutoff; it decays between moves)
- **Capture quiescence:** a position with a jump pending is never scored
  statically. Mandatory captures mean the line keeps going past the
  nominal depth until the position is quiet. This avoids the horizon
  effect, which is otherwise the main tactical weakness of a fixed-depth
  checkers search.
- **Forced-move extension:** when the side to move has exactly one legal
  move, it doesn't use up depth (capped at ply 60).
- **Draw detection inside the search:** if a position repeats along the
  current line, or across game history since the last irreversible move,
  it scores 0. So does reaching the 40-move limit. This lets the engine
  steer toward a draw when it is losing and avoid one when it is winning.
- **Interruptible:** the clock is checked every 1024 nodes and a timeout
  unwinds through an exception. The search runs on a pywebview worker
  thread, so the window stays responsive.

In pure Python the engine typically reaches **depth 7 in 0.3 s and depth 9
in 2 s** from the opening (plus the quiescence and forced-move extensions).
In self-play it won 4–0 against a stand-in for the original design
(material-only evaluation, fixed depth 4).

### Evaluation (`evaluate`)

The evaluation is a hand-tuned weighted sum of features, in the tradition of
Samuel's checkers program and Chinook. Chinook's evaluation had about 25
weighted features with separate weights per game phase. This one is smaller
but uses the same ideas. It is computed for Red and negated for Black.

| Feature | Weight / rule | Why |
|---|---|---|
| Material | man 100, king 150 | Kings are worth about 1.5 men. |
| Advancement | 0,0,1,2,4,7,11 by rows advanced, scaled by `1 + (24 − pieces)/12` | Advancing is risky early and decisive late. |
| Back rank | +6 per man on its own back row, +4 more on the two "bridge" guard squares; off when ≤ 8 pieces remain | Stops the opponent from crowning cheaply. |
| Center control | +5 for the four central squares, +2 near center | Central men restrict the opponent and support attacks. |
| King centralization | up to +8, by distance from the center | Central kings have more mobility and can reach either flank. |
| Runaway checkers | up to +40 for a man whose forward cone has no enemy pieces (halved if the enemy has kings) | A man nobody can stop is almost a king already. |
| Trade down | lead side gets `3 × (24 − pieces)` | The same lead is worth more with fewer pieces, so the leader should trade. |
| Endgame mop-up (≤ 10 pieces, leader only) | −3 per square of distance from each leader king to its nearest target | Brings the winning kings in to finish the game instead of shuffling. |
| Double-corner defence (≤ 10 pieces) | +10 per king of the losing side on a double-corner square | The double corner is the hardest place to trap a king. |

To change the playing style, edit the constants at the top of the
evaluation section (`ADVANCE_BONUS`, `BRIDGE`, `CENTER`, `KING_VALUE`) and
the weights inside `evaluate`.

### Game state (`Game`)

- **History:** `Game` keeps the board, the side to move, an undo stack and
  the move log.
- **Draw tracking:** it also keeps the hashes of every position since the
  last irreversible move (a capture or a man move), plus a ply counter for
  the 40-move rule. These are passed to `Engine.search` so the search's
  repetition detection includes the real game history.
- **Undo:** `undo()` pops states until it is the human's turn again.

### WebKit UI (`GuiApi` + `GUI_HTML`)

The GUI follows the same host ↔ web bridge pattern as the Common Lisp
`webkit-cl` experiments in `loving-common-lisp/src/webkit-cl`, with
[pywebview](https://pywebview.flowrl.com/) doing the work of the custom
Objective-C shim. On macOS, pywebview drives the system **WKWebView**
through PyObjC.

| webkit-cl (Common Lisp) | checkers.py (Python) |
|---|---|
| `with-app (:title … :width … :height …)` | `webview.create_window(…)` + `webview.start()` |
| `load-html "<html>…"` | `create_window(html=GUI_HTML)` |
| `register-handler "cmd" (lambda (payload) …)` | public method `GuiApi.cmd(self, …)` |
| `await window.webkit_cl.invoke("cmd", payload)` | `await pywebview.api.cmd(…)` |
| handlers return JSON strings | methods return dicts/lists; pywebview converts them to JSON |
| `eval-js` | `window.evaluate_js(…)` |

- **Python is the single source of truth.** Every bridge method (`state`,
  `new_game`, `human_move`, `ai_move`, `hint`, `undo`, `set_time`) returns
  a full snapshot of the state: board, legal move paths, whether it is your
  turn, last move, move log, result, and the AI search statistics. The
  JavaScript just redraws from that snapshot, so the UI never needs its own
  copy of the rules.
- **The browser builds multi-jumps.** It gets the list of legal paths and
  narrows it with each click (prefix matching), so it can only ever send a
  legal path to `human_move`.
- **Threading:** pywebview runs each JS→Python call on a worker thread. The
  long `ai_move` call doesn't block the Cocoa main thread, and a lock in
  `GuiApi` serializes access to the game.
- **Startup:** the page waits for pywebview's `pywebviewready` event before
  touching `pywebview.api`. The board flips when you play Black, so your
  pieces are always at the bottom.

### Ideas for further strength

- Opening book for the well-known 3-move ballots.
- Endgame databases for positions with few pieces (Chinook's main strength).
- Tune the evaluation weights automatically from self-play
  (temporal-difference learning, as in Samuel's and later programs).
- Faster move generation with bitboards (32-bit masks for each piece type).
