# Chess Engine & AI Bot

An educational, high-performance, pure-Python chess engine and AI bot designed for the *Python AI Book*. 

This project implements a complete chess board representation, legal move generator, evaluation system, and interactive command-line interface (CLI) to play against an AI player or watch the bot play against itself. It is optimized to run efficiently in standard Python environments and is structured for educational clarity.

---

## 🛠 Features

### 1. Core Chess Engine (`chess_engine.py`)
*   **Board Representation**: Fast 64-element list representation of the board using custom bitwise piece values for colors and types.
*   **Precomputed Lookups**: King, Knight, and sliding piece (Rook, Bishop, Queen) ray-casts are precomputed at startup and frozen as immutable nested tuples for speed.
*   **Fully Legal Move Generator**: Supports all complex chess rules:
    *   Castling (verifying king path is not attacked, castling rights tracking).
    *   En Passant (capture and square tracking).
    *   Pawn Double Pushes.
    *   Underpromotions and Queen Promotions (Queen, Rook, Bishop, Knight).
*   **Check, Mate, and Stalemate Detection**: Computes pin-aware legal moves by validating that the moving king is not under check post-move.
*   **Zobrist Hashing**: Generates unique 64-bit keys for board states to track duplicate positions. Volatile metadata (side-to-move, castling rights, en-passant square) and pieces are XORed incrementally in `make_move`.
*   **Efficient Undo-Redo**: Incremental move updates using a custom lightweight history state (`BoardState`) to record board changes instead of cloning the entire board. `BoardState` captures the board reference to automatically save and restore castling rights, en-passant squares, and the Zobrist hash.
*   **FEN Parser & Generator**: Full support for Forsyth-Edwards Notation (FEN) to save and load custom chess positions.

### 2. Chess AI Bot (`chess_bot.py`)
*   **Negamax Search**: A symmetric variant of Minimax search for zero-sum games, with **Alpha-Beta Pruning** to cut off unhelpful branches early.
*   **Transposition Tables (TT)**: Caches evaluated search nodes in a hash map keyed by Zobrist hash to avoid redundant subtree searches. Saves evaluation scores, depth, node flags (exact, alpha, beta bounds), and the best move.
*   **Iterative Deepening**: Progressively searches from depth 1 up to the target depth. The transposition table is populated with the best moves from shallower iterations, resulting in superior move ordering and a higher rate of alpha-beta cutoffs at deeper levels.
*   **Quiescence Search**: Solves the *horizon effect* by extending search depth on capture/promotion sequences until a stable position ("quiet" state) is reached.
*   **Smart Move Ordering**: Orders moves before searching to maximize Alpha-Beta cutoffs using:
    *   **Transposition Table Move**: The best move found at a shallower depth is searched first.
    *   **MVV-LVA (Most Valuable Victim - Least Valuable Attacker)** heuristic for captures.
    *   Promotions and Castling prioritization.
    *   Positional progression delta based on Piece-Square Tables.
*   **Piece-Square Tables (PST)**: Custom tables representing position values for Pawns, Knights, Bishops, Rooks, and Queens from a mirrored perspective for Black.
*   **Dynamic King Safety**: Differentiates between middlegame king safety (pushing the king to the corners) and endgame centralization (drawing the king to the center of the board).
*   **Endgame Detection**: Dynamically transitions to endgame evaluation tables when total non-pawn/non-king material drops to 3000 points or lower.

### 3. Interactive CLI Interface (`main.py`)
*   **Three Game Modes**:
    1.  **Play as White**: Human player makes the first move.
    2.  **Play as Black**: Bot plays White, and the human controls Black.
    3.  **Bot vs. Bot**: Watch the AI play against itself with custom timeouts.
*   **Visual Board Display**: Uses ANSI color coding for terminal-based board visualization, including an information sidebar showing active turn, move counts, 50-move rule status, check status, en-passant coordinate, castling rights, and current evaluation score.
*   **Configurable Depth**: Supports adjustable bot search depths (typically depth 3 or 4 for fast play, max depth 5–6).
*   **Robust Command System**: Includes utility commands to query legal moves, print current FEN, set custom FEN states, reset the board, and show help menus.

### 4. Move Generator & Zobrist Verification (`test_engine.py`)
*   **Perft Testing**: Runs automated performance tests by recursively traversing the move tree to verify engine accuracy. Evaluates starting positions up to depth 3 (8,902 nodes) against international standards to ensure 100% rule-compliance.
*   **Hash Correctness Assertions**: Automatically validates that incremental Zobrist hashes match fresh re-evaluations after every move and restore perfectly on undo.

---

## 📂 Project Structure

```bash
Chess_game/
├── chess_engine.py      # Board state representation, BoardState history, & Zobrist keys
├── chess_bot.py         # Iterative Deepening Negamax AI, Transposition Table, & evaluation
├── main.py              # CLI interface & interactive game loop
├── tests/
│   ├── conftest.py      # sys.path shim so tests can import the flat modules
│   └── test_engine.py   # pytest-based Perft & Zobrist verification suite
├── pyproject.toml       # Project dependencies & dev tooling
├── pyrefly.toml         # Strict type-checker config
├── justfile             # Task runner (fmt / lint / typecheck / test)
├── Makefile             # clean / test / run
├── .claude/             # Claude Code hooks that gate every edit and turn end
└── uv.lock              # Locked dependency tree
```

---

## 🚀 Installation & Running

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management and [`just`](https://just.systems/) as its task runner.

### Prerequisites
```bash
# uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# just (Rust task runner — do NOT install the "just" package from PyPI, it will shadow this one)
brew install just     # macOS
# or: cargo install just
```

### 1. Install dev dependencies

```bash
uv sync
```

This creates `.venv/` and installs `ruff`, `pyrefly`, `pytest`, `hypothesis`, and friends.

### 2. Playing the Game

```bash
uv run python main.py
# or
make run
```

### 3. Running Move Generation & Zobrist Tests (Perft)

```bash
just test          # fast run via pytest-testmon
just test-all      # full parallel run
# or
uv run pytest -v
```

### 4. Full quality gate

```bash
just check   # fmt-check + lint + typecheck + test
```

The same gate runs automatically when Claude Code ends a turn. Per-edit, the `.claude/hooks/py-check.sh` hook formats + typechecks the file you just touched.

---

## 📖 Deep Dive: Engine Architecture

### Bitwise Piece Representations
Pieces are stored as integers combining their type and color using bitwise operations:
*   **Types**: `PAWN = 1`, `KNIGHT = 2`, `BISHOP = 3`, `ROOK = 4`, `QUEEN = 5`, `KING = 6` (occupying bits 0–2).
*   **Colors**: `WHITE = 8`, `BLACK = 16` (occupying bits 3–4).

Examples:
*   `WHITE | PAWN` $\rightarrow$ `8 | 1` $\rightarrow$ `9`
*   `BLACK | QUEEN` $\rightarrow$ `16 | 5` $\rightarrow$ `21`

We extract type using `piece & TYPE_MASK` (where `TYPE_MASK = 7`) and color using `piece & COLOR_MASK` (where `COLOR_MASK = 24`).

### Zobrist Hashing Details
Zobrist hashing maps a board configuration to a unique 64-bit integer. At start, we generate random 64-bit bitstrings for:
1.  Each piece type and color at each of the 64 squares (`ZOBRIST_PIECES`).
2.  The active side to move (`ZOBRIST_SIDE`), XORed in if it is Black's turn.
3.  Castling rights combinations 0–15 (`ZOBRIST_CASTLING`).
4.  The en-passant target square file 0–63 (`ZOBRIST_EP`).

During `make_move`, we update the hash incrementally by XORing out the moving piece from its original square, XORing in the piece at its new square, and XORing any captured piece.

### Incremental Move Undo
Rather than performing deep-copies of the board which are extremely slow during AI search, `chess_engine.py` uses `make_move` and `unmake_move` methods. To undo a move, it stores a lightweight `BoardState` history object containing volatile board metadata:
*   En-passant target square
*   Castling rights
*   50-move rule halfmove clock
*   Zobrist hash key

By passing `board` directly to `BoardState(board)`, the object captures all volatile states in one constructor call. During `unmake_move`, these fields are restored in $O(1)$ time.

---

## 🤖 AI Search & Evaluation Details

### Iterative Deepening
Iterative deepening runs the Negamax search starting from depth 1, then depth 2, up to the target depth. The best move found at each depth is saved in the Transposition Table. At the start of the next depth search, this best move is ordered first. This dramatically increases the number of alpha-beta cutoffs, meaning searching depths 1, 2, and 3 sequentially is often faster than searching depth 3 directly.

### Transposition Table (TT) Layout
The transposition table is a dictionary mapping the 64-bit Zobrist hash of a position to:
*   `depth`: The search depth at which the position was evaluated.
*   `score`: The evaluation score. If it is a checkmate score, it is adjusted to be independent of its distance to the root:
    *   Storing: `score + distance_from_root` (for checkmate values)
    *   Retrieving: `score - distance_from_root`
*   `flag`: Node boundary indicators:
    *   `TT_EXACT` (0): The evaluation is exact.
    *   `TT_ALPHA` (1): The evaluation is an upper bound (fail-low, value $\le \alpha$).
    *   `TT_BETA` (2): The evaluation is a lower bound (fail-high, value $\ge \beta$).
*   `best_move`: The best move evaluated at this state.

### Heuristics & Piece-Square Tables (PST)
The evaluation function assigns values to pieces:
*   **Pawn**: 100 | **Knight**: 320 | **Bishop**: 330 | **Rook**: 500 | **Queen**: 900 | **King**: 20000

These base values are adjusted according to where pieces stand on the board:
*   **Pawns**: Encouraged to advance in the center and get extra points on the 7th rank (near promotion).
*   **Knights**: Penalized heavily on the edges of the board where they control fewer squares (centralization).
*   **Bishops**: Encouraged to occupy active diagonals.
*   **Rooks**: Encouraged to seize the 7th rank and occupy central files.
*   **Kings**: Guided to castle safety in the middlegame, but pushed to the center to assist pawns in endgames.

---

## 📊 Perft Benchmarks
The Perft (Performance Test) suite validates that the move generator computes the exact number of legal moves at each depth. Below are the verified results for the starting position:

| Depth | Expected Positions (Nodes) | Meaning | Status |
| :---: | :------------------------: | :------ | :----: |
| **1** | 20                         | White's first moves | **PASS** |
| **2** | 400                        | Black's reply moves | **PASS** |
| **3** | 8,902                      | White's second moves | **PASS** |

---

## 🗺 Engine Roadmap / Future Features
To further increase the performance of this chess engine, the following upgrades can be added:
1.  **Null Move Pruning**: Assume a passing move (do nothing) doesn't worsen the position; if the evaluation is still $\ge \beta$, cut off search to prune large subtrees.
2.  **Killer Moves / History Heuristic**: Store moves that cause beta cutoffs to order them first in sibling nodes.
3.  **UCI Protocol Support**: Adapt the CLI to communicate with chess GUIs (like Arena, Lichess, or ChessBase) via the Universal Chess Interface protocol.
