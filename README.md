# cuda_monte_four

CUDA/C++ experiments for 6x7 Connect Four using two 64-bit bitboards:

- `black`: black stones
- `white`: white stones

The bit index is `row + col * 7`; row `0` is the bottom of a column and col is `0..6`. Row `6` in each column is an unused sentinel bit that keeps shift-based win checks from crossing column boundaries.

## Targets

- `monte_carlo_cuda`: runs random rollouts in parallel on CUDA.
- `play_monte_carlo_cuda`: interactive human-vs-computer game using CUDA Monte Carlo win-rate estimates per column.
- `perfect_frontier_cuda`: hybrid limited-depth exact search; CPU builds a frontier and CUDA evaluates leaf subtrees.
- `exhaustive`: DFS traversal of legal game states, with optional binary dumps.
- `solver_cpu`: memoized minimax solver keyed by `(black, white, side_to_move)`.
- `perfect_solver_cpu`: negamax/alpha-beta solver for forced win/loss/draw and best move.
- `perfect_db_cpu`: persistent exact minimax DB builder/query tool for preprocessing.
- `gomoku_solver_cpu`: limited-depth Gomoku alpha-beta search.
- `gomoku_monte_carlo_cuda`: CUDA Gomoku rollout evaluator.

Gomoku uses the exact-five rule throughout the CPU, CUDA, and web implementations: exactly five contiguous stones win, while an overline of six or more does not.

## Build

```powershell
cmake -S . -B build
cmake --build build --config Release
```

CUDA is optional in CMake. If no CUDA compiler is available, only `exhaustive` is generated.

This local environment did not have `cmake`, `nvcc`, or MSVC `cl` on PATH, so only a direct `g++` compile of `exhaustive` was verified:

```powershell
g++ -std=c++17 -Wall -Wextra -pedantic -I include src/exhaustive.cpp -o build_exhaustive.exe
.\build_exhaustive.exe --max-depth 4 --dump sample_states.bin
```

## Monte Carlo

```powershell
.\build\Release\monte_carlo_cuda.exe --simulations 1000000 --threads 256 --seed 12345
.\build\Release\monte_carlo_cuda.exe --moves 334455 --simulations 2000000
```

`--moves` is a string of columns `0..6` from the empty board. Each rollout picks random legal columns until a terminal result.

## Play Against CUDA

```powershell
.\build\Release\play_monte_carlo_cuda.exe --simulations-per-move 100000 --threads 256
.\build\Release\play_monte_carlo_cuda.exe --computer-first --simulations-per-move 250000
```

You are `X`; the computer is `O`. On each computer turn, it evaluates every legal column with the requested number of random rollouts, prints the estimated computer win rate for each column, and plays the best-scoring column. The score used for choosing is `computer_win_rate + 0.5 * draw_rate`.

For Colab without CMake:

```bash
nvcc -std=c++17 -I include src/play_monte_carlo_cuda.cu -o play_monte_carlo_cuda
./play_monte_carlo_cuda --simulations-per-move 100000 --threads 256
```

Colab notebook cells are not a comfortable interactive terminal. Use `--moves` to play one computer turn at a time. If you are first and choose column `3`:

```bash
./play_monte_carlo_cuda --moves 3 --simulations-per-move 100000
```

The program prints `next_moves`. Re-run with that string plus your next move. For example, if it prints `next_moves 33` and you choose column `2`:

```bash
./play_monte_carlo_cuda --moves 332 --simulations-per-move 100000
```

Use `--computer-first --once` to ask the computer for the first move:

```bash
./play_monte_carlo_cuda --computer-first --once --simulations-per-move 100000
```

## Exhaustive Search

```powershell
.\build\Release\exhaustive.exe --max-depth 10
.\build\Release\exhaustive.exe --limit-states 1000000 --dump states.bin
```

Full-depth exhaustive traversal is intentionally not optimized yet and can become very large. Use `--max-depth` and `--limit-states` while testing.

Dump layout is documented in [docs/exhaustive_format.md](docs/exhaustive_format.md).

## Memoized Solver

```powershell
.\build\Release\solver_cpu.exe --max-depth 12 --save states.c4s --text states.csv
.\build\Release\solver_cpu.exe --load states.c4s --moves 334455
```

The solver result is from the side-to-move perspective. `win` means the side to move has a forced win; `best_move` is the selected winning or drawing column. The binary and CSV formats are documented in [docs/solver_format.md](docs/solver_format.md).

## Perfect Solver

```powershell
.\build\Release\perfect_solver_cpu.exe --max-depth 14
.\build\Release\perfect_solver_cpu.exe --moves 334455 --max-depth 42 --text perfect_states.csv
```

`perfect_solver_cpu` is the version intended for perfect play research. It uses negamax with alpha-beta pruning and a transposition table. Its result is from the side-to-move perspective:

- `win`: the side to move can force a win.
- `loss`: the side to move loses against perfect play.
- `draw`: both sides can avoid losing.

Full `--max-depth 42` from the empty board is the real proof search and may still take a long time without more Connect Four-specific pruning. Use smaller depths first, then run deeper positions or add larger `--reserve` values when memory allows.

## Persistent Perfect DB

Use this when the goal is preprocessing once and querying best moves later.

Build a partial DB:

```powershell
.\build\Release\perfect_db_cpu.exe --max-depth 14 --save perfect.c4db
```

Query the DB after a move sequence:

```powershell
.\build\Release\perfect_db_cpu.exe --load perfect.c4db --moves 334455 --save perfect.c4db
```

For a full proof from the empty board:

```powershell
.\build\Release\perfect_db_cpu.exe --max-depth 42 --reserve 50000000 --save perfect.c4db
```

Unlike `perfect_solver_cpu`, this DB builder does not use alpha-beta cutoffs for solved states, so it is meant to populate a reusable state table. That makes lookup more suitable for repeated play, but the full empty-board DB can be very large.

## Hybrid CUDA Exact Search

`perfect_frontier_cuda` is an experimental middle ground. It does not build a persistent DB. Instead:

- CPU expands the current position for a few plies.
- CUDA evaluates those frontier leaves with limited-depth negamax.
- CPU backs up the scores and prints the best move.

Build in Colab:

```bash
nvcc -std=c++17 -I include src/perfect_frontier_cuda.cu -o perfect_frontier_cuda
```

Run:

```bash
./perfect_frontier_cuda --moves 334455 --max-depth 16 --frontier-depth 5 --threads 128
./perfect_frontier_cuda --max-depth 18 --frontier-depth 6 --threads 128 --stack-bytes 16384
```

Use this when you want to see whether CUDA helps for deeper per-position search. Use `perfect_db_cpu` when you want reusable preprocessing files.
