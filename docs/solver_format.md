# Solver State Files

`solver_cpu` stores memoized minimax states keyed by:

```text
black uint64
white uint64
side_to_move uint8
```

The bit index is `row + col * 7`; row 0 is the bottom cell, and row 6 is an unused sentinel bit for each column.

The result is always from the perspective of the side to move:

- `win`: the side to move can force a win.
- `loss`: the side to move cannot avoid losing.
- `draw`: the side to move can avoid losing, but cannot force a win.
- `unknown`: search stopped by `--max-depth` or `--limit-states`.

## Binary Format

Use `--save states.c4s` and `--load states.c4s`.

Header, 24 bytes:

| Offset | Type | Name |
| --- | --- | --- |
| 0 | `uint32_t` | magic, `0x31465343` (`CSF1`) |
| 4 | `uint16_t` | version, currently `1` |
| 6 | `uint16_t` | record size, currently `24` |
| 8 | `uint64_t` | record count |
| 16 | `uint8_t` | rows, `6` |
| 17 | `uint8_t` | cols, `7` |
| 18 | `uint8_t[6]` | reserved |

Record, 24 bytes:

| Offset | Type | Name |
| --- | --- | --- |
| 0 | `uint64_t` | black bitboard |
| 8 | `uint64_t` | white bitboard |
| 16 | `uint8_t` | side to move, `0` black or `1` white |
| 17 | `uint8_t` | result, `0` unknown, `1` win, `2` loss, `3` draw |
| 18 | `int8_t` | best move column, `-1` if none |
| 19 | `uint8_t` | depth |
| 20 | `uint8_t[4]` | reserved |

## Text Format

Use `--text states.csv` for inspection. Rows are:

```text
black_hex,white_hex,side_to_move,result,best_move,depth,board
```

`board` is six slash-separated rows from top to bottom. `B` is black, `W` is white, and `.` is empty.
