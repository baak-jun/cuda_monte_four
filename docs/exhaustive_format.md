# Exhaustive Dump Format

`src/exhaustive.cpp` optionally writes a binary stream with:

- one 16-byte header
- zero or more 24-byte records, one per visited DFS state

All integer fields are written in the host's native little-endian layout expected by the current Windows/x86_64 CUDA/C++ test environment.

## Header

| Offset | Type | Name | Description |
| --- | --- | --- | --- |
| 0 | `uint32_t` | `magic` | `0x34464643`, bytes `43 46 46 34` (`CFF4`) |
| 4 | `uint16_t` | `version` | currently `1` |
| 6 | `uint16_t` | `record_size` | currently `24` |
| 8 | `uint8_t` | `rows` | `6` |
| 9 | `uint8_t` | `cols` | `7` |
| 10 | `uint8_t[6]` | `reserved` | zero |

## Record

| Offset | Type | Name | Description |
| --- | --- | --- | --- |
| 0 | `uint64_t` | `black` | black stone bitboard |
| 8 | `uint64_t` | `white` | white stone bitboard |
| 16 | `uint8_t` | `side_to_move` | `0` for black, `1` for white |
| 17 | `uint8_t` | `outcome` | `0` ongoing, `1` black win, `2` white win, `3` draw |
| 18 | `uint8_t[6]` | `reserved` | zero |

The bit index convention is owned by `include/connect4/board.hpp`: `bit = 1ULL << (row + col * 7)`, where row 0 is the bottom cell of a column. Row 6 is an unused sentinel bit in every column. The exhaustive executable preserves `connect4::Board::black` and `connect4::Board::white` directly.
