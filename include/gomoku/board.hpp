#pragma once

#include <array>
#include <cstdint>

namespace gomoku {

constexpr int kRows = 15;
constexpr int kCols = 15;
constexpr int kCells = kRows * kCols;
constexpr int kMaxMoves = kCells;

enum class Player : std::uint8_t {
    None = 0,
    Black = 1, // X
    White = 2, // O
};

enum class Outcome : std::uint8_t {
    Unknown = 0,
    BlackWin = 1,
    WhiteWin = 2,
    Draw = 3,
};

struct Board {
    std::uint8_t cells[kCells] = {0}; // 0=None, 1=Black, 2=White
    std::uint16_t moves = 0;
    Player side_to_move = Player::Black;
    std::int16_t last_move = -1;
};

struct MoveResult {
    Board board;
    bool legal = false;
};

inline Player other(Player player) {
    return player == Player::Black ? Player::White : Player::Black;
}

inline bool can_play(const Board& board, int r, int c) {
    if (r < 0 || r >= kRows || c < 0 || c >= kCols) return false;
    return board.cells[r * kCols + c] == 0;
}

inline bool play_inline(Board& board, int r, int c) {
    if (!can_play(board, r, c)) return false;
    const int idx = r * kCols + c;
    board.cells[idx] = static_cast<std::uint8_t>(board.side_to_move);
    board.last_move = idx;
    board.side_to_move = other(board.side_to_move);
    ++board.moves;
    return true;
}

// Optimized 5-in-a-row check around the last move (O(1) complexity)
inline bool has_five(const Board& board, int last_idx) {
    if (last_idx < 0 || last_idx >= kCells) return false;
    const std::uint8_t color = board.cells[last_idx];
    if (color == 0) return false;

    const int r = last_idx / kCols;
    const int c = last_idx % kCols;

    // 4 Directions: horizontal, vertical, diagonal down-right, diagonal up-right
    constexpr int dr[4] = {0, 1, 1, -1};
    constexpr int dc[4] = {1, 0, 1, 1};

    for (int d = 0; d < 4; ++d) {
        int count = 1;

        // Positive direction
        for (int step = 1; step < 5; ++step) {
            const int nr = r + dr[d] * step;
            const int nc = c + dc[d] * step;
            if (nr < 0 || nr >= kRows || nc < 0 || nc >= kCols) break;
            if (board.cells[nr * kCols + nc] == color) {
                ++count;
            } else {
                break;
            }
        }

        // Negative direction
        for (int step = 1; step < 5; ++step) {
            const int nr = r - dr[d] * step;
            const int nc = c - dc[d] * step;
            if (nr < 0 || nr >= kRows || nc < 0 || nc >= kCols) break;
            if (board.cells[nr * kCols + nc] == color) {
                ++count;
            } else {
                break;
            }
        }

        if (count == 5) {
            return true;
        }
    }

    return false;
}

inline Outcome check_outcome(const Board& board) {
    if (board.last_move != -1 && has_five(board, board.last_move)) {
        // The player who JUST moved won the game.
        // board.side_to_move has already been swapped to the other player.
        // So the winner is other(board.side_to_move)
        const Player winner = other(board.side_to_move);
        return winner == Player::Black ? Outcome::BlackWin : Outcome::WhiteWin;
    }
    if (board.moves >= kMaxMoves) {
        return Outcome::Draw;
    }
    return Outcome::Unknown;
}

} // namespace gomoku
