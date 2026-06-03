#pragma once

#include <array>
#include <cstdint>

namespace connect4 {

constexpr int kRows = 6;
constexpr int kCols = 7;
constexpr int kCells = kRows * kCols;
constexpr int kMaxMoves = kCells;
constexpr int kStride = kRows + 1;

enum class Player : std::uint8_t {
    Black = 0,
    White = 1,
};

enum class Outcome : std::uint8_t {
    Unknown = 0,
    BlackWin = 1,
    WhiteWin = 2,
    Draw = 3,
};

struct Board {
    std::uint64_t black = 0;
    std::uint64_t white = 0;
    std::uint8_t moves = 0;
    Player side_to_move = Player::Black;
};

struct MoveResult {
    Board board;
    bool legal = false;
};

constexpr std::uint64_t bit_at(int row, int col) {
    return 1ULL << (row + col * kStride);
}

constexpr std::uint64_t column_mask(int col) {
    return ((1ULL << kRows) - 1ULL) << (col * kStride);
}

inline std::uint64_t occupied(const Board& board) {
    return board.black | board.white;
}

inline Player other(Player player) {
    return player == Player::Black ? Player::White : Player::Black;
}

inline std::uint64_t stones_for(const Board& board, Player player) {
    return player == Player::Black ? board.black : board.white;
}

inline int column_height(const Board& board, int col) {
    const std::uint64_t occ = occupied(board);
    for (int row = 0; row < kRows; ++row) {
        if ((occ & bit_at(row, col)) == 0) {
            return row;
        }
    }
    return kRows;
}

inline bool can_play(const Board& board, int col) {
    return col >= 0 && col < kCols && column_height(board, col) < kRows;
}

inline MoveResult play(const Board& board, int col) {
    const int row = column_height(board, col);
    if (col < 0 || col >= kCols || row >= kRows) {
        return {board, false};
    }

    Board next = board;
    const std::uint64_t bit = bit_at(row, col);
    if (board.side_to_move == Player::Black) {
        next.black |= bit;
    } else {
        next.white |= bit;
    }
    next.side_to_move = other(board.side_to_move);
    ++next.moves;
    return {next, true};
}

inline bool has_four(std::uint64_t stones) {
    std::uint64_t connected = stones & (stones >> 1);
    if ((connected & (connected >> 2)) != 0) {
        return true;
    }

    connected = stones & (stones >> kStride);
    if ((connected & (connected >> (2 * kStride))) != 0) {
        return true;
    }

    connected = stones & (stones >> (kStride - 1));
    if ((connected & (connected >> (2 * (kStride - 1)))) != 0) {
        return true;
    }

    connected = stones & (stones >> (kStride + 1));
    if ((connected & (connected >> (2 * (kStride + 1)))) != 0) {
        return true;
    }

    return false;
}

inline Outcome terminal_outcome(const Board& board) {
    if (has_four(board.black)) {
        return Outcome::BlackWin;
    }
    if (has_four(board.white)) {
        return Outcome::WhiteWin;
    }
    if (board.moves == kMaxMoves) {
        return Outcome::Draw;
    }
    return Outcome::Unknown;
}

inline std::array<int, kCols> legal_moves(const Board& board, int& count) {
    std::array<int, kCols> moves{};
    count = 0;
    for (int col = 0; col < kCols; ++col) {
        if (can_play(board, col)) {
            moves[count++] = col;
        }
    }
    return moves;
}

inline const char* outcome_name(Outcome outcome) {
    switch (outcome) {
    case Outcome::BlackWin:
        return "black";
    case Outcome::WhiteWin:
        return "white";
    case Outcome::Draw:
        return "draw";
    default:
        return "unknown";
    }
}

} // namespace connect4
