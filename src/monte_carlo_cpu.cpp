#include <iostream>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "connect4/board.hpp"

namespace {

int parse_int_arg(int argc, char** argv, const std::string& flag, int default_val) {
    for (int i = 1; i < argc - 1; ++i) {
        if (argv[i] == flag) {
            return std::stoi(argv[i + 1]);
        }
    }
    return default_val;
}

std::uint32_t xorshift32(std::uint32_t& state) {
    std::uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return state = x;
}

int heuristic_legal_move(const connect4::Board& board, std::uint32_t& rng) {
    int count = 0;
    std::array<int, connect4::kCols> legal = connect4::legal_moves(board, count);
    if (count == 0) {
        return -1;
    }

    // 1. Immediate win check for active player
    std::uint64_t my_stones = (board.side_to_move == connect4::Player::Black) ? board.black : board.white;
    for (int i = 0; i < count; ++i) {
        int col = legal[i];
        int row = connect4::column_height(board, col);
        std::uint64_t bit = connect4::bit_at(row, col);
        if (connect4::has_four(my_stones | bit)) {
            return col;
        }
    }

    // 2. Block opponent's immediate win
    std::uint64_t opp_stones = (board.side_to_move == connect4::Player::Black) ? board.white : board.black;
    for (int i = 0; i < count; ++i) {
        int col = legal[i];
        int row = connect4::column_height(board, col);
        std::uint64_t bit = connect4::bit_at(row, col);
        if (connect4::has_four(opp_stones | bit)) {
            return col;
        }
    }

    // 3. Avoid giving opponent an immediate win
    std::array<int, connect4::kCols> safe_legal;
    int safe_count = 0;
    for (int i = 0; i < count; ++i) {
        int col = legal[i];
        int row = connect4::column_height(board, col);
        if (row + 1 < connect4::kRows) {
            std::uint64_t opp_bit_above = connect4::bit_at(row + 1, col);
            if (!connect4::has_four(opp_stones | opp_bit_above)) {
                safe_legal[safe_count++] = col;
            }
        } else {
            safe_legal[safe_count++] = col;
        }
    }

    if (safe_count > 0) {
        return safe_legal[xorshift32(rng) % safe_count];
    }
    return legal[xorshift32(rng) % count];
}

connect4::Outcome rollout(connect4::Board board, std::uint32_t& rng) {
    while (board.moves < connect4::kMaxMoves) {
        if (connect4::has_four(board.black)) {
            return connect4::Outcome::BlackWin;
        }
        if (connect4::has_four(board.white)) {
            return connect4::Outcome::WhiteWin;
        }

        const int col = heuristic_legal_move(board, rng);
        if (col < 0) {
            break;
        }
        auto res = connect4::play(board, col);
        if (!res.legal) break;
        board = res.board;
    }


    if (connect4::has_four(board.black)) {
        return connect4::Outcome::BlackWin;
    }
    if (connect4::has_four(board.white)) {
        return connect4::Outcome::WhiteWin;
    }
    return connect4::Outcome::Draw;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const int simulations = parse_int_arg(argc, argv, "--simulations", 1'000'000);
        const int seed = parse_int_arg(argc, argv, "--seed", 12345);
        
        connect4::Board start_board; // Empty board by default
        
        std::uint64_t black_wins = 0;
        std::uint64_t white_wins = 0;
        std::uint64_t draws = 0;

        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::uint32_t rng_state = seed;
        for (int idx = 0; idx < simulations; ++idx) {
            std::uint32_t rng = rng_state ^ (0x9E3779B9U * (idx + 1));
            connect4::Outcome outcome = rollout(start_board, rng);
            if (outcome == connect4::Outcome::BlackWin) {
                black_wins++;
            } else if (outcome == connect4::Outcome::WhiteWin) {
                white_wins++;
            } else {
                draws++;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        
        std::cout << "simulations=" << simulations << "\n";
        std::cout << "black_wins=" << black_wins << "\n";
        std::cout << "white_wins=" << white_wins << "\n";
        std::cout << "draws=" << draws << "\n";
        std::cout << "duration_sec=" << duration.count() << "\n";
        
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
