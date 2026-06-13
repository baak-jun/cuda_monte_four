#include <gomoku/board.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int kWinScore = 1000000;
constexpr int kInfinity = 2000000;

struct Options {
    int max_depth = 4;
    std::vector<std::string> moves;
};

// Window-based evaluation for Gomoku (572 windows on 15x15 board)
int evaluate_board(const gomoku::Board& board) {
    int score = 0;

    // Helper to score a window of 5 cells
    auto score_window = [](int black, int white) -> int {
        if (black > 0 && white > 0) return 0;
        if (black == 5) return kWinScore;
        if (white == 5) return -kWinScore;

        if (black == 4) return 5000;
        if (white == 4) return -5000;

        if (black == 3) return 400;
        if (white == 3) return -400;

        if (black == 2) return 30;
        if (white == 2) return -30;

        if (black == 1) return 1;
        if (white == 1) return -1;

        return 0;
    };

    // 1. Horizontal windows
    for (int r = 0; r < gomoku::kRows; ++r) {
        for (int c = 0; c <= gomoku::kCols - 5; ++c) {
            int black = 0, white = 0;
            for (int i = 0; i < 5; ++i) {
                std::uint8_t val = board.cells[r * gomoku::kCols + (c + i)];
                if (val == 1) ++black;
                else if (val == 2) ++white;
            }
            score += score_window(black, white);
        }
    }

    // 2. Vertical windows
    for (int c = 0; c < gomoku::kCols; ++c) {
        for (int r = 0; r <= gomoku::kRows - 5; ++r) {
            int black = 0, white = 0;
            for (int i = 0; i < 5; ++i) {
                std::uint8_t val = board.cells[(r + i) * gomoku::kCols + c];
                if (val == 1) ++black;
                else if (val == 2) ++white;
            }
            score += score_window(black, white);
        }
    }

    // 3. Diagonal down-right windows
    for (int r = 0; r <= gomoku::kRows - 5; ++r) {
        for (int c = 0; c <= gomoku::kCols - 5; ++c) {
            int black = 0, white = 0;
            for (int i = 0; i < 5; ++i) {
                std::uint8_t val = board.cells[(r + i) * gomoku::kCols + (c + i)];
                if (val == 1) ++black;
                else if (val == 2) ++white;
            }
            score += score_window(black, white);
        }
    }

    // 4. Diagonal up-right windows
    for (int r = 4; r < gomoku::kRows; ++r) {
        for (int c = 0; c <= gomoku::kCols - 5; ++c) {
            int black = 0, white = 0;
            for (int i = 0; i < 5; ++i) {
                std::uint8_t val = board.cells[(r - i) * gomoku::kCols + (c + i)];
                if (val == 1) ++black;
                else if (val == 2) ++white;
            }
            score += score_window(black, white);
        }
    }

    return score;
}

struct MoveCandidate {
    int row;
    int col;
    int heuristic_score;

    bool operator>(const MoveCandidate& other) const {
        return heuristic_score > other.heuristic_score;
    }
};

// Find candidate moves close to existing stones (within distance of 2)
std::vector<MoveCandidate> get_candidates(const gomoku::Board& board) {
    std::vector<MoveCandidate> candidates;
    candidates.reserve(40);

    // If the board is completely empty, play in the center
    if (board.moves == 0) {
        candidates.push_back({7, 7, 0});
        return candidates;
    }

    // Check cells adjacent to existing stones
    for (int r = 0; r < gomoku::kRows; ++r) {
        for (int c = 0; c < gomoku::kCols; ++c) {
            if (board.cells[r * gomoku::kCols + c] != 0) continue;

            // Look for occupied cells within distance 2
            bool near_stone = false;
            int dist_score = 0;
            for (int dr = -2; dr <= 2; ++dr) {
                for (int dc = -2; dc <= 2; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    const int nr = r + dr;
                    const int nc = c + dc;
                    if (nr >= 0 && nr < gomoku::kRows && nc >= 0 && nc < gomoku::kCols) {
                        std::uint8_t val = board.cells[nr * gomoku::kCols + nc];
                        if (val != 0) {
                            near_stone = true;
                            // Prefer moves closer to existing stones
                            dist_score += (std::abs(dr) <= 1 && std::abs(dc) <= 1) ? 2 : 1;
                        }
                    }
                }
            }

            if (near_stone) {
                candidates.push_back({r, c, dist_score});
            }
        }
    }

    // Sort candidates to examine the most promising moves first (improves pruning)
    std::sort(candidates.begin(), candidates.end(), std::greater<MoveCandidate>());
    return candidates;
}

std::uint64_t g_nodes = 0;

// Alpha-Beta Minimax search
int alphabeta(gomoku::Board& board, int depth, int alpha, int beta, bool is_max, int& best_r, int& best_c) {
    ++g_nodes;

    gomoku::Outcome outcome = gomoku::check_outcome(board);
    if (outcome == gomoku::Outcome::BlackWin) return kWinScore - board.moves;
    if (outcome == gomoku::Outcome::WhiteWin) return -kWinScore + board.moves;
    if (outcome == gomoku::Outcome::Draw) return 0;

    if (depth <= 0) {
        return evaluate_board(board);
    }

    std::vector<MoveCandidate> candidates = get_candidates(board);
    if (candidates.empty()) return 0;

    int local_best_r = -1;
    int local_best_c = -1;

    if (is_max) {
        int max_eval = -kInfinity;
        for (const auto& move : candidates) {
            gomoku::Board next = board;
            gomoku::play_inline(next, move.row, move.col);

            int dummy_r, dummy_c;
            int eval = alphabeta(next, depth - 1, alpha, beta, false, dummy_r, dummy_c);

            if (eval > max_eval) {
                max_eval = eval;
                local_best_r = move.row;
                local_best_c = move.col;
            }
            alpha = std::max(alpha, eval);
            if (beta <= alpha) {
                break; // Beta cutoff
            }
        }
        best_r = local_best_r;
        best_c = local_best_c;
        return max_eval;
    } else {
        int min_eval = kInfinity;
        for (const auto& move : candidates) {
            gomoku::Board next = board;
            gomoku::play_inline(next, move.row, move.col);

            int dummy_r, dummy_c;
            int eval = alphabeta(next, depth - 1, alpha, beta, true, dummy_r, dummy_c);

            if (eval < min_eval) {
                min_eval = eval;
                local_best_r = move.row;
                local_best_c = move.col;
            }
            beta = std::min(beta, eval);
            if (beta <= alpha) {
                break; // Alpha cutoff
            }
        }
        best_r = local_best_r;
        best_c = local_best_c;
        return min_eval;
    }
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--moves" && i + 1 < argc) {
            std::string moves_str = argv[++i];
            std::stringstream ss(moves_str);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) {
                    options.moves.push_back(item);
                }
            }
        } else if (arg == "--max-depth" && i + 1 < argc) {
            options.max_depth = std::atoi(argv[++i]);
        }
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_args(argc, argv);
        gomoku::Board board;

        // Populate board from moves
        for (const auto& m : options.moves) {
            size_t underscore = m.find('_');
            if (underscore != std::string::npos) {
                int r = std::atoi(m.substr(0, underscore).c_str());
                int c = std::atoi(m.substr(underscore + 1).c_str());
                gomoku::play_inline(board, r, c);
            }
        }

        const auto start_time = std::chrono::high_resolution_clock::now();
        int best_r = -1, best_c = -1;
        bool is_max = (board.side_to_move == gomoku::Player::Black);

        int score = alphabeta(board, options.max_depth, -kInfinity, kInfinity, is_max, best_r, best_c);
        const auto end_time = std::chrono::high_resolution_clock::now();
        const double duration = std::chrono::duration<double>(end_time - start_time).count();

        std::cout << "result " << (score > 100000 ? "black_win" : (score < -100000 ? "white_win" : "draw")) << '\n';
        std::cout << "score " << score << '\n';
        if (best_r != -1 && best_c != -1) {
            std::cout << "best_move " << best_r << "_" << best_c << '\n';
        } else {
            std::cout << "best_move -1_-1\n";
        }
        std::cout << "nodes " << g_nodes << '\n';
        std::cout << "duration " << duration << '\n';

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
