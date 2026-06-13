#include <gomoku/board.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

namespace {

struct XORShift {
    std::uint32_t state;
    __device__ std::uint32_t next() {
        std::uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }
};

__device__ bool device_has_five(const std::uint8_t* cells, int last_idx) {
    const int r = last_idx / 15;
    const int c = last_idx % 15;

    constexpr int dr[4] = {0, 1, 1, -1};
    constexpr int dc[4] = {1, 0, 1, 1};

    const std::uint8_t color = cells[last_idx];

    for (int d = 0; d < 4; ++d) {
        int count = 1;

        // Positive direction
        for (int step = 1; step < 5; ++step) {
            const int nr = r + dr[d] * step;
            const int nc = c + dc[d] * step;
            if (nr < 0 || nr >= 15 || nc < 0 || nc >= 15) break;
            if (cells[nr * 15 + nc] == color) {
                ++count;
            } else {
                break;
            }
        }

        // Negative direction
        for (int step = 1; step < 5; ++step) {
            const int nr = r - dr[d] * step;
            const int nc = c - dc[d] * step;
            if (nr < 0 || nr >= 15 || nc < 0 || nc >= 15) break;
            if (cells[nr * 15 + nc] == color) {
                ++count;
            } else {
                break;
            }
        }

        if (count >= 5) {
            return true;
        }
    }

    return false;
}

__global__ void playout_kernel(
    const std::uint8_t* init_cells,
    int init_moves,
    std::uint8_t init_side,
    int candidate_r,
    int candidate_c,
    int simulations,
    unsigned int seed,
    int* win_counts // [draws, black_wins, white_wins]
) {
    __shared__ int s_draws[256];
    __shared__ int s_black[256];
    __shared__ int s_white[256];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    s_draws[local_tid] = 0;
    s_black[local_tid] = 0;
    s_white[local_tid] = 0;

    int outcome = -1; // -1 = running, 0 = draw, 1 = black, 2 = white

    if (tid < simulations) {
        XORShift rng;
        rng.state = seed + tid * 1099087573ULL;
        if (rng.state == 0) rng.state = 1;

        std::uint8_t cells[225];
        for (int i = 0; i < 225; ++i) cells[i] = init_cells[i];

        int last_move = candidate_r * 15 + candidate_c;
        cells[last_move] = init_side;
        std::uint8_t side = (init_side == 1) ? 2 : 1;
        int moves = init_moves + 1;

        if (device_has_five(cells, last_move)) {
            outcome = init_side;
        } else if (moves >= 225) {
            outcome = 0;
        } else {
            // Collect empty cells
            int empty_indices[225];
            int empty_count = 0;
            for (int i = 0; i < 225; ++i) {
                if (cells[i] == 0) {
                    empty_indices[empty_count++] = i;
                }
            }

            while (moves < 225) {
                if (empty_count == 0) {
                    outcome = 0;
                    break;
                }

                int r_idx = rng.next() % empty_count;
                int play_idx = empty_indices[r_idx];

                // Remove from list
                empty_indices[r_idx] = empty_indices[empty_count - 1];
                --empty_count;

                cells[play_idx] = side;
                last_move = play_idx;

                if (device_has_five(cells, last_move)) {
                    outcome = side;
                    break;
                }

                side = (side == 1) ? 2 : 1;
                ++moves;
            }
            if (outcome == -1) outcome = 0;
        }

        if (outcome == 0) s_draws[local_tid] = 1;
        else if (outcome == 1) s_black[local_tid] = 1;
        else if (outcome == 2) s_white[local_tid] = 1;
    }

    __syncthreads();

    // Block level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (local_tid < stride) {
            s_draws[local_tid] += s_draws[local_tid + stride];
            s_black[local_tid] += s_black[local_tid + stride];
            s_white[local_tid] += s_white[local_tid + stride];
        }
        __syncthreads();
    }

    if (local_tid == 0) {
        atomicAdd(&win_counts[0], s_draws[0]);
        atomicAdd(&win_counts[1], s_black[0]);
        atomicAdd(&win_counts[2], s_white[0]);
    }
}

struct MoveCandidate {
    int row;
    int col;
    int heuristic_score;

    bool operator>(const MoveCandidate& other) const {
        return heuristic_score > other.heuristic_score;
    }
};

std::vector<MoveCandidate> get_candidates(const gomoku::Board& board) {
    std::vector<MoveCandidate> candidates;
    candidates.reserve(40);

    if (board.moves == 0) {
        candidates.push_back({7, 7, 0});
        return candidates;
    }

    for (int r = 0; r < gomoku::kRows; ++r) {
        for (int c = 0; c < gomoku::kCols; ++c) {
            if (board.cells[r * gomoku::kCols + c] != 0) continue;

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

    std::sort(candidates.begin(), candidates.end(), std::greater<MoveCandidate>());
    return candidates;
}

struct Options {
    int simulations = 10000;
    int threads = 256;
    unsigned int seed = 12345;
    std::vector<std::string> moves;
};

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
        } else if (arg == "--simulations-per-move" && i + 1 < argc) {
            options.simulations = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            options.threads = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            options.seed = std::strtoul(argv[++i], nullptr, 10);
        }
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_args(argc, argv);
        gomoku::Board board;

        for (const auto& m : options.moves) {
            size_t underscore = m.find('_');
            if (underscore != std::string::npos) {
                int r = std::atoi(m.substr(0, underscore).c_str());
                int c = std::atoi(m.substr(underscore + 1).c_str());
                gomoku::play_inline(board, r, c);
            }
        }

        std::vector<MoveCandidate> candidates = get_candidates(board);
        if (candidates.empty()) {
            std::cout << "result draw\nscore 0\nbest_move -1_-1\n";
            return 0;
        }

        // Handle first move shortcut
        if (board.moves == 0) {
            std::cout << "result unknown\nscore 0\nbest_move 7_7\n";
            return 0;
        }

        // CUDA Allocations
        std::uint8_t* d_cells = nullptr;
        cudaMalloc(&d_cells, 225);
        cudaMemcpy(d_cells, board.cells, 225, cudaMemcpyHostToDevice);

        int* d_win_counts = nullptr;
        cudaMalloc(&d_win_counts, 3 * sizeof(int));

        const auto start_time = std::chrono::high_resolution_clock::now();

        int best_r = -1, best_c = -1;
        double best_win_rate = -1.0;
        std::vector<double> win_rates(candidates.size(), 0.0);

        std::uint8_t current_side = static_cast<std::uint8_t>(board.side_to_move);

        for (size_t idx = 0; idx < candidates.size(); ++idx) {
            const auto& move = candidates[idx];

            // Reset win counts on device
            int h_win_counts[3] = {0, 0, 0};
            cudaMemcpy(d_win_counts, h_win_counts, 3 * sizeof(int), cudaMemcpyHostToDevice);

            int blocks = (options.simulations + options.threads - 1) / options.threads;

            playout_kernel<<<blocks, options.threads>>>(
                d_cells,
                board.moves,
                current_side,
                move.row,
                move.col,
                options.simulations,
                options.seed + idx * 8129,
                d_win_counts
            );
            cudaDeviceSynchronize();

            cudaMemcpy(h_win_counts, d_win_counts, 3 * sizeof(int), cudaMemcpyDeviceToHost);

            int draws = h_win_counts[0];
            int black_wins = h_win_counts[1];
            int white_wins = h_win_counts[2];

            // Win rate for the side to move
            double win_rate = 0.0;
            if (current_side == 1) { // Black
                win_rate = (double)black_wins / options.simulations;
            } else { // White
                win_rate = (double)white_wins / options.simulations;
            }

            win_rates[idx] = win_rate;

            if (win_rate > best_win_rate) {
                best_win_rate = win_rate;
                best_r = move.row;
                best_c = move.col;
            }
        }

        const auto end_time = std::chrono::high_resolution_clock::now();
        const double duration = std::chrono::duration<double>(end_time - start_time).count();

        // Print rates for top moves
        std::cout << "Gomoku Monte Carlo CUDA\n";
        std::cout << "win rates for candidates:\n";
        for (size_t idx = 0; idx < std::min(candidates.size(), size_t(7)); ++idx) {
            std::cout << "move " << candidates[idx].row << "_" << candidates[idx].col 
                      << " win_rate: " << (win_rates[idx] * 100.0) << "%\n";
        }

        std::cout << "best_move " << best_r << "_" << best_c << '\n';
        std::cout << "duration " << duration << '\n';

        cudaFree(d_cells);
        cudaFree(d_win_counts);

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
