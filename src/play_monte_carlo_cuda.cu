#include <connect4/board.hpp>

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct DeviceBoard {
    std::uint64_t black;
    std::uint64_t white;
    std::uint8_t moves;
    std::uint8_t side_to_move;
};

struct Counts {
    unsigned long long black_wins;
    unsigned long long white_wins;
    unsigned long long draws;
};

struct CandidateStats {
    int col = -1;
    Counts counts{0, 0, 0};
    double computer_win_rate = 0.0;
    double draw_rate = 0.0;
    double score = 0.0;
};

__host__ __device__ constexpr std::uint64_t bit_at_device(int row, int col) {
    return 1ULL << (row + col * connect4::kStride);
}

__host__ __device__ int column_height_device(const DeviceBoard& board, int col) {
    const std::uint64_t occ = board.black | board.white;
    for (int row = 0; row < connect4::kRows; ++row) {
        if ((occ & bit_at_device(row, col)) == 0) {
            return row;
        }
    }
    return connect4::kRows;
}

__host__ __device__ bool has_four_device(std::uint64_t stones) {
    std::uint64_t connected = stones & (stones >> 1);
    if ((connected & (connected >> 2)) != 0) {
        return true;
    }

    connected = stones & (stones >> connect4::kStride);
    if ((connected & (connected >> (2 * connect4::kStride))) != 0) {
        return true;
    }

    connected = stones & (stones >> (connect4::kStride - 1));
    if ((connected & (connected >> (2 * (connect4::kStride - 1)))) != 0) {
        return true;
    }

    connected = stones & (stones >> (connect4::kStride + 1));
    if ((connected & (connected >> (2 * (connect4::kStride + 1)))) != 0) {
        return true;
    }

    return false;
}

__device__ std::uint32_t xorshift32(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ int heuristic_legal_move(const DeviceBoard& board, std::uint32_t& rng) {
    int legal[connect4::kCols];
    int count = 0;
    for (int col = 0; col < connect4::kCols; ++col) {
        if (column_height_device(board, col) < connect4::kRows) {
            legal[count++] = col;
        }
    }
    if (count == 0) {
        return -1;
    }

    // 1. Immediate win check for active player
    std::uint64_t my_stones = (board.side_to_move == 0) ? board.black : board.white;
    for (int i = 0; i < count; ++i) {
        int col = legal[i];
        int row = column_height_device(board, col);
        std::uint64_t bit = bit_at_device(row, col);
        if (has_four_device(my_stones | bit)) {
            return col;
        }
    }

    // 2. Block opponent's immediate win
    std::uint64_t opp_stones = (board.side_to_move == 0) ? board.white : board.black;
    for (int i = 0; i < count; ++i) {
        int col = legal[i];
        int row = column_height_device(board, col);
        std::uint64_t bit = bit_at_device(row, col);
        if (has_four_device(opp_stones | bit)) {
            return col;
        }
    }

    // 3. Avoid giving opponent an immediate win
    int safe_legal[connect4::kCols];
    int safe_count = 0;
    for (int i = 0; i < count; ++i) {
        int col = legal[i];
        int row = column_height_device(board, col);
        if (row + 1 < connect4::kRows) {
            std::uint64_t opp_bit_above = bit_at_device(row + 1, col);
            if (!has_four_device(opp_stones | opp_bit_above)) {
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

__device__ void play_device(DeviceBoard& board, int col) {
    const int row = column_height_device(board, col);
    const std::uint64_t bit = bit_at_device(row, col);
    if (board.side_to_move == 0) {
        board.black |= bit;
    } else {
        board.white |= bit;
    }
    board.side_to_move = 1 - board.side_to_move;
    ++board.moves;
}

__device__ connect4::Outcome rollout(DeviceBoard board, std::uint32_t& rng) {
    while (board.moves < connect4::kMaxMoves) {
        if (has_four_device(board.black)) {
            return connect4::Outcome::BlackWin;
        }
        if (has_four_device(board.white)) {
            return connect4::Outcome::WhiteWin;
        }

        const int col = heuristic_legal_move(board, rng);
        if (col < 0) {
            break;
        }
        play_device(board, col);
    }

    if (has_four_device(board.black)) {
        return connect4::Outcome::BlackWin;
    }
    if (has_four_device(board.white)) {
        return connect4::Outcome::WhiteWin;
    }
    return connect4::Outcome::Draw;
}

__global__ void evaluate_kernel(
    const DeviceBoard* starts,
    int candidates,
    int simulations_per_candidate,
    std::uint32_t seed,
    Counts* counts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = candidates * simulations_per_candidate;
    
    // Shared memory for candidate counters in this block (max 7 candidates)
    __shared__ unsigned long long s_black_wins[7];
    __shared__ unsigned long long s_white_wins[7];
    __shared__ unsigned long long s_draws[7];
    
    if (threadIdx.x < 7) {
        s_black_wins[threadIdx.x] = 0;
        s_white_wins[threadIdx.x] = 0;
        s_draws[threadIdx.x] = 0;
    }
    __syncthreads();
    
    int my_candidate = -1;
    connect4::Outcome outcome = connect4::Outcome::Unknown;
    
    if (idx < total) {
        my_candidate = idx / simulations_per_candidate;
        const int sim = idx - my_candidate * simulations_per_candidate;
        std::uint32_t rng = seed ^ (0x9E3779B9U * (idx + 1)) ^ (0x85EBCA6BU * (my_candidate + 17));
        rng ^= static_cast<std::uint32_t>(starts[my_candidate].black + starts[my_candidate].white + sim);
        
        outcome = rollout(starts[my_candidate], rng);
    }
    
    // Accumulate locally using block-level atomicAdd to shared memory
    if (my_candidate >= 0) {
        if (outcome == connect4::Outcome::BlackWin) {
            atomicAdd(&s_black_wins[my_candidate], 1ULL);
        } else if (outcome == connect4::Outcome::WhiteWin) {
            atomicAdd(&s_white_wins[my_candidate], 1ULL);
        } else {
            atomicAdd(&s_draws[my_candidate], 1ULL);
        }
    }
    __syncthreads();
    
    // Write back block totals to global memory once per block
    if (threadIdx.x < 7 && threadIdx.x < candidates) {
        const int c = threadIdx.x;
        if (s_black_wins[c] > 0) atomicAdd(&counts[c].black_wins, s_black_wins[c]);
        if (s_white_wins[c] > 0) atomicAdd(&counts[c].white_wins, s_white_wins[c]);
        if (s_draws[c] > 0)      atomicAdd(&counts[c].draws, s_draws[c]);
    }
}


void check_cuda(cudaError_t error, const char* label) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(error));
    }
}

int parse_int_arg(int argc, char** argv, const std::string& name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return std::atoi(argv[i + 1]);
        }
    }
    return fallback;
}

bool has_flag(int argc, char** argv, const std::string& name) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) {
            return true;
        }
    }
    return false;
}

std::string parse_string_arg(int argc, char** argv, const std::string& name, const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

std::vector<int> parse_moves_text(const std::string& text) {
    std::vector<int> moves;
    moves.reserve(text.size());
    for (char ch : text) {
        if (ch < '0' || ch > '6') {
            throw std::runtime_error("--moves must contain only columns 0..6");
        }
        moves.push_back(ch - '0');
    }
    return moves;
}

connect4::Board board_from_moves(const std::vector<int>& moves) {
    connect4::Board board;
    for (int col : moves) {
        const connect4::MoveResult next = connect4::play(board, col);
        if (!next.legal) {
            throw std::runtime_error("illegal move in --moves");
        }
        board = next.board;
        if (connect4::terminal_outcome(board) != connect4::Outcome::Unknown) {
            break;
        }
    }
    return board;
}

DeviceBoard to_device_board(const connect4::Board& board) {
    return DeviceBoard{
        board.black,
        board.white,
        board.moves,
        board.side_to_move == connect4::Player::Black ? std::uint8_t{0} : std::uint8_t{1},
    };
}

const char* player_name(connect4::Player player) {
    return player == connect4::Player::Black ? "black" : "white";
}

char stone_char(connect4::Player player, connect4::Player human, connect4::Player computer) {
    if (player == human) {
        return 'X';
    }
    if (player == computer) {
        return 'O';
    }
    return '?';
}

void print_board(const connect4::Board& board, connect4::Player human, connect4::Player computer) {
    std::cout << "\n";
    for (int row = connect4::kRows - 1; row >= 0; --row) {
        std::cout << "|";
        for (int col = 0; col < connect4::kCols; ++col) {
            const std::uint64_t bit = connect4::bit_at(row, col);
            char ch = '.';
            if ((board.black & bit) != 0) {
                ch = stone_char(connect4::Player::Black, human, computer);
            } else if ((board.white & bit) != 0) {
                ch = stone_char(connect4::Player::White, human, computer);
            }
            std::cout << ' ' << ch;
        }
        std::cout << " |\n";
    }
    std::cout << "  0 1 2 3 4 5 6\n";
    std::cout << "X=you(" << player_name(human) << "), O=computer(" << player_name(computer) << ")\n";
}

void print_rates(const std::array<CandidateStats, connect4::kCols>& by_col, int simulations_per_move) {
    std::cout << "\ncomputer win rate after each legal move (" << simulations_per_move << " simulations each)\n";
    std::cout << "col: ";
    for (int col = 0; col < connect4::kCols; ++col) {
        std::cout << std::setw(7) << col;
    }
    std::cout << "\nwin: ";
    for (int col = 0; col < connect4::kCols; ++col) {
        if (by_col[col].col < 0) {
            std::cout << std::setw(7) << "full";
        } else {
            std::cout << std::setw(6) << std::fixed << std::setprecision(1) << (by_col[col].computer_win_rate * 100.0) << "%";
        }
    }
    std::cout << "\ndraw:";
    for (int col = 0; col < connect4::kCols; ++col) {
        if (by_col[col].col < 0) {
            std::cout << std::setw(7) << "-";
        } else {
            std::cout << std::setw(6) << std::fixed << std::setprecision(1) << (by_col[col].draw_rate * 100.0) << "%";
        }
    }
    std::cout << "\n";
}

std::array<CandidateStats, connect4::kCols> evaluate_moves(
    const connect4::Board& board,
    connect4::Player computer,
    int simulations_per_move,
    int threads,
    std::uint32_t seed) {
    std::vector<DeviceBoard> starts;
    std::vector<int> cols;
    starts.reserve(connect4::kCols);
    cols.reserve(connect4::kCols);

    for (int col = 0; col < connect4::kCols; ++col) {
        if (!connect4::can_play(board, col)) {
            continue;
        }
        const connect4::MoveResult next = connect4::play(board, col);
        starts.push_back(to_device_board(next.board));
        cols.push_back(col);
    }

    if (starts.empty()) {
        throw std::runtime_error("no legal moves to evaluate");
    }

    DeviceBoard* device_starts = nullptr;
    Counts* device_counts = nullptr;
    std::vector<Counts> zero(starts.size(), Counts{0, 0, 0});
    std::vector<Counts> host_counts(starts.size(), Counts{0, 0, 0});

    check_cuda(cudaMalloc(&device_starts, starts.size() * sizeof(DeviceBoard)), "cudaMalloc starts");
    check_cuda(cudaMalloc(&device_counts, starts.size() * sizeof(Counts)), "cudaMalloc counts");
    check_cuda(cudaMemcpy(device_starts, starts.data(), starts.size() * sizeof(DeviceBoard), cudaMemcpyHostToDevice), "copy starts");
    check_cuda(cudaMemcpy(device_counts, zero.data(), zero.size() * sizeof(Counts), cudaMemcpyHostToDevice), "zero counts");

    const int total = static_cast<int>(starts.size()) * simulations_per_move;
    const int blocks = (total + threads - 1) / threads;
    evaluate_kernel<<<blocks, threads>>>(device_starts, static_cast<int>(starts.size()), simulations_per_move, seed, device_counts);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel sync");
    check_cuda(cudaMemcpy(host_counts.data(), device_counts, host_counts.size() * sizeof(Counts), cudaMemcpyDeviceToHost), "copy counts");
    check_cuda(cudaFree(device_starts), "free starts");
    check_cuda(cudaFree(device_counts), "free counts");

    std::array<CandidateStats, connect4::kCols> by_col{};
    for (int col = 0; col < connect4::kCols; ++col) {
        by_col[col].col = -1;
    }

    for (std::size_t i = 0; i < cols.size(); ++i) {
        const Counts counts = host_counts[i];
        const double denom = static_cast<double>(simulations_per_move);
        const double computer_wins =
            computer == connect4::Player::Black ? static_cast<double>(counts.black_wins) : static_cast<double>(counts.white_wins);
        CandidateStats stats;
        stats.col = cols[i];
        stats.counts = counts;
        stats.computer_win_rate = computer_wins / denom;
        stats.draw_rate = static_cast<double>(counts.draws) / denom;
        stats.score = stats.computer_win_rate + 0.5 * stats.draw_rate;
        by_col[cols[i]] = stats;
    }

    return by_col;
}

int choose_best_move(const std::array<CandidateStats, connect4::kCols>& by_col) {
    int best_col = -1;
    double best_score = -1.0;
    for (int col = 0; col < connect4::kCols; ++col) {
        if (by_col[col].col < 0) {
            continue;
        }
        if (by_col[col].score > best_score) {
            best_score = by_col[col].score;
            best_col = col;
        }
    }
    return best_col;
}

int play_computer_turn(
    connect4::Board& board,
    connect4::Player computer,
    int simulations_per_move,
    int threads,
    std::uint32_t seed) {
    std::cout << "computer thinking...\n";
    const auto rates = evaluate_moves(board, computer, simulations_per_move, threads, seed);
    print_rates(rates, simulations_per_move);
    const int best_col = choose_best_move(rates);
    if (best_col < 0) {
        throw std::runtime_error("no legal move");
    }

    std::cout << "computer plays column " << best_col << "\n";
    board = connect4::play(board, best_col).board;
    return best_col;
}

bool print_terminal_if_done(const connect4::Board& board, connect4::Player human, connect4::Player computer) {
    const connect4::Outcome outcome = connect4::terminal_outcome(board);
    if (outcome == connect4::Outcome::Unknown) {
        return false;
    }
    if (outcome == connect4::Outcome::Draw) {
        std::cout << "draw\n";
    } else {
        const connect4::Player winner = outcome == connect4::Outcome::BlackWin ? connect4::Player::Black : connect4::Player::White;
        if (winner == human) {
            std::cout << "you win\n";
        } else if (winner == computer) {
            std::cout << "computer wins\n";
        } else {
            std::cout << player_name(winner) << " wins\n";
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const int simulations_per_move = parse_int_arg(argc, argv, "--simulations-per-move", 100'000);
        const int threads = parse_int_arg(argc, argv, "--threads", 256);
        const int seed = parse_int_arg(argc, argv, "--seed", 12345);
        if (simulations_per_move <= 0 || threads <= 0) {
            throw std::runtime_error("--simulations-per-move and --threads must be positive");
        }

        const bool computer_first = has_flag(argc, argv, "--computer-first");
        const connect4::Player computer = computer_first ? connect4::Player::Black : connect4::Player::White;
        const connect4::Player human = connect4::other(computer);
        const std::string moves_text = parse_string_arg(argc, argv, "--moves", "");
        const bool non_interactive = !moves_text.empty() || has_flag(argc, argv, "--once");

        connect4::Board board = board_from_moves(parse_moves_text(moves_text));
        std::cout << "Connect Four Monte Carlo CUDA\n";
        if (!non_interactive) {
            std::cout << "Enter a column 0..6, or q to quit.\n";
        }

        if (non_interactive) {
            print_board(board, human, computer);
            if (print_terminal_if_done(board, human, computer)) {
                return 0;
            }
            if (board.side_to_move == human) {
                std::cout << "human_to_move yes\n";
                std::cout << "moves_so_far " << moves_text << "\n";
                std::cout << "append your column to --moves and run again\n";
                return 0;
            }

            const int best_col = play_computer_turn(
                board,
                computer,
                simulations_per_move,
                threads,
                static_cast<std::uint32_t>(seed + board.moves * 1009));
            print_board(board, human, computer);
            print_terminal_if_done(board, human, computer);
            std::cout << "best_move " << best_col << "\n";
            std::cout << "next_moves " << moves_text << best_col << "\n";
            return 0;
        }

        while (true) {
            print_board(board, human, computer);
            if (print_terminal_if_done(board, human, computer)) {
                break;
            }

            if (board.side_to_move == human) {
                std::cout << "your move: ";
                std::string input;
                if (!(std::cin >> input) || input == "q" || input == "quit") {
                    std::cout << "quit\n";
                    break;
                }
                if (input.size() != 1 || input[0] < '0' || input[0] > '6') {
                    std::cout << "enter one column number from 0 to 6\n";
                    continue;
                }
                const int col = input[0] - '0';
                const connect4::MoveResult next = connect4::play(board, col);
                if (!next.legal) {
                    std::cout << "column " << col << " is full\n";
                    continue;
                }
                board = next.board;
                continue;
            }

            play_computer_turn(
                board,
                computer,
                simulations_per_move,
                threads,
                static_cast<std::uint32_t>(seed + board.moves * 1009));
        }
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
