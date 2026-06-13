#include <connect4/board.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
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

__global__ void monte_carlo_kernel(DeviceBoard start, int simulations, std::uint32_t seed, Counts* counts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for block-level accumulation
    __shared__ unsigned long long s_black_wins;
    __shared__ unsigned long long s_white_wins;
    __shared__ unsigned long long s_draws;
    
    if (threadIdx.x == 0) {
        s_black_wins = 0;
        s_white_wins = 0;
        s_draws = 0;
    }
    __syncthreads();
    
    unsigned long long my_black = 0;
    unsigned long long my_white = 0;
    unsigned long long my_draw = 0;
    
    if (idx < simulations) {
        std::uint32_t rng = seed ^ (0x9E3779B9U * (idx + 1));
        const connect4::Outcome outcome = rollout(start, rng);
        if (outcome == connect4::Outcome::BlackWin) {
            my_black = 1;
        } else if (outcome == connect4::Outcome::WhiteWin) {
            my_white = 1;
        } else {
            my_draw = 1;
        }
    }
    
    // Add block-level atomicAdd to shared memory
    if (my_black) atomicAdd(&s_black_wins, 1ULL);
    if (my_white) atomicAdd(&s_white_wins, 1ULL);
    if (my_draw)  atomicAdd(&s_draws, 1ULL);
    __syncthreads();
    
    // Add block-level sums to the global counters once per block
    if (threadIdx.x == 0) {
        if (s_black_wins > 0) atomicAdd(&counts->black_wins, s_black_wins);
        if (s_white_wins > 0) atomicAdd(&counts->white_wins, s_white_wins);
        if (s_draws > 0)      atomicAdd(&counts->draws, s_draws);
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

std::vector<int> parse_moves(int argc, char** argv) {
    std::vector<int> moves;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--moves" && i + 1 < argc) {
            const std::string text = argv[++i];
            for (char ch : text) {
                if (ch >= '0' && ch <= '6') {
                    moves.push_back(ch - '0');
                }
            }
        }
    }
    return moves;
}

connect4::Board board_from_moves(const std::vector<int>& moves) {
    connect4::Board board;
    for (int col : moves) {
        const auto result = connect4::play(board, col);
        if (!result.legal) {
            throw std::runtime_error("illegal move in --moves");
        }
        board = result.board;
        const connect4::Outcome outcome = connect4::terminal_outcome(board);
        if (outcome != connect4::Outcome::Unknown) {
            break;
        }
    }
    return board;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const int simulations = parse_int_arg(argc, argv, "--simulations", 1'000'000);
        const int threads = parse_int_arg(argc, argv, "--threads", 256);
        const int seed = parse_int_arg(argc, argv, "--seed", 12345);
        if (simulations <= 0 || threads <= 0) {
            throw std::runtime_error("--simulations and --threads must be positive");
        }

        const connect4::Board host_board = board_from_moves(parse_moves(argc, argv));
        const connect4::Outcome already_done = connect4::terminal_outcome(host_board);
        if (already_done != connect4::Outcome::Unknown) {
            std::cout << "terminal=" << connect4::outcome_name(already_done) << "\n";
            return 0;
        }

        DeviceBoard start{
            host_board.black,
            host_board.white,
            host_board.moves,
            host_board.side_to_move == connect4::Player::Black ? std::uint8_t{0} : std::uint8_t{1},
        };

        Counts zero{0, 0, 0};
        Counts* device_counts = nullptr;
        check_cuda(cudaMalloc(&device_counts, sizeof(Counts)), "cudaMalloc");
        check_cuda(cudaMemcpy(device_counts, &zero, sizeof(Counts), cudaMemcpyHostToDevice), "cudaMemcpy H2D");

        const int blocks = (simulations + threads - 1) / threads;
        monte_carlo_kernel<<<blocks, threads>>>(start, simulations, static_cast<std::uint32_t>(seed), device_counts);
        check_cuda(cudaGetLastError(), "kernel launch");
        check_cuda(cudaDeviceSynchronize(), "kernel sync");

        Counts counts{};
        check_cuda(cudaMemcpy(&counts, device_counts, sizeof(Counts), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        check_cuda(cudaFree(device_counts), "cudaFree");

        std::cout << "simulations=" << simulations << "\n";
        std::cout << "black_wins=" << counts.black_wins << "\n";
        std::cout << "white_wins=" << counts.white_wins << "\n";
        std::cout << "draws=" << counts.draws << "\n";
        std::cout << "black_rate=" << static_cast<double>(counts.black_wins) / simulations << "\n";
        std::cout << "white_rate=" << static_cast<double>(counts.white_wins) / simulations << "\n";
        std::cout << "draw_rate=" << static_cast<double>(counts.draws) / simulations << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
