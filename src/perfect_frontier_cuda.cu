

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


#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int kWinScore = 100000;
constexpr int kInfinity = 1000000;

struct DeviceBoard {
    std::uint64_t black;
    std::uint64_t white;
    std::uint8_t moves;
    std::uint8_t side_to_move;
};

struct TreeNode {
    connect4::Board board;
    int parent = -1;
    int root_move = -1;
    int depth = 0;
    int score = 0;
    bool leaf = false;
    std::vector<int> children;
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

__device__ int terminal_score_device(const DeviceBoard& board) {
    const bool black_win = has_four_device(board.black);
    const bool white_win = has_four_device(board.white);
    if (!black_win && !white_win) {
        return board.moves == connect4::kMaxMoves ? 0 : kInfinity;
    }

    const std::uint8_t winner = black_win ? 0 : 1;
    const int score = kWinScore - static_cast<int>(board.moves);
    return winner == board.side_to_move ? score : -score;
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

__device__ int negamax_device(DeviceBoard board, int max_depth, int alpha, int beta) {
    const int terminal = terminal_score_device(board);
    if (terminal != kInfinity) {
        return terminal;
    }
    if (board.moves >= max_depth) {
        return 0;
    }

    constexpr int order[connect4::kCols] = {3, 2, 4, 1, 5, 0, 6};
    int best = -kInfinity;
    for (int i = 0; i < connect4::kCols; ++i) {
        const int col = order[i];
        if (column_height_device(board, col) >= connect4::kRows) {
            continue;
        }

        DeviceBoard child = board;
        play_device(child, col);
        const int score = -negamax_device(child, max_depth, -beta, -alpha);
        best = max(best, score);
        alpha = max(alpha, score);
        if (alpha >= beta) {
            break;
        }
    }
    return best == -kInfinity ? 0 : best;
}

__global__ void solve_leaves_kernel(const DeviceBoard* leaves, int leaf_count, int max_depth, int* scores) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= leaf_count) {
        return;
    }
    scores[idx] = negamax_device(leaves[idx], max_depth, -kInfinity, kInfinity);
}

void check_cuda(cudaError_t error, const char* label) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(error));
    }
}

void validate_thread_count(int threads) {
    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    if (threads > properties.maxThreadsPerBlock) {
        throw std::runtime_error(
            "--threads exceeds this GPU's maxThreadsPerBlock (" +
            std::to_string(properties.maxThreadsPerBlock) + ")");
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

const std::vector<int>& move_order() {
    static const std::vector<int> order{3, 2, 4, 1, 5, 0, 6};
    return order;
}

int add_node(std::vector<TreeNode>& nodes, const connect4::Board& board, int parent, int root_move, int depth) {
    TreeNode node;
    node.board = board;
    node.parent = parent;
    node.root_move = root_move;
    node.depth = depth;
    nodes.push_back(std::move(node));
    const int index = static_cast<int>(nodes.size()) - 1;
    if (parent >= 0) {
        nodes[parent].children.push_back(index);
    }
    return index;
}

void build_frontier(std::vector<TreeNode>& nodes, int node_index, int frontier_depth, int max_depth) {
    const connect4::Board board = nodes[node_index].board;
    const int depth = nodes[node_index].depth;
    const int root_move_from_parent = nodes[node_index].root_move;

    if (connect4::terminal_outcome(board) != connect4::Outcome::Unknown ||
        depth >= frontier_depth ||
        board.moves >= max_depth) {
        nodes[node_index].leaf = true;
        return;
    }

    for (int col : move_order()) {
        if (!connect4::can_play(board, col)) {
            continue;
        }
        const connect4::MoveResult child = connect4::play(board, col);
        const int root_move = root_move_from_parent >= 0 ? root_move_from_parent : col;
        const int child_index = add_node(nodes, child.board, node_index, root_move, depth + 1);
        build_frontier(nodes, child_index, frontier_depth, max_depth);
    }

    if (nodes[node_index].children.empty()) {
        nodes[node_index].leaf = true;
    }
}

int terminal_score_host(const connect4::Board& board) {
    const connect4::Outcome terminal = connect4::terminal_outcome(board);
    if (terminal == connect4::Outcome::Unknown) {
        return kInfinity;
    }
    if (terminal == connect4::Outcome::Draw) {
        return 0;
    }
    const connect4::Player winner =
        terminal == connect4::Outcome::BlackWin ? connect4::Player::Black : connect4::Player::White;
    const int score = kWinScore - static_cast<int>(board.moves);
    return winner == board.side_to_move ? score : -score;
}

const char* score_name(int score) {
    if (score > 0) {
        return "win";
    }
    if (score < 0) {
        return "loss";
    }
    return "draw";
}

void print_board(const connect4::Board& board) {
    for (int row = connect4::kRows - 1; row >= 0; --row) {
        std::cout << "|";
        for (int col = 0; col < connect4::kCols; ++col) {
            const std::uint64_t bit = connect4::bit_at(row, col);
            char ch = '.';
            if ((board.black & bit) != 0) {
                ch = 'B';
            } else if ((board.white & bit) != 0) {
                ch = 'W';
            }
            std::cout << ' ' << ch;
        }
        std::cout << " |\n";
    }
    std::cout << "  0 1 2 3 4 5 6\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const int max_depth = parse_int_arg(argc, argv, "--max-depth", 16);
        const int frontier_extra_depth = parse_int_arg(argc, argv, "--frontier-depth", 5);
        const int threads = parse_int_arg(argc, argv, "--threads", 128);
        const int stack_bytes = parse_int_arg(argc, argv, "--stack-bytes", 8192);
        const std::string moves_text = parse_string_arg(argc, argv, "--moves", "");

        if (max_depth < 0 || max_depth > connect4::kMaxMoves) {
            throw std::runtime_error("--max-depth must be between 0 and 42");
        }
        if (frontier_extra_depth < 1 || threads < 1 || stack_bytes < 1024) {
            throw std::runtime_error("--frontier-depth, --threads, and --stack-bytes are invalid");
        }

        const connect4::Board root_board = board_from_moves(parse_moves_text(moves_text));
        if (connect4::terminal_outcome(root_board) != connect4::Outcome::Unknown) {
            print_board(root_board);
            std::cout << "terminal position\n";
            return 0;
        }
        if (max_depth <= root_board.moves) {
            throw std::runtime_error("--max-depth must be greater than the number of moves in the position");
        }

        std::vector<TreeNode> nodes;
        nodes.reserve(100000);
        const int root = add_node(nodes, root_board, -1, -1, static_cast<int>(root_board.moves));
        const int frontier_depth = std::min(max_depth, static_cast<int>(root_board.moves) + frontier_extra_depth);
        build_frontier(nodes, root, frontier_depth, max_depth);

        std::vector<int> leaf_node_indices;
        std::vector<DeviceBoard> leaf_boards;
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            if (!nodes[i].leaf) {
                continue;
            }
            const int terminal = terminal_score_host(nodes[i].board);
            if (terminal != kInfinity) {
                nodes[i].score = terminal;
            } else {
                leaf_node_indices.push_back(i);
                leaf_boards.push_back(to_device_board(nodes[i].board));
            }
        }

        float kernel_ms = 0.0F;
        if (!leaf_boards.empty()) {
            validate_thread_count(threads);
            check_cuda(cudaDeviceSetLimit(cudaLimitStackSize, static_cast<std::size_t>(stack_bytes)), "cudaDeviceSetLimit stack");

            DeviceBoard* device_leaves = nullptr;
            int* device_scores = nullptr;
            std::vector<int> leaf_scores(leaf_boards.size(), 0);
            check_cuda(cudaMalloc(&device_leaves, leaf_boards.size() * sizeof(DeviceBoard)), "cudaMalloc leaves");
            check_cuda(cudaMalloc(&device_scores, leaf_scores.size() * sizeof(int)), "cudaMalloc scores");
            check_cuda(cudaMemcpy(device_leaves, leaf_boards.data(), leaf_boards.size() * sizeof(DeviceBoard), cudaMemcpyHostToDevice), "copy leaves");

            const int blocks = (static_cast<int>(leaf_boards.size()) + threads - 1) / threads;
            cudaEvent_t kernel_start = nullptr;
            cudaEvent_t kernel_stop = nullptr;
            check_cuda(cudaEventCreate(&kernel_start), "cudaEventCreate start");
            check_cuda(cudaEventCreate(&kernel_stop), "cudaEventCreate stop");
            check_cuda(cudaEventRecord(kernel_start), "cudaEventRecord start");
            solve_leaves_kernel<<<blocks, threads>>>(device_leaves, static_cast<int>(leaf_boards.size()), max_depth, device_scores);
            check_cuda(cudaGetLastError(), "kernel launch");
            check_cuda(cudaEventRecord(kernel_stop), "cudaEventRecord stop");
            check_cuda(cudaEventSynchronize(kernel_stop), "kernel sync");
            check_cuda(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop), "cudaEventElapsedTime");
            check_cuda(cudaEventDestroy(kernel_start), "cudaEventDestroy start");
            check_cuda(cudaEventDestroy(kernel_stop), "cudaEventDestroy stop");
            check_cuda(cudaMemcpy(leaf_scores.data(), device_scores, leaf_scores.size() * sizeof(int), cudaMemcpyDeviceToHost), "copy scores");
            check_cuda(cudaFree(device_leaves), "free leaves");
            check_cuda(cudaFree(device_scores), "free scores");

            for (std::size_t i = 0; i < leaf_node_indices.size(); ++i) {
                nodes[leaf_node_indices[i]].score = leaf_scores[i];
            }
        }

        for (int i = static_cast<int>(nodes.size()) - 1; i >= 0; --i) {
            if (nodes[i].children.empty()) {
                continue;
            }
            int best = -kInfinity;
            for (int child : nodes[i].children) {
                best = std::max(best, -nodes[child].score);
            }
            nodes[i].score = best == -kInfinity ? 0 : best;
        }

        int best_move = -1;
        int best_score = -kInfinity;
        std::vector<int> root_move_scores(connect4::kCols, -kInfinity);
        for (int child : nodes[root].children) {
            const int score = -nodes[child].score;
            root_move_scores[nodes[child].root_move] = score;
            if (score > best_score) {
                best_score = score;
                best_move = nodes[child].root_move;
            }
        }

        print_board(root_board);
        std::cout << "max_depth " << max_depth << "\n";
        std::cout << "frontier_depth " << frontier_depth << "\n";
        std::cout << "tree_nodes " << nodes.size() << "\n";
        std::cout << "gpu_leaves " << leaf_boards.size() << "\n";
        std::cout << "kernel_ms " << kernel_ms << "\n";
        std::cout << "result " << score_name(best_score) << "\n";
        std::cout << "score " << best_score << "\n";
        std::cout << "best_move " << best_move << "\n";
        std::cout << "move_scores";
        for (int col = 0; col < connect4::kCols; ++col) {
            if (root_move_scores[col] == -kInfinity) {
                std::cout << " " << col << ":full";
            } else {
                std::cout << " " << col << ":" << root_move_scores[col];
            }
        }
        std::cout << "\n";
        if (max_depth < connect4::kMaxMoves) {
            std::cout << "note this is limited-depth exact search, not full proof\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
