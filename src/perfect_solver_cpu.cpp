#include <connect4/board.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kWinScore = 100000;
constexpr int kInfinity = 1000000;

enum class Bound : std::uint8_t {
    Exact = 0,
    Lower = 1,
    Upper = 2,
};

struct Key {
    std::uint64_t black = 0;
    std::uint64_t white = 0;
    std::uint8_t side_to_move = 0;

    bool operator==(const Key& other) const {
        return black == other.black && white == other.white && side_to_move == other.side_to_move;
    }
};

struct KeyHash {
    std::size_t operator()(const Key& key) const {
        std::uint64_t x = key.black;
        x ^= key.white + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
        x ^= static_cast<std::uint64_t>(key.side_to_move) * 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return static_cast<std::size_t>(x);
    }
};

struct Entry {
    int score = 0;
    std::int8_t best_move = -1;
    std::uint8_t depth_remaining = 0;
    Bound bound = Bound::Exact;
};

struct Options {
    int max_depth = connect4::kMaxMoves;
    std::uint64_t limit_nodes = 0;
    std::uint64_t reserve = 1'000'000;
    std::optional<std::string> text_path;
    std::vector<int> moves;
};

struct Stats {
    std::uint64_t nodes = 0;
    std::uint64_t cache_hits = 0;
    std::uint64_t cutoffs = 0;
    bool hit_limit = false;
};

class PerfectSolver {
public:
    explicit PerfectSolver(const Options& options) : options_(options) {
        table_.reserve(static_cast<std::size_t>(options.reserve));
    }

    int solve(const connect4::Board& board) {
        return negamax(board, -kInfinity, kInfinity);
    }

    std::optional<Entry> lookup(const connect4::Board& board) const {
        const auto it = table_.find(make_key(board));
        if (it == table_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    void save_text(const std::string& path) const {
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("failed to open text file: " + path);
        }

        out << "black_hex,white_hex,side_to_move,score,result,best_move,depth_remaining,board\n";
        for (const auto& [key, entry] : table_) {
            out << hex64(key.black) << ','
                << hex64(key.white) << ','
                << (key.side_to_move == 0 ? "black" : "white") << ','
                << entry.score << ','
                << score_name(entry.score) << ','
                << static_cast<int>(entry.best_move) << ','
                << static_cast<int>(entry.depth_remaining) << ','
                << board_compact(key.black, key.white) << '\n';
        }
    }

    const Stats& stats() const {
        return stats_;
    }

    std::size_t table_size() const {
        return table_.size();
    }

private:
    int negamax(const connect4::Board& board, int alpha, int beta) {
        if (options_.limit_nodes != 0 && stats_.nodes >= options_.limit_nodes) {
            stats_.hit_limit = true;
            return 0;
        }

        ++stats_.nodes;

        const connect4::Outcome terminal = connect4::terminal_outcome(board);
        if (terminal != connect4::Outcome::Unknown) {
            return terminal_score(board, terminal);
        }

        const int depth_remaining = options_.max_depth - static_cast<int>(board.moves);
        if (depth_remaining <= 0) {
            return 0;
        }

        const Key key = make_key(board);
        const int original_alpha = alpha;

        const auto cached = table_.find(key);
        if (cached != table_.end() && cached->second.depth_remaining >= depth_remaining) {
            const Entry& entry = cached->second;
            if (entry.bound == Bound::Exact) {
                ++stats_.cache_hits;
                return entry.score;
            }
            if (entry.bound == Bound::Lower) {
                alpha = std::max(alpha, entry.score);
            } else if (entry.bound == Bound::Upper) {
                beta = std::min(beta, entry.score);
            }
            if (alpha >= beta) {
                ++stats_.cache_hits;
                return entry.score;
            }
        }

        int best_score = -kInfinity;
        std::int8_t best_move = -1;

        for (int col : move_order()) {
            if (!connect4::can_play(board, col)) {
                continue;
            }

            const connect4::MoveResult next = connect4::play(board, col);
            int score = 0;
            const connect4::Outcome child_terminal = connect4::terminal_outcome(next.board);
            if (child_terminal != connect4::Outcome::Unknown) {
                score = -terminal_score(next.board, child_terminal);
            } else {
                score = -negamax(next.board, -beta, -alpha);
            }

            if (score > best_score) {
                best_score = score;
                best_move = static_cast<std::int8_t>(col);
            }
            alpha = std::max(alpha, score);

            if (alpha >= beta || stats_.hit_limit) {
                ++stats_.cutoffs;
                break;
            }
        }

        Bound bound = Bound::Exact;
        if (best_score <= original_alpha) {
            bound = Bound::Upper;
        } else if (best_score >= beta) {
            bound = Bound::Lower;
        }

        table_[key] = Entry{
            best_score,
            best_move,
            static_cast<std::uint8_t>(depth_remaining),
            bound,
        };
        return best_score;
    }

    static int terminal_score(const connect4::Board& board, connect4::Outcome terminal) {
        if (terminal == connect4::Outcome::Draw) {
            return 0;
        }

        const connect4::Player winner =
            terminal == connect4::Outcome::BlackWin ? connect4::Player::Black : connect4::Player::White;
        const int score = kWinScore - static_cast<int>(board.moves);
        return winner == board.side_to_move ? score : -score;
    }

    static Key make_key(const connect4::Board& board) {
        return Key{
            board.black,
            board.white,
            board.side_to_move == connect4::Player::Black ? std::uint8_t{0} : std::uint8_t{1},
        };
    }

    static const std::array<int, connect4::kCols>& move_order() {
        static constexpr std::array<int, connect4::kCols> order{3, 2, 4, 1, 5, 0, 6};
        return order;
    }

    static const char* score_name(int score) {
        if (score > 0) {
            return "win";
        }
        if (score < 0) {
            return "loss";
        }
        return "draw";
    }

    static std::string hex64(std::uint64_t value) {
        std::ostringstream oss;
        oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
        return oss.str();
    }

    static std::string board_compact(std::uint64_t black, std::uint64_t white) {
        std::string text;
        text.reserve(connect4::kRows * connect4::kCols + connect4::kRows - 1);
        for (int row = connect4::kRows - 1; row >= 0; --row) {
            for (int col = 0; col < connect4::kCols; ++col) {
                const std::uint64_t bit = connect4::bit_at(row, col);
                if ((black & bit) != 0) {
                    text.push_back('B');
                } else if ((white & bit) != 0) {
                    text.push_back('W');
                } else {
                    text.push_back('.');
                }
            }
            if (row != 0) {
                text.push_back('/');
            }
        }
        return text;
    }

    const Options& options_;
    Stats stats_;
    std::unordered_map<Key, Entry, KeyHash> table_;
};

std::uint64_t parse_u64(std::string_view text, std::string_view flag) {
    if (text.empty()) {
        throw std::runtime_error(std::string(flag) + " requires a value");
    }
    std::uint64_t value = 0;
    for (char ch : text) {
        if (ch < '0' || ch > '9') {
            throw std::runtime_error(std::string(flag) + " requires a base-10 integer");
        }
        const std::uint64_t digit = static_cast<std::uint64_t>(ch - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
            throw std::runtime_error(std::string(flag) + " is too large");
        }
        value = value * 10 + digit;
    }
    return value;
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto require_value = [&](std::string_view flag) -> std::string_view {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string(flag) + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--moves") {
            const std::string_view moves = require_value(arg);
            for (char ch : moves) {
                if (ch < '0' || ch > '6') {
                    throw std::runtime_error("--moves must contain only columns 0..6");
                }
                options.moves.push_back(ch - '0');
            }
        } else if (arg == "--max-depth") {
            const std::uint64_t depth = parse_u64(require_value(arg), arg);
            if (depth > connect4::kMaxMoves) {
                throw std::runtime_error("--max-depth must be between 0 and 42");
            }
            options.max_depth = static_cast<int>(depth);
        } else if (arg == "--limit-nodes") {
            options.limit_nodes = parse_u64(require_value(arg), arg);
        } else if (arg == "--reserve") {
            options.reserve = parse_u64(require_value(arg), arg);
        } else if (arg == "--text") {
            options.text_path = std::string(require_value(arg));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: perfect_solver_cpu [--moves COLUMNS] [--max-depth N]\n"
                << "                          [--limit-nodes N] [--reserve N] [--text CSV]\n"
                << "\n"
                << "Negamax/alpha-beta solver. Positive score means the side to move can force a win.\n"
                << "Default max depth is 42, which is a full-width perfect search and can take a long time.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    return options;
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

const char* score_name(int score) {
    if (score > 0) {
        return "win";
    }
    if (score < 0) {
        return "loss";
    }
    return "draw";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const connect4::Board root = board_from_moves(options.moves);

        PerfectSolver solver(options);
        const int score = solver.solve(root);
        const std::optional<Entry> root_entry = solver.lookup(root);

        if (options.text_path.has_value()) {
            solver.save_text(*options.text_path);
        }

        const Stats& stats = solver.stats();
        std::cout << "result " << score_name(score) << '\n';
        std::cout << "score " << score << '\n';
        std::cout << "best_move " << (root_entry.has_value() ? static_cast<int>(root_entry->best_move) : -1) << '\n';
        std::cout << "nodes " << stats.nodes << '\n';
        std::cout << "table_states " << solver.table_size() << '\n';
        std::cout << "cache_hits " << stats.cache_hits << '\n';
        std::cout << "cutoffs " << stats.cutoffs << '\n';
        std::cout << "truncated " << (stats.hit_limit ? "yes" : "no") << '\n';
        if (options.max_depth < connect4::kMaxMoves) {
            std::cout << "note max-depth is partial search, not a proof\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        std::cerr << "try --help\n";
        return 1;
    }
    return 0;
}
