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

enum class Result : std::uint8_t {
    Unknown = 0,
    Win = 1,
    Loss = 2,
    Draw = 3,
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
    Result result = Result::Unknown;
    std::int8_t best_move = -1;
    std::uint8_t depth = 0;
};

struct Options {
    int max_depth = connect4::kMaxMoves;
    std::uint64_t limit_states = 0;
    std::optional<std::string> save_path;
    std::optional<std::string> load_path;
    std::optional<std::string> text_path;
    std::vector<int> moves;
};

struct Stats {
    std::uint64_t calls = 0;
    std::uint64_t cache_hits = 0;
    std::uint64_t solved = 0;
    bool hit_limit = false;
};

constexpr std::uint32_t kMagic = 0x31465343u; // "CSF1", little-endian.
constexpr std::uint16_t kVersion = 1;

struct FileHeader {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t record_size;
    std::uint64_t record_count;
    std::uint8_t rows;
    std::uint8_t cols;
    std::uint8_t reserved[6];
};

struct FileRecord {
    std::uint64_t black;
    std::uint64_t white;
    std::uint8_t side_to_move;
    std::uint8_t result;
    std::int8_t best_move;
    std::uint8_t depth;
    std::uint8_t reserved[4];
};

static_assert(sizeof(FileHeader) == 24);
static_assert(sizeof(FileRecord) == 24);

class Solver {
public:
    explicit Solver(Options options) : options_(std::move(options)) {}

    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open load file: " + path);
        }

        FileHeader header{};
        read_pod(in, header, "header");
        if (header.magic != kMagic || header.version != kVersion || header.record_size != sizeof(FileRecord)) {
            throw std::runtime_error("unsupported solver file format: " + path);
        }

        table_.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(header.record_count, 50'000'000ULL)));
        for (std::uint64_t i = 0; i < header.record_count; ++i) {
            FileRecord record{};
            read_pod(in, record, "record");
            table_[Key{record.black, record.white, record.side_to_move}] =
                Entry{static_cast<Result>(record.result), record.best_move, record.depth};
        }
    }

    Result solve(const connect4::Board& board, int depth) {
        ++stats_.calls;
        const Key key = make_key(board);
        const auto cached = table_.find(key);
        if (cached != table_.end()) {
            if (cached->second.result != Result::Unknown || depth >= options_.max_depth) {
                ++stats_.cache_hits;
                return cached->second.result;
            }
        }

        if (limit_reached()) {
            stats_.hit_limit = true;
            return Result::Unknown;
        }

        const connect4::Outcome terminal = connect4::terminal_outcome(board);
        if (terminal != connect4::Outcome::Unknown) {
            const Result result = terminal_to_result(board, terminal);
            table_[key] = Entry{result, -1, static_cast<std::uint8_t>(depth)};
            ++stats_.solved;
            return result;
        }

        if (depth >= options_.max_depth) {
            table_[key] = Entry{Result::Unknown, -1, static_cast<std::uint8_t>(depth)};
            ++stats_.solved;
            return Result::Unknown;
        }

        bool has_draw = false;
        bool has_unknown = false;
        std::int8_t draw_move = -1;
        std::int8_t unknown_move = -1;

        for (int col : move_order()) {
            if (!connect4::can_play(board, col)) {
                continue;
            }

            const connect4::MoveResult next = connect4::play(board, col);
            const Result child = solve(next.board, depth + 1);
            if (child == Result::Loss) {
                table_[key] = Entry{Result::Win, static_cast<std::int8_t>(col), static_cast<std::uint8_t>(depth)};
                ++stats_.solved;
                return Result::Win;
            }
            if (child == Result::Draw) {
                has_draw = true;
                if (draw_move < 0) {
                    draw_move = static_cast<std::int8_t>(col);
                }
            } else if (child == Result::Unknown) {
                has_unknown = true;
                if (unknown_move < 0) {
                    unknown_move = static_cast<std::int8_t>(col);
                }
            }
            if (stats_.hit_limit) {
                break;
            }
        }

        Entry entry;
        entry.depth = static_cast<std::uint8_t>(depth);
        if (has_draw) {
            entry.result = Result::Draw;
            entry.best_move = draw_move;
        } else if (has_unknown) {
            entry.result = Result::Unknown;
            entry.best_move = unknown_move;
        } else {
            entry.result = Result::Loss;
            entry.best_move = first_legal_move(board);
        }

        table_[key] = entry;
        ++stats_.solved;
        return entry.result;
    }

    void save_binary(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("failed to open save file: " + path);
        }

        const FileHeader header{
            kMagic,
            kVersion,
            static_cast<std::uint16_t>(sizeof(FileRecord)),
            static_cast<std::uint64_t>(table_.size()),
            static_cast<std::uint8_t>(connect4::kRows),
            static_cast<std::uint8_t>(connect4::kCols),
            {0, 0, 0, 0, 0, 0},
        };
        write_pod(out, header);

        for (const auto& [key, entry] : table_) {
            const FileRecord record{
                key.black,
                key.white,
                key.side_to_move,
                static_cast<std::uint8_t>(entry.result),
                entry.best_move,
                entry.depth,
                {0, 0, 0, 0},
            };
            write_pod(out, record);
        }
    }

    void save_text(const std::string& path) const {
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("failed to open text file: " + path);
        }

        out << "black_hex,white_hex,side_to_move,result,best_move,depth,board\n";
        std::vector<std::pair<Key, Entry>> rows(table_.begin(), table_.end());
        std::sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first.black != rhs.first.black) {
                return lhs.first.black < rhs.first.black;
            }
            if (lhs.first.white != rhs.first.white) {
                return lhs.first.white < rhs.first.white;
            }
            return lhs.first.side_to_move < rhs.first.side_to_move;
        });

        for (const auto& [key, entry] : rows) {
            out << hex64(key.black) << ','
                << hex64(key.white) << ','
                << side_name(key.side_to_move) << ','
                << result_name(entry.result) << ','
                << static_cast<int>(entry.best_move) << ','
                << static_cast<int>(entry.depth) << ','
                << board_compact(key.black, key.white) << '\n';
        }
    }

    const Stats& stats() const {
        return stats_;
    }

    const std::unordered_map<Key, Entry, KeyHash>& table() const {
        return table_;
    }

    std::optional<Entry> lookup(const connect4::Board& board) const {
        const auto it = table_.find(make_key(board));
        if (it == table_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

private:
    static Key make_key(const connect4::Board& board) {
        return Key{
            board.black,
            board.white,
            board.side_to_move == connect4::Player::Black ? std::uint8_t{0} : std::uint8_t{1},
        };
    }

    static Result terminal_to_result(const connect4::Board& board, connect4::Outcome terminal) {
        const bool black_to_move = board.side_to_move == connect4::Player::Black;
        if (terminal == connect4::Outcome::Draw) {
            return Result::Draw;
        }
        if (terminal == connect4::Outcome::BlackWin) {
            return black_to_move ? Result::Win : Result::Loss;
        }
        if (terminal == connect4::Outcome::WhiteWin) {
            return black_to_move ? Result::Loss : Result::Win;
        }
        return Result::Unknown;
    }

    bool limit_reached() const {
        return options_.limit_states != 0 && table_.size() >= options_.limit_states;
    }

    static const std::array<int, connect4::kCols>& move_order() {
        static constexpr std::array<int, connect4::kCols> order{3, 2, 4, 1, 5, 0, 6};
        return order;
    }

    static std::int8_t first_legal_move(const connect4::Board& board) {
        for (int col : move_order()) {
            if (connect4::can_play(board, col)) {
                return static_cast<std::int8_t>(col);
            }
        }
        return -1;
    }

    template <typename T>
    static void read_pod(std::ifstream& in, T& value, const char* label) {
        in.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!in) {
            throw std::runtime_error(std::string("failed to read ") + label);
        }
    }

    template <typename T>
    static void write_pod(std::ofstream& out, const T& value) {
        out.write(reinterpret_cast<const char*>(&value), sizeof(T));
        if (!out) {
            throw std::runtime_error("failed to write solver file");
        }
    }

    static std::string hex64(std::uint64_t value) {
        std::ostringstream oss;
        oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
        return oss.str();
    }

    static const char* result_name(Result result) {
        switch (result) {
        case Result::Win:
            return "win";
        case Result::Loss:
            return "loss";
        case Result::Draw:
            return "draw";
        default:
            return "unknown";
        }
    }

    static const char* side_name(std::uint8_t side) {
        return side == 0 ? "black" : "white";
    }

    static std::string board_compact(std::uint64_t black, std::uint64_t white) {
        std::string text;
        text.reserve(connect4::kRows * connect4::kCols);
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

    Options options_;
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

        if (arg == "--max-depth") {
            const std::uint64_t value = parse_u64(require_value(arg), arg);
            if (value > connect4::kMaxMoves) {
                throw std::runtime_error("--max-depth must be between 0 and 42");
            }
            options.max_depth = static_cast<int>(value);
        } else if (arg == "--limit-states") {
            options.limit_states = parse_u64(require_value(arg), arg);
        } else if (arg == "--save") {
            options.save_path = std::string(require_value(arg));
        } else if (arg == "--load") {
            options.load_path = std::string(require_value(arg));
        } else if (arg == "--text") {
            options.text_path = std::string(require_value(arg));
        } else if (arg == "--moves") {
            const std::string_view moves = require_value(arg);
            for (char ch : moves) {
                if (ch < '0' || ch > '6') {
                    throw std::runtime_error("--moves must contain only columns 0..6");
                }
                options.moves.push_back(ch - '0');
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: solver_cpu [--moves COLUMNS] [--max-depth N] [--limit-states N]\n"
                << "                  [--load FILE] [--save FILE] [--text CSV]\n"
                << "\n"
                << "Solves states with memoized minimax. Results are for the side to move.\n"
                << "Result win means the side to move has a forced win; loss means it cannot avoid losing.\n";
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

const char* result_name(Result result) {
    switch (result) {
    case Result::Win:
        return "win";
    case Result::Loss:
        return "loss";
    case Result::Draw:
        return "draw";
    default:
        return "unknown";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_args(argc, argv);
        Solver solver(options);

        if (options.load_path.has_value()) {
            solver.load(*options.load_path);
        }

        const connect4::Board root = board_from_moves(options.moves);
        const Result result = solver.solve(root, static_cast<int>(root.moves));
        const std::optional<Entry> root_entry = solver.lookup(root);

        if (options.save_path.has_value()) {
            solver.save_binary(*options.save_path);
        }
        if (options.text_path.has_value()) {
            solver.save_text(*options.text_path);
        }

        const Stats& stats = solver.stats();
        std::cout << "root_result " << result_name(result) << '\n';
        std::cout << "best_move " << (root_entry.has_value() ? static_cast<int>(root_entry->best_move) : -1) << '\n';
        std::cout << "states " << solver.table().size() << '\n';
        std::cout << "calls " << stats.calls << '\n';
        std::cout << "cache_hits " << stats.cache_hits << '\n';
        std::cout << "solved " << stats.solved << '\n';
        std::cout << "truncated " << (stats.hit_limit ? "yes" : "no") << '\n';
        if (options.max_depth < connect4::kMaxMoves) {
            std::cout << "note max-depth can leave unknown partial results\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        std::cerr << "try --help\n";
        return 1;
    }
    return 0;
}
