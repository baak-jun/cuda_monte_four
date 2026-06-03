#include <connect4/board.hpp>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr std::uint32_t kDumpMagic = 0x34464643u; // "CFF4", little-endian.
constexpr std::uint16_t kDumpVersion = 1;

enum class Outcome : std::uint8_t {
    Ongoing = 0,
    BlackWin = 1,
    WhiteWin = 2,
    Draw = 3,
};

struct Options {
    int max_depth = connect4::kMaxMoves;
    std::uint64_t limit_states = 0;
    std::optional<std::string> dump_path;
};

struct Counts {
    std::uint64_t visited = 0;
    std::uint64_t black_wins = 0;
    std::uint64_t white_wins = 0;
    std::uint64_t draws = 0;
    bool hit_limit = false;
};

struct DumpHeader {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t record_size;
    std::uint8_t rows;
    std::uint8_t cols;
    std::uint8_t reserved[6];
};

struct DumpRecord {
    std::uint64_t black;
    std::uint64_t white;
    std::uint8_t side_to_move;
    std::uint8_t outcome;
    std::uint8_t reserved[6];
};

static_assert(sizeof(DumpHeader) == 16);
static_assert(sizeof(DumpRecord) == 24);

class DumpWriter {
public:
    explicit DumpWriter(const std::string& path) : out_(path, std::ios::binary) {
        if (!out_) {
            throw std::runtime_error("failed to open dump file: " + path);
        }

        const DumpHeader header{
            kDumpMagic,
            kDumpVersion,
            static_cast<std::uint16_t>(sizeof(DumpRecord)),
            static_cast<std::uint8_t>(connect4::kRows),
            static_cast<std::uint8_t>(connect4::kCols),
            {0, 0, 0, 0, 0, 0},
        };
        write_pod(header);
    }

    void write(std::uint64_t black, std::uint64_t white, std::uint8_t side_to_move, Outcome outcome) {
        const DumpRecord record{
            black,
            white,
            side_to_move,
            static_cast<std::uint8_t>(outcome),
            {0, 0, 0, 0, 0, 0},
        };
        write_pod(record);
    }

private:
    template <typename T>
    void write_pod(const T& value) {
        out_.write(reinterpret_cast<const char*>(&value), sizeof(T));
        if (!out_) {
            throw std::runtime_error("failed to write dump file");
        }
    }

    std::ofstream out_;
};

std::uint64_t parse_u64(std::string_view text, std::string_view flag) {
    if (text.empty()) {
        throw std::runtime_error(std::string(flag) + " requires a non-empty integer");
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

int parse_depth(std::string_view text) {
    const std::uint64_t value = parse_u64(text, "--max-depth");
    if (value > static_cast<std::uint64_t>(connect4::kMaxMoves)) {
        throw std::runtime_error("--max-depth must be between 0 and 42");
    }
    return static_cast<int>(value);
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

        if (arg == "--dump") {
            options.dump_path = std::string(require_value(arg));
        } else if (arg == "--max-depth") {
            options.max_depth = parse_depth(require_value(arg));
        } else if (arg == "--limit-states") {
            options.limit_states = parse_u64(require_value(arg), arg);
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: exhaustive [--max-depth N] [--limit-states N] [--dump FILE]\n"
                << "\n"
                << "DFS explores legal 6x7 Connect Four game states from an empty board.\n"
                << "--max-depth N     Stop expanding at depth N, 0 <= N <= 42. Default: 42.\n"
                << "--limit-states N  Stop after visiting N states. 0 means unlimited.\n"
                << "--dump FILE       Write each visited state's black/white masks, side, and outcome.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    return options;
}

Outcome classify_terminal(const connect4::Board& board) {
    switch (connect4::terminal_outcome(board)) {
    case connect4::Outcome::BlackWin:
        return Outcome::BlackWin;
    case connect4::Outcome::WhiteWin:
        return Outcome::WhiteWin;
    case connect4::Outcome::Draw:
        return Outcome::Draw;
    case connect4::Outcome::Unknown:
    default:
        return Outcome::Ongoing;
    }
}

void count_terminal(Counts& counts, Outcome outcome) {
    switch (outcome) {
    case Outcome::BlackWin:
        ++counts.black_wins;
        break;
    case Outcome::WhiteWin:
        ++counts.white_wins;
        break;
    case Outcome::Draw:
        ++counts.draws;
        break;
    case Outcome::Ongoing:
        break;
    }
}

bool limit_reached(const Counts& counts, const Options& options) {
    return options.limit_states != 0 && counts.visited >= options.limit_states;
}

void dfs(
    const connect4::Board& board,
    int depth,
    const Options& options,
    Counts& counts,
    DumpWriter* dump) {
    if (limit_reached(counts, options)) {
        counts.hit_limit = true;
        return;
    }

    const Outcome outcome = classify_terminal(board);
    ++counts.visited;

    if (dump != nullptr) {
        dump->write(
            board.black,
            board.white,
            board.side_to_move == connect4::Player::Black ? std::uint8_t{0} : std::uint8_t{1},
            outcome);
    }

    if (outcome != Outcome::Ongoing) {
        count_terminal(counts, outcome);
        return;
    }

    if (depth >= options.max_depth) {
        return;
    }

    for (int col = 0; col < connect4::kCols; ++col) {
        if (!connect4::can_play(board, col)) {
            continue;
        }

        const connect4::MoveResult next = connect4::play(board, col);
        if (next.legal) {
            dfs(next.board, depth + 1, options, counts, dump);
        }

        if (counts.hit_limit) {
            return;
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        std::optional<DumpWriter> dump;
        if (options.dump_path.has_value()) {
            dump.emplace(*options.dump_path);
        }

        Counts counts;
        const connect4::Board root{};
        dfs(root, 0, options, counts, dump.has_value() ? &*dump : nullptr);

        std::cout << "visited " << counts.visited << '\n';
        std::cout << "black_win " << counts.black_wins << '\n';
        std::cout << "white_win " << counts.white_wins << '\n';
        std::cout << "draw " << counts.draws << '\n';
        std::cout << "truncated " << (counts.hit_limit ? "yes" : "no") << '\n';

        if (options.max_depth < connect4::kMaxMoves) {
            std::cout << "note max-depth stopped expansion before full terminal coverage\n";
        }
        if (counts.hit_limit) {
            std::cout << "note limit-states stopped traversal early\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        std::cerr << "try --help\n";
        return 1;
    }
}
