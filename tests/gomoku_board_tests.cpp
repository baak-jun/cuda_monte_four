#include <gomoku/bitboard.hpp>
#include <gomoku/board.hpp>

#include <cstdint>
#include <iostream>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

gomoku::Board horizontal_board(int stones) {
    gomoku::Board board;
    for (int col = 0; col < stones; ++col) {
        board.cells[col] = static_cast<std::uint8_t>(gomoku::Player::Black);
    }
    board.moves = static_cast<std::uint16_t>(stones);
    board.last_move = static_cast<std::int16_t>(stones - 1);
    board.side_to_move = gomoku::Player::White;
    return board;
}

gomoku::GomokuBits horizontal_bits(int stones) {
    gomoku::GomokuBits bits{};
    for (int col = 0; col < stones; ++col) {
        gomoku::set_bit(bits, 0, col);
    }
    return bits;
}

std::uint32_t next_random(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

void verify_winning_cell_masks() {
    std::uint32_t random_state = 0xC0FFEEU;
    for (int sample = 0; sample < 500; ++sample) {
        gomoku::GomokuBits stones{};
        gomoku::GomokuBits empty{};

        for (int row = 0; row < 15; ++row) {
            for (int col = 0; col < 15; ++col) {
                if (next_random(random_state) % 5U == 0U) {
                    gomoku::set_bit(stones, row, col);
                } else {
                    gomoku::set_bit(empty, row, col);
                }
            }
        }

        gomoku::GomokuBits actual{};
        gomoku::get_winning_cells(stones, empty, actual);

        for (int row = 0; row < 15; ++row) {
            for (int col = 0; col < 15; ++col) {
                const int index = row * 15 + col;
                bool expected = false;
                if (gomoku::get_bit(empty, row, col)) {
                    gomoku::GomokuBits with_move = stones;
                    gomoku::set_bit(with_move, row, col);
                    expected = gomoku::has_exact_five(with_move, index);
                }
                if (gomoku::get_bit(actual, row, col) != expected) {
                    std::cerr << "FAIL: winning-cell bitmask mismatch at sample "
                              << sample << ", row " << row << ", col " << col << '\n';
                    ++failures;
                    return;
                }
            }
        }
    }
}

} // namespace

int main() {
    const gomoku::Board five = horizontal_board(5);
    expect(gomoku::has_five(five, 4), "exactly five stones should win");
    expect(gomoku::check_outcome(five) == gomoku::Outcome::BlackWin, "exact five outcome should name black");

    const gomoku::Board six = horizontal_board(6);
    expect(!gomoku::has_five(six, 5), "an edge extension to six must not count as exact five");
    expect(!gomoku::has_five(six, 2), "a last move inside an overline must not count as exact five");

    const gomoku::GomokuBits five_bits = horizontal_bits(5);
    expect(gomoku::has_exact_five(five_bits, 4), "bitboard exact five should win");

    const gomoku::GomokuBits six_bits = horizontal_bits(6);
    expect(!gomoku::has_exact_five(six_bits, 5), "bitboard edge overline must not win");
    expect(!gomoku::has_exact_five(six_bits, 2), "bitboard interior overline must not win");

    gomoku::GomokuBits four_with_gap{};
    gomoku::set_bit(four_with_gap, 0, 0);
    gomoku::set_bit(four_with_gap, 0, 1);
    gomoku::set_bit(four_with_gap, 0, 2);
    gomoku::set_bit(four_with_gap, 0, 3);
    gomoku::GomokuBits empty{};
    gomoku::set_bit(empty, 0, 4);
    gomoku::GomokuBits wins{};
    gomoku::get_winning_cells(four_with_gap, empty, wins);
    expect(gomoku::get_bit(wins, 0, 4), "four stones should expose the exact-five winning cell");

    gomoku::set_bit(four_with_gap, 0, 5);
    gomoku::get_winning_cells(four_with_gap, empty, wins);
    expect(!gomoku::get_bit(wins, 0, 4), "a move that creates six must not be marked winning");

    verify_winning_cell_masks();

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "gomoku board tests passed\n";
    return 0;
}
