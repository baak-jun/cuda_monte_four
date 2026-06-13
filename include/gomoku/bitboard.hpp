#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define GOMOKU_HOST_DEVICE __host__ __device__
#else
#define GOMOKU_HOST_DEVICE
#endif

namespace gomoku {

struct GomokuBits {
    std::uint16_t rows[15];
};

GOMOKU_HOST_DEVICE inline void clear_bits(GomokuBits& b) {
    for (int r = 0; r < 15; ++r) b.rows[r] = 0;
}

GOMOKU_HOST_DEVICE inline void set_bit(GomokuBits& b, int r, int c) {
    b.rows[r] |= (1 << c);
}

GOMOKU_HOST_DEVICE inline bool get_bit(const GomokuBits& b, int r, int c) {
    return (b.rows[r] & (1 << c)) != 0;
}

GOMOKU_HOST_DEVICE inline int find_first_bit(const GomokuBits& b) {
    for (int r = 0; r < 15; ++r) {
        if (b.rows[r] != 0) {
            // Find the index of the lowest set bit (1-indexed from __ffs)
            int c = 0;
            std::uint16_t row = b.rows[r];
            while ((row & 1) == 0) {
                row >>= 1;
                c++;
            }
            return r * 15 + c;
        }
    }
    return -1;
}

GOMOKU_HOST_DEVICE inline void get_winning_cells(const GomokuBits& stones, const GomokuBits& empty, GomokuBits& wins) {
    clear_bits(wins);

    // Horizontal
    for (int r = 0; r < 15; ++r) {
        std::uint16_t S = stones.rows[r];
        std::uint16_t w0 = (S >> 1) & (S >> 2) & (S >> 3) & (S >> 4);
        std::uint16_t w1 = S & (S >> 2) & (S >> 3) & (S >> 4);
        std::uint16_t w2 = S & (S >> 1) & (S >> 3) & (S >> 4);
        std::uint16_t w3 = S & (S >> 1) & (S >> 2) & (S >> 4);
        std::uint16_t w4 = S & (S >> 1) & (S >> 2) & (S >> 3);
        wins.rows[r] |= (w0 | (w1 << 1) | (w2 << 2) | (w3 << 3) | (w4 << 4)) & empty.rows[r];
    }

    // Vertical
    for (int r = 0; r < 15 - 4; ++r) {
        std::uint16_t S0 = stones.rows[r], S1 = stones.rows[r+1], S2 = stones.rows[r+2], S3 = stones.rows[r+3], S4 = stones.rows[r+4];
        wins.rows[r]   |= (S1 & S2 & S3 & S4) & empty.rows[r];
        wins.rows[r+1] |= (S0 & S2 & S3 & S4) & empty.rows[r+1];
        wins.rows[r+2] |= (S0 & S1 & S3 & S4) & empty.rows[r+2];
        wins.rows[r+3] |= (S0 & S1 & S2 & S4) & empty.rows[r+3];
        wins.rows[r+4] |= (S0 & S1 & S2 & S3) & empty.rows[r+4];
    }

    // Diagonal Down-Right
    for (int r = 0; r < 15 - 4; ++r) {
        std::uint16_t S0 = stones.rows[r], S1 = stones.rows[r+1]>>1, S2 = stones.rows[r+2]>>2, S3 = stones.rows[r+3]>>3, S4 = stones.rows[r+4]>>4;
        std::uint16_t w0 = S1 & S2 & S3 & S4;
        std::uint16_t w1 = S0 & S2 & S3 & S4;
        std::uint16_t w2 = S0 & S1 & S3 & S4;
        std::uint16_t w3 = S0 & S1 & S2 & S4;
        std::uint16_t w4 = S0 & S1 & S2 & S3;
        wins.rows[r]   |= w0 & empty.rows[r];
        wins.rows[r+1] |= (w1 << 1) & empty.rows[r+1];
        wins.rows[r+2] |= (w2 << 2) & empty.rows[r+2];
        wins.rows[r+3] |= (w3 << 3) & empty.rows[r+3];
        wins.rows[r+4] |= (w4 << 4) & empty.rows[r+4];
    }

    // Diagonal Down-Left
    for (int r = 0; r < 15 - 4; ++r) {
        std::uint16_t S0 = stones.rows[r], S1 = stones.rows[r+1]<<1, S2 = stones.rows[r+2]<<2, S3 = stones.rows[r+3]<<3, S4 = stones.rows[r+4]<<4;
        std::uint16_t w0 = S1 & S2 & S3 & S4;
        std::uint16_t w1 = S0 & S2 & S3 & S4;
        std::uint16_t w2 = S0 & S1 & S3 & S4;
        std::uint16_t w3 = S0 & S1 & S2 & S4;
        std::uint16_t w4 = S0 & S1 & S2 & S3;
        wins.rows[r]   |= w0 & empty.rows[r];
        wins.rows[r+1] |= (w1 >> 1) & empty.rows[r+1];
        wins.rows[r+2] |= (w2 >> 2) & empty.rows[r+2];
        wins.rows[r+3] |= (w3 >> 3) & empty.rows[r+3];
        wins.rows[r+4] |= (w4 >> 4) & empty.rows[r+4];
    }
}

GOMOKU_HOST_DEVICE inline void get_open_threes(const GomokuBits& stones, const GomokuBits& empty, GomokuBits& out) {
    clear_bits(out);

    // Horizontal: .XXX.
    for (int r = 0; r < 15; ++r) {
        std::uint16_t S = stones.rows[r];
        std::uint16_t E = empty.rows[r];
        std::uint16_t p = E & (S >> 1) & (S >> 2) & (S >> 3) & (E >> 4);
        out.rows[r] |= p | (p << 4);
    }

    // Vertical: .XXX.
    for (int r = 0; r < 15 - 4; ++r) {
        std::uint16_t E0 = empty.rows[r];
        std::uint16_t S1 = stones.rows[r+1];
        std::uint16_t S2 = stones.rows[r+2];
        std::uint16_t S3 = stones.rows[r+3];
        std::uint16_t E4 = empty.rows[r+4];
        std::uint16_t p = E0 & S1 & S2 & S3 & E4;
        out.rows[r]   |= p;
        out.rows[r+4] |= p;
    }

    // Diagonal Down-Right: .XXX.
    for (int r = 0; r < 15 - 4; ++r) {
        std::uint16_t E0 = empty.rows[r];
        std::uint16_t S1 = stones.rows[r+1] >> 1;
        std::uint16_t S2 = stones.rows[r+2] >> 2;
        std::uint16_t S3 = stones.rows[r+3] >> 3;
        std::uint16_t E4 = empty.rows[r+4] >> 4;
        std::uint16_t p = E0 & S1 & S2 & S3 & E4;
        out.rows[r]   |= p;
        out.rows[r+4] |= (p << 4);
    }

    // Diagonal Down-Left: .XXX.
    for (int r = 0; r < 15 - 4; ++r) {
        std::uint16_t E0 = empty.rows[r];
        std::uint16_t S1 = stones.rows[r+1] << 1;
        std::uint16_t S2 = stones.rows[r+2] << 2;
        std::uint16_t S3 = stones.rows[r+3] << 3;
        std::uint16_t E4 = empty.rows[r+4] << 4;
        std::uint16_t p = E0 & S1 & S2 & S3 & E4;
        out.rows[r]   |= p;
        out.rows[r+4] |= (p >> 4);
    }
}

} // namespace gomoku
