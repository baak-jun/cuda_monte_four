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

GOMOKU_HOST_DEVICE inline bool has_exact_five(const GomokuBits& stones, int last_idx) {
    if (last_idx < 0 || last_idx >= 225) return false;

    const int r = last_idx / 15;
    const int c = last_idx % 15;
    if (!get_bit(stones, r, c)) return false;

    constexpr int dr[4] = {0, 1, 1, -1};
    constexpr int dc[4] = {1, 0, 1, 1};

    for (int d = 0; d < 4; ++d) {
        int count = 1;

        for (int step = 1; step < 15; ++step) {
            const int nr = r + dr[d] * step;
            const int nc = c + dc[d] * step;
            if (nr < 0 || nr >= 15 || nc < 0 || nc >= 15 || !get_bit(stones, nr, nc)) break;
            ++count;
        }

        for (int step = 1; step < 15; ++step) {
            const int nr = r - dr[d] * step;
            const int nc = c - dc[d] * step;
            if (nr < 0 || nr >= 15 || nc < 0 || nc >= 15 || !get_bit(stones, nr, nc)) break;
            ++count;
        }

        if (count == 5) return true;
    }

    return false;
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
        
        std::uint16_t mask = ~(S << 1) & ~(S >> 5) & 0x7FFF;
        w0 &= mask;
        w1 &= mask;
        w2 &= mask;
        w3 &= mask;
        w4 &= mask;
        
        wins.rows[r] |= (w0 | (w1 << 1) | (w2 << 2) | (w3 << 3) | (w4 << 4)) & empty.rows[r];
    }

    // Vertical
    for (int r = 0; r < 15 - 4; ++r) {
        std::uint16_t S0 = stones.rows[r], S1 = stones.rows[r+1], S2 = stones.rows[r+2], S3 = stones.rows[r+3], S4 = stones.rows[r+4];
        
        std::uint16_t mask = 0x7FFF;
        if (r > 0) mask &= ~stones.rows[r-1];
        if (r < 10) mask &= ~stones.rows[r+5];
        
        wins.rows[r]   |= (S1 & S2 & S3 & S4) & empty.rows[r] & mask;
        wins.rows[r+1] |= (S0 & S2 & S3 & S4) & empty.rows[r+1] & mask;
        wins.rows[r+2] |= (S0 & S1 & S3 & S4) & empty.rows[r+2] & mask;
        wins.rows[r+3] |= (S0 & S1 & S2 & S4) & empty.rows[r+3] & mask;
        wins.rows[r+4] |= (S0 & S1 & S2 & S3) & empty.rows[r+4] & mask;
    }

    // Diagonal Down-Right
    for (int r = 0; r < 15 - 4; ++r) {
        std::uint16_t S0 = stones.rows[r], S1 = stones.rows[r+1]>>1, S2 = stones.rows[r+2]>>2, S3 = stones.rows[r+3]>>3, S4 = stones.rows[r+4]>>4;
        
        std::uint16_t mask = 0x7FFF;
        if (r > 0) mask &= ~(stones.rows[r-1] << 1);
        if (r < 10) mask &= ~(stones.rows[r+5] >> 5);
        mask &= 0x7FFF;
        
        std::uint16_t w0 = S1 & S2 & S3 & S4;
        std::uint16_t w1 = S0 & S2 & S3 & S4;
        std::uint16_t w2 = S0 & S1 & S3 & S4;
        std::uint16_t w3 = S0 & S1 & S2 & S4;
        std::uint16_t w4 = S0 & S1 & S2 & S3;
        
        wins.rows[r]   |= w0 & empty.rows[r] & mask;
        wins.rows[r+1] |= (w1 << 1) & empty.rows[r+1] & (mask << 1) & 0x7FFF;
        wins.rows[r+2] |= (w2 << 2) & empty.rows[r+2] & (mask << 2) & 0x7FFF;
        wins.rows[r+3] |= (w3 << 3) & empty.rows[r+3] & (mask << 3) & 0x7FFF;
        wins.rows[r+4] |= (w4 << 4) & empty.rows[r+4] & (mask << 4) & 0x7FFF;
    }

    // Diagonal Down-Left
    for (int r = 0; r < 15 - 4; ++r) {
        std::uint16_t S0 = stones.rows[r], S1 = stones.rows[r+1]<<1, S2 = stones.rows[r+2]<<2, S3 = stones.rows[r+3]<<3, S4 = stones.rows[r+4]<<4;
        
        std::uint16_t mask = 0x7FFF;
        if (r > 0) mask &= ~(stones.rows[r-1] >> 1);
        if (r < 10) mask &= ~(stones.rows[r+5] << 5);
        mask &= 0x7FFF;
        
        std::uint16_t w0 = S1 & S2 & S3 & S4;
        std::uint16_t w1 = S0 & S2 & S3 & S4;
        std::uint16_t w2 = S0 & S1 & S3 & S4;
        std::uint16_t w3 = S0 & S1 & S2 & S4;
        std::uint16_t w4 = S0 & S1 & S2 & S3;
        
        wins.rows[r]   |= w0 & empty.rows[r] & mask;
        wins.rows[r+1] |= (w1 >> 1) & empty.rows[r+1] & (mask >> 1) & 0x7FFF;
        wins.rows[r+2] |= (w2 >> 2) & empty.rows[r+2] & (mask >> 2) & 0x7FFF;
        wins.rows[r+3] |= (w3 >> 3) & empty.rows[r+3] & (mask >> 3) & 0x7FFF;
        wins.rows[r+4] |= (w4 >> 4) & empty.rows[r+4] & (mask >> 4) & 0x7FFF;
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
