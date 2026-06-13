def gen_code():
    code = ""
    # Horizontal
    code += "    // Horizontal\n"
    code += "    for (int r = 0; r < 15; ++r) {\n"
    code += "        uint16_t S = stones.rows[r];\n"
    code += "        uint16_t w0 = (S >> 1) & (S >> 2) & (S >> 3) & (S >> 4);\n"
    code += "        uint16_t w1 = S & (S >> 2) & (S >> 3) & (S >> 4);\n"
    code += "        uint16_t w2 = S & (S >> 1) & (S >> 3) & (S >> 4);\n"
    code += "        uint16_t w3 = S & (S >> 1) & (S >> 2) & (S >> 4);\n"
    code += "        uint16_t w4 = S & (S >> 1) & (S >> 2) & (S >> 3);\n"
    code += "        wins.rows[r] |= (w0 | (w1 << 1) | (w2 << 2) | (w3 << 3) | (w4 << 4)) & empty.rows[r];\n"
    code += "    }\n"
    
    # Vertical
    code += "    // Vertical\n"
    code += "    for (int r = 0; r < 15 - 4; ++r) {\n"
    code += "        uint16_t S0 = stones.rows[r], S1 = stones.rows[r+1], S2 = stones.rows[r+2], S3 = stones.rows[r+3], S4 = stones.rows[r+4];\n"
    code += "        wins.rows[r]   |= (S1 & S2 & S3 & S4) & empty.rows[r];\n"
    code += "        wins.rows[r+1] |= (S0 & S2 & S3 & S4) & empty.rows[r+1];\n"
    code += "        wins.rows[r+2] |= (S0 & S1 & S3 & S4) & empty.rows[r+2];\n"
    code += "        wins.rows[r+3] |= (S0 & S1 & S2 & S4) & empty.rows[r+3];\n"
    code += "        wins.rows[r+4] |= (S0 & S1 & S2 & S3) & empty.rows[r+4];\n"
    code += "    }\n"
    
    # Diagonal Down-Right (shifting right by 1 each step)
    code += "    // Diagonal Down-Right\n"
    code += "    for (int r = 0; r < 15 - 4; ++r) {\n"
    code += "        uint16_t S0 = stones.rows[r], S1 = stones.rows[r+1]>>1, S2 = stones.rows[r+2]>>2, S3 = stones.rows[r+3]>>3, S4 = stones.rows[r+4]>>4;\n"
    code += "        uint16_t w0 = S1 & S2 & S3 & S4;\n"
    code += "        uint16_t w1 = S0 & S2 & S3 & S4;\n"
    code += "        uint16_t w2 = S0 & S1 & S3 & S4;\n"
    code += "        uint16_t w3 = S0 & S1 & S2 & S4;\n"
    code += "        uint16_t w4 = S0 & S1 & S2 & S3;\n"
    code += "        wins.rows[r]   |= w0 & empty.rows[r];\n"
    code += "        wins.rows[r+1] |= (w1 << 1) & empty.rows[r+1];\n"
    code += "        wins.rows[r+2] |= (w2 << 2) & empty.rows[r+2];\n"
    code += "        wins.rows[r+3] |= (w3 << 3) & empty.rows[r+3];\n"
    code += "        wins.rows[r+4] |= (w4 << 4) & empty.rows[r+4];\n"
    code += "    }\n"
    
    # Diagonal Down-Left (shifting left by 1 each step)
    code += "    // Diagonal Down-Left\n"
    code += "    for (int r = 0; r < 15 - 4; ++r) {\n"
    code += "        uint16_t S0 = stones.rows[r], S1 = stones.rows[r+1]<<1, S2 = stones.rows[r+2]<<2, S3 = stones.rows[r+3]<<3, S4 = stones.rows[r+4]<<4;\n"
    code += "        uint16_t w0 = S1 & S2 & S3 & S4;\n"
    code += "        uint16_t w1 = S0 & S2 & S3 & S4;\n"
    code += "        uint16_t w2 = S0 & S1 & S3 & S4;\n"
    code += "        uint16_t w3 = S0 & S1 & S2 & S4;\n"
    code += "        uint16_t w4 = S0 & S1 & S2 & S3;\n"
    code += "        wins.rows[r]   |= w0 & empty.rows[r];\n"
    code += "        wins.rows[r+1] |= (w1 >> 1) & empty.rows[r+1];\n"
    code += "        wins.rows[r+2] |= (w2 >> 2) & empty.rows[r+2];\n"
    code += "        wins.rows[r+3] |= (w3 >> 3) & empty.rows[r+3];\n"
    code += "        wins.rows[r+4] |= (w4 >> 4) & empty.rows[r+4];\n"
    code += "    }\n"
    
    print(code)

gen_code()
