import sys

def make_mask(cols_to_keep):
    # cols_to_keep is a list of column indices (0-14) that we want to keep
    # Return 4 uint64_t values
    bits = [0] * 256
    for r in range(15):
        for c in cols_to_keep:
            bits[r * 15 + c] = 1
            
    words = [0, 0, 0, 0]
    for i in range(4):
        val = 0
        for j in range(64):
            if i * 64 + j < 225:
                val |= (bits[i * 64 + j] << j)
        words[i] = val
    return words

print("MASK_NOT_COL_14 (can shift right 1):", [hex(x) for x in make_mask(range(14))])
print("MASK_NOT_COL_0 (can shift left 1):", [hex(x) for x in make_mask(range(1, 15))])
