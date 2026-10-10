/**
 * flute/dequant.cuh
 *
 * Streaming dequantization utilities for FLUTE-Extended.
 *
 * Palettization format (fixed by the FLUTE weight layout):
 *   W[n, k] = LUT[ n / group_size , index[n, k] ]
 *
 * Indices are packed 4-bit, uint8 storage, LSB-first:
 *   byte = indices[n, k/2]
 *   even-k index  =  byte & 0x0F
 *   odd-k  index  = (byte >> 4) & 0x0F
 *
 * LUT shape: [num_groups, 16], FP16.
 *
 * All routines in this header are device-only.
 */

#pragma once

#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace flute {

// Decode a single 4-bit index from a packed byte stream.
// `byte_ptr` points at the byte holding both indices for (k, k^1).
// `lane` is 0 for even k, 1 for odd k.
__device__ __forceinline__ uint8_t decode_nibble(const uint8_t* byte_ptr, int lane) {
    uint8_t b = byte_ptr[0];
    return (lane == 0) ? uint8_t(b & 0x0F) : uint8_t((b >> 4) & 0x0F);
}

// Decode a full uint8_t containing two nibbles into two FP16 values
// using a 16-entry FP16 LUT. Returns both values via pointers.
__device__ __forceinline__ void decode_pair(uint8_t packed,
    const __half* lut_row,   // 16 FP16 entries for this group
    __half& lo,              // index = packed & 0x0F
    __half& hi               // index = (packed >> 4) & 0x0F
) {
    lo = lut_row[packed & 0x0F];
    hi = lut_row[(packed >> 4) & 0x0F];
}

// Stride (in bytes) of one row of packed 4-bit indices for K columns.
__device__ __forceinline__ int indices_row_stride(int K) {
    return (K + 1) >> 1;   // ceil(K/2)
}

// Stride (in bytes) of one row of packed b-bit indices for K columns
// (idxN family, b in {1,2,3,4}). K is a multiple of 32 in every kernel
// path, so K*b is divisible by 8 and the stride is exact.
__device__ __forceinline__ int indices_row_stride_b(int K, int bits) {
    return (K * bits) >> 3;
}

// Number of groups along N for given N and group_size.
__device__ __forceinline__ int num_groups(int N, int group_size) {
    return (N + group_size - 1) / group_size;
}

// Given n (row of W) and group_size, return group id.
__device__ __forceinline__ int group_of(int n, int group_size) {
    return n / group_size;
}

// ---------------------------------------------------------------------------
// Sub-byte pair decode (idxN family: 1/2/3-bit, src/docs/QUANTIZATION.md
// section 4).
// The unit of consumption is the k-PAIR (k, k+1) with k even: two
// consecutive values that dequantize into one mma.m16n8k16 B-fragment
// u32 {W[k], W[k+1]} (low half = W[k]). In the LOGICAL LSB-first stream
// the pair occupies 2*b consecutive bits at bit offset k*b of the row,
// so for b in {1,2} both fields share one byte (k even => k*b mod 8 in
// {0,4}), and for b == 3 the 6-bit field may span a byte boundary (the
// little-endian two-byte window below covers every legal offset; the
// field never crosses the row end, so the window never reads past the
// row).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void decode_pair_b1(const uint8_t* row, int k, uint8_t& v0, uint8_t& v1
) {
    const uint8_t b = row[k >> 3];
    const int off = k & 7;               // k even => off in {0,2,4,6}
    v0 = uint8_t((b >> off) & 0x1);
    v1 = uint8_t((b >> (off + 1)) & 0x1);
}

__device__ __forceinline__ void decode_pair_b2(const uint8_t* row, int k, uint8_t& v0, uint8_t& v1
) {
    const uint8_t b = row[k >> 2];
    const int off = 2 * (k & 3);         // k even => off in {0,2,4,6}
    v0 = uint8_t((b >> off) & 0x3);
    v1 = uint8_t((b >> (off + 2)) & 0x3);
}

__device__ __forceinline__ void decode_pair_b3(const uint8_t* row, int k, uint8_t& v0, uint8_t& v1
) {
    const int bit = 3 * k;               // k even => bit mod 8 in {0,6,4,2}
    const int byte = bit >> 3;
    const int off = bit & 7;
    // 6-bit field; spans into byte+1 only when off+6 > 8 (off in {4,6}),
    // and a spanning field ends strictly inside the row, so byte+1 is
    // always in range. The non-spanning case (off in {0,2}) never reads
    // byte+1 — the last pair of a row lands there and must not read
    // past the row end.
    uint32_t w = uint32_t(row[byte]);
    if (off > 2) w |= uint32_t(row[byte + 1]) << 8;
    const uint32_t field = (w >> off) & 0x3Fu;
    v0 = uint8_t(field & 0x7);
    v1 = uint8_t((field >> 3) & 0x7);
}

}  // namespace flute
