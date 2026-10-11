#pragma once
#include <cstddef>
#include <cstdint>
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace blocklearn {
// Return one bit per matching lane. Never load beyond count/array bounds.
inline std::uint32_t tag_mask(const std::uint16_t* tags, std::size_t count,
                              std::uint16_t wanted) {
#if defined(__AVX2__)
  if (count == 16) {
    const auto data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tags));
    const auto equal = _mm256_cmpeq_epi16(data, _mm256_set1_epi16(static_cast<short>(wanted)));
    auto m = std::uint32_t(_mm256_movemask_epi8(equal)) & 0x55555555U;
    m = (m | (m >> 1)) & 0x33333333U;
    m = (m | (m >> 2)) & 0x0f0f0f0fU;
    m = (m | (m >> 4)) & 0x00ff00ffU;
    return (m | (m >> 8)) & 0x0000ffffU;
  }
#elif defined(__aarch64__)
  if (count == 16) {
    const std::uint16_t weights[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    const auto w = vld1q_u16(weights), target = vdupq_n_u16(wanted);
    const auto low = vandq_u16(vceqq_u16(vld1q_u16(tags), target), w);
    const auto high = vandq_u16(vceqq_u16(vld1q_u16(tags + 8), target), w);
    return std::uint32_t(vaddvq_u16(low)) | (std::uint32_t(vaddvq_u16(high)) << 8);
  }
#endif
  std::uint32_t mask = 0;
  for (std::size_t i = 0; i < count; ++i) mask |= std::uint32_t(tags[i] == wanted) << i;
  return mask;
}
}  // namespace blocklearn
