#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace blocklearn {

inline int compare(std::string_view a, std::string_view b) {
  const auto n = std::min(a.size(), b.size());
  const int c = n ? std::memcmp(a.data(), b.data(), n) : 0;
  return c ? c : (a.size() > b.size()) - (a.size() < b.size());
}
inline bool less(std::string_view a, std::string_view b) {
  return compare(a, b) < 0;
}
inline std::size_t common_prefix(std::string_view a, std::string_view b) {
  std::size_t i = 0;
  while (i < std::min(a.size(), b.size()) && a[i] == b[i]) ++i;
  return i;
}
// EOS=0; byte values become 1..256. Seven symbols fit in 63 bits.
// Truncation is non-decreasing, not injective. Duplicate coordinates must
// remain together when fitting segments, and are covered by the certificate.
inline std::uint64_t encode(std::string_view key, std::size_t prefix,
                            unsigned bytes) {
  std::uint64_t x = 0;
  for (unsigned i = 0; i < bytes; ++i) {
    const auto at = prefix + i;
    x = (x << 9) | (at < key.size()
        ? std::uint64_t(static_cast<unsigned char>(key[at])) + 1 : 0);
  }
  return x;
}
inline std::uint16_t hash_tag(std::string_view key, std::size_t prefix) {
  std::uint64_t h = 14695981039346656037ULL;
  for (std::size_t i = prefix; i < key.size(); ++i)
    h = (h ^ static_cast<unsigned char>(key[i])) * 1099511628211ULL;
  h ^= key.size();
  h ^= h >> 32;
  h ^= h >> 16;
  return static_cast<std::uint16_t>(h);
}

}  // namespace blocklearn
