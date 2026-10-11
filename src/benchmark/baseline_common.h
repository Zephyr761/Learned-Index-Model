#pragma once
#include <index_contract.h>
#include <limits>
#include "key_bytes.h"

namespace blocklearn::bench {
using namespace string_index_bench::v1;
inline void no_parameters(const Parameters& p) {
  if (!p.empty()) throw std::invalid_argument("baseline accepts no parameters");
}
inline void validate_rows(const std::vector<Row>& rows) {
  for (std::size_t i = 1; i < rows.size(); ++i)
    if (!less(rows[i - 1].key, rows[i].key))
      throw std::invalid_argument("bulk rows must be strictly byte-sorted and unique");
}
inline std::size_t string_heap(const std::string& s) {
  const auto data = reinterpret_cast<std::uintptr_t>(s.data());
  const auto object = reinterpret_cast<std::uintptr_t>(&s);
  return data >= object && data < object + sizeof(s) ? 0 : s.capacity() + 1;
}
inline std::size_t input_bytes(const std::vector<Row>& rows) {
  auto bytes = rows.capacity() * sizeof(Row);
  for (const auto& r : rows) bytes += string_heap(r.key);
  return bytes;
}
inline void cstring_key(std::string_view key, std::size_t max_bytes) {
  if (key.size() > max_bytes || key.find('\0') != std::string_view::npos)
    throw std::invalid_argument("native C-string baseline key domain exceeded");
}
}  // namespace blocklearn::bench
