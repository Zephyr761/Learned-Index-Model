#pragma once

#include "key_bytes.h"
#include <index_contract.h>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace blocklearn::bench {

// Read the teammate's frozen files, not a newly generated "similar" workload.
// Retain only sampled query strings, so profiling does not materialize millions
// of extra queries. All lengths, flags, ordering and trailing bytes are checked.
class FrozenInput {
 public:
  FrozenInput(const std::string& path, std::string_view magic, std::size_t minimum)
      : in_(path, std::ios::binary) {
    if (!in_) fail("cannot open interchange");
    in_.seekg(0, std::ios::end); size_ = in_.tellg(); in_.seekg(0);
    char actual[8];
    if (!in_.read(actual, 8) || std::string_view(actual, 8) != magic)
      fail("wrong interchange magic");
    count = u64();
    if (size_ < 16 || count > (std::uint64_t(size_) - 16) / minimum)
      fail("invalid record count");
  }
  std::uint64_t u64() {
    unsigned char bytes[8];
    if (!in_.read(reinterpret_cast<char*>(bytes), 8)) fail("truncated integer");
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= std::uint64_t(bytes[i]) << (8 * i);
    return value;
  }
  bool flag() {
    const auto value = in_.get();
    if (value != 0 && value != 1) fail("invalid present flag");
    return value != 0;
  }
  std::string key(std::uint64_t length, std::uint64_t trailing, bool keep = true) {
    const auto pos = in_.tellg();
    if (pos < 0 || pos > size_) fail("invalid stream position");
    const auto left = std::uint64_t(size_ - pos);
    if (trailing > left || length > left - trailing) fail("invalid key length");
    std::string result;
    if (keep) {
      if (length > result.max_size()) fail("key too large");
      result.resize(static_cast<std::size_t>(length));
      if (length && !in_.read(result.data(), static_cast<std::streamsize>(length)))
        fail("truncated key");
    } else if (!in_.seekg(static_cast<std::streamoff>(length), std::ios::cur)) {
      fail("cannot skip query key");
    }
    return result;
  }
  void finish() {
    if (in_.peek() != std::char_traits<char>::eof() || in_.bad())
      fail("trailing bytes or read failure");
  }
  std::uint64_t count = 0;
 private:
  [[noreturn]] static void fail(const char* message) { throw std::runtime_error(message); }
  std::ifstream in_;
  std::streamoff size_ = 0;
};

inline std::vector<string_index_bench::v1::Row> read_profile_rows(const std::string& path) {
  FrozenInput input(path, std::string_view("SIBROW1\0", 8), 16);
  std::vector<string_index_bench::v1::Row> rows;
  if (input.count > rows.max_size()) throw std::runtime_error("too many rows");
  rows.reserve(static_cast<std::size_t>(input.count));
  for (std::uint64_t i = 0; i < input.count; ++i) {
    const auto length = input.u64();
    auto key = input.key(length, 8);
    const auto value = input.u64();
    if (!rows.empty() && !less(rows.back().key, key))
      throw std::runtime_error("rows not strictly unsigned-byte sorted/unique");
    rows.push_back({std::move(key), value});
  }
  input.finish();
  return rows;
}
struct ProfileQuery {
  std::uint64_t position;
  std::string key;
  bool present;
  string_index_bench::v1::Value value;
};
struct ProfileQueries { std::uint64_t total; std::vector<ProfileQuery> sample; };
inline ProfileQueries read_profile_queries(const std::string& path, std::size_t maximum) {
  if (!maximum) throw std::invalid_argument("profile sample count must be positive");
  FrozenInput input(path, std::string_view("SIBQRY1\0", 8), 17);
  ProfileQueries result{input.count, {}};
  const auto samples = std::min<std::uint64_t>(input.count, maximum);
  if (!samples) { input.finish(); return result; }
  if (samples > result.sample.max_size()) throw std::runtime_error("too many query samples");
  result.sample.reserve(static_cast<std::size_t>(samples));
  const auto step = input.count / samples, remainder = input.count % samples;
  std::uint64_t next = 0, carry = 0;
  for (std::uint64_t i = 0; i < input.count; ++i) {
    const auto present = input.flag();
    const auto value = input.u64();
    const auto length = input.u64();
    const bool keep = result.sample.size() < samples && i == next;
    auto key = input.key(length, 0, keep);
    if (keep) {
      result.sample.push_back({i, std::move(key), present, value});
      next += step; carry += remainder;
      if (carry >= samples) { ++next; carry -= samples; }
    }
  }
  input.finish();
  return result;
}

}  // namespace blocklearn::bench
