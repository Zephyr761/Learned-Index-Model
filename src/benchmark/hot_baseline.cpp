#include "factories.h"
#include "baseline_common.h"
#include <array>
#include <deque>
#include <cstdlib>

#ifdef BLOCKLEARN_NATIVE_HOT
#if defined(__APPLE__)
#define __bswap_64 __builtin_bswap64
#define __bswap_32 __builtin_bswap32
#define __bswap_16 __builtin_bswap16
#endif
#include <hot/singlethreaded/HOTSingleThreaded.hpp>
#include <idx/contenthelpers/PairPointerKeyExtractor.hpp>
#endif

namespace {
using namespace string_index_bench::v1;
using namespace blocklearn::bench;
#ifdef BLOCKLEARN_NATIVE_HOT
class HotIndex final : public Index {
  using Record = std::pair<const char*, Value>;
  using Trie = hot::singlethreaded::HOTSingleThreaded<Record*, idx::contenthelpers::PairPointerKeyExtractor>;
  // Trie dies before its referenced records/strings.
  const std::vector<Row>* input_ = nullptr;
  std::deque<Row> owned_;
  std::deque<Record> records_;
  mutable Trie trie_;
  bool loaded_ = false;
  void ready() const { if (!loaded_) throw std::logic_error("bulk_load required"); }
  static std::array<char, 256> query(std::string_view key) {
    cstring_key(key, 254);
    std::array<char, 256> k{};
    if (!key.empty()) std::memcpy(k.data(), key.data(), key.size());
    return k;
  }
 public:
  Descriptor descriptor() const override {
    Descriptor d; d.implementation = "hot-single-threaded";
    d.implementation_version = "96bf6fb7103b27e50e16a6026db8974c090ee84a";
    d.adapter_version = "blocklearn-baselines-v1";
    auto& c = d.capabilities;
    c.keys.allows_nul = false; c.keys.max_bytes = 254;
    c.bulk_load = Support::adapted;
    c.find = c.insert = c.scan = Support::native; c.range_sum = Support::adapted;
    d.adaptation_notes = "Unmodified speedskater/HOT single-threaded C-string trie. "
        "Bulkload uses repeated insert. Explicit 254-byte limit preserves a terminal "
        "NUL within HOT's 255-byte representation. Bounded query copy and native "
        "padding costs are timed. Requires x86-64 AVX2/BMI/BMI2/LZCNT/POPCNT. "
        "range_sum is native lower_bound + iteration.";
    return d;
  }
  void bulk_load(const std::vector<Row>& rows) override {
    if (loaded_) throw std::logic_error("bulk_load called twice");
    validate_rows(rows);
    for (const auto& r : rows) cstring_key(r.key, 254);
    input_ = &rows;
    for (const auto& r : rows) {
      records_.emplace_back(r.key.c_str(), r.value); trie_.insert(&records_.back());
    }
    loaded_ = true;
  }
  std::optional<Value> find(std::string_view key) const override {
    ready(); auto k = query(key); const char* pointer = k.data();
    const auto result = trie_.lookup(pointer);
    return result.mIsValid ? std::optional<Value>(result.mValue->second) : std::nullopt;
  }
  InsertResult insert(std::string_view key, Value value) override {
    ready(); cstring_key(key, 254);
    owned_.push_back({std::string(key), value});
    records_.emplace_back(owned_.back().key.c_str(), value);
    if (!trie_.insert(&records_.back())) {
      records_.pop_back(); owned_.pop_back(); return InsertResult::already_exists;
    }
    return InsertResult::inserted;
  }
  UpdateResult update(std::string_view, Value) override { throw UnsupportedOperation("update unsupported"); }
  EraseResult erase(std::string_view) override { throw UnsupportedOperation("erase unsupported"); }
  std::vector<Row> scan(std::string_view lower, std::size_t limit) const override {
    ready(); auto k = query(lower); const char* pointer = k.data();
    std::vector<Row> result;
    for (auto it = trie_.lower_bound(pointer); it != trie_.end() && result.size() < limit; ++it)
      result.push_back({(*it)->first, (*it)->second});
    return result;
  }
  Value range_sum(std::string_view lower, std::string_view upper) const override {
    ready(); cstring_key(upper, 254);
    if (blocklearn::less(upper, lower)) return 0;
    auto k = query(lower); const char* pointer = k.data();
    Value total = 0;
    for (auto it = trie_.lower_bound(pointer); it != trie_.end(); ++it) {
      if (blocklearn::less(upper, (*it)->first)) break;
      total += (*it)->second;
    }
    return total;
  }
  MemoryUsage memory_usage() const override {
    ready();
    std::size_t adapter = sizeof(*this) + records_.size() * sizeof(Record) + owned_.size() * sizeof(Row);
    for (const auto& r : owned_) adapter += string_heap(r.key);
    const std::size_t tree_bytes = trie_.isEmpty() ? 0 : trie_.getStatistics().first;
    return {tree_bytes, adapter, input_bytes(*input_),
      "HOT native live-node statistics; adapter record/string estimate; borrowed "
      "input retained in full. Excludes HOT free-list pool, allocator/deque overhead; not peak RSS."};
  }
};
#else
class HotIndex final : public Index {
 public:
  Descriptor descriptor() const override {
    Descriptor d; d.implementation = "hot-single-threaded-unavailable";
    d.implementation_version = "96bf6fb7103b27e50e16a6026db8974c090ee84a";
    d.adapter_version = "blocklearn-baselines-v1";
    d.capabilities.keys.allows_nul = false; d.capabilities.keys.max_bytes = 254;
    d.adaptation_notes = "Native HOT not built: requires GCC, source and x86-64 "
        "AVX2/BMI/BMI2/LZCNT/POPCNT. All operations unsupported. No substitute.";
    return d;
  }
  void bulk_load(const std::vector<Row>&) override { throw UnsupportedOperation("native HOT unavailable"); }
  std::optional<Value> find(std::string_view) const override { throw UnsupportedOperation("native HOT unavailable"); }
  InsertResult insert(std::string_view, Value) override { throw UnsupportedOperation("native HOT unavailable"); }
  UpdateResult update(std::string_view, Value) override { throw UnsupportedOperation("native HOT unavailable"); }
  EraseResult erase(std::string_view) override { throw UnsupportedOperation("native HOT unavailable"); }
  std::vector<Row> scan(std::string_view, std::size_t) const override { throw UnsupportedOperation("native HOT unavailable"); }
  Value range_sum(std::string_view, std::string_view) const override { throw UnsupportedOperation("native HOT unavailable"); }
  MemoryUsage memory_usage() const override { return {}; }
};
#endif
}
std::unique_ptr<string_index_bench::v1::Index> make_hot_baseline(
    const string_index_bench::v1::Parameters& params) {
  no_parameters(params); return std::make_unique<HotIndex>();
}
