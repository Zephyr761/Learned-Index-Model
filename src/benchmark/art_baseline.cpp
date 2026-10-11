#include "factories.h"
#include "baseline_common.h"
#include <array>
#include <deque>
#include <climits>

#ifdef BLOCKLEARN_HAVE_ART
#include <art.h>
#endif

namespace {
using namespace string_index_bench::v1;
using namespace blocklearn::bench;
#ifdef BLOCKLEARN_HAVE_ART
class ArtIndex final : public Index {
  art_tree tree_{};
  const std::vector<Row>* input_ = nullptr;
  std::deque<Value> values_;
  bool loaded_ = false;
  void ready() const { if (!loaded_) throw std::logic_error("bulk_load required"); }
  // libart needs an explicit EOS to represent prefix-related keys safely.
  // Embedded NUL is excluded by the descriptor; no dataset is silently filtered.
  struct Key {
    std::array<unsigned char, 256> small{};
    std::vector<unsigned char> large;
    const unsigned char* data;
    explicit Key(std::string_view k) {
      cstring_key(k, INT_MAX - 1);
      if (k.size() + 1 <= small.size()) {
        if (!k.empty()) std::memcpy(small.data(), k.data(), k.size());
        data = small.data();
      } else {
        large.assign(k.begin(), k.end()); large.push_back(0); data = large.data();
      }
    }
  };
  static std::size_t bytes(const art_node* n) {
    if (!n) return 0;
    const auto bits = reinterpret_cast<std::uintptr_t>(n);
    if (bits & 1) {
      auto* leaf = reinterpret_cast<const art_leaf*>(bits & ~std::uintptr_t(1));
      return sizeof(art_leaf) + leaf->key_len;
    }
    std::size_t result = 0;
    auto add = [&](const auto* node, std::size_t count) {
      result += sizeof(*node);
      for (std::size_t i = 0; i < count; ++i) result += bytes(node->children[i]);
    };
    switch (n->type) {
      case NODE4: add(reinterpret_cast<const art_node4*>(n), n->num_children); break;
      case NODE16: add(reinterpret_cast<const art_node16*>(n), n->num_children); break;
      case NODE48: add(reinterpret_cast<const art_node48*>(n), 48); break;
      case NODE256: add(reinterpret_cast<const art_node256*>(n), 256); break;
      default: throw std::logic_error("bad ART node type");
    }
    return result;
  }
 public:
  ArtIndex() { art_tree_init(&tree_); }
  ~ArtIndex() override { art_tree_destroy(&tree_); }
  Descriptor descriptor() const override {
    Descriptor d; d.implementation = "art-libart";
    d.implementation_version = "301046804af165269e37da6725f5a4aec9ecc881";
    d.adapter_version = "blocklearn-baselines-v1";
    auto& c = d.capabilities;
    c.keys.allows_nul = false; c.keys.max_bytes = INT_MAX - 1;
    c.bulk_load = Support::adapted;
    c.find = c.insert = Support::native;
    c.scan = c.range_sum = Support::adapted;
    d.adaptation_notes = "Unmodified armon/libart. Bulkload uses repeated native "
        "inserts; explicit terminal NUL included in libart key length. Query key "
        "copy/EOS cost is timed. Scan/range_sum use native ordered iteration "
        "and filtering (not optimized lower_bound). Bulk values borrow input.";
    return d;
  }
  void bulk_load(const std::vector<Row>& rows) override {
    if (loaded_) throw std::logic_error("bulk_load called twice");
    validate_rows(rows);
    for (const auto& r : rows) cstring_key(r.key, INT_MAX - 1);
    input_ = &rows;
    for (const auto& r : rows) {
      Key k(r.key);
      art_insert_no_replace(&tree_, k.data, int(r.key.size() + 1), const_cast<Value*>(&r.value));
    }
    loaded_ = true;
  }
  std::optional<Value> find(std::string_view key) const override {
    ready(); Key k(key);
    auto* p = static_cast<Value*>(art_search(&tree_, k.data, int(key.size() + 1)));
    return p ? std::optional<Value>(*p) : std::nullopt;
  }
  InsertResult insert(std::string_view key, Value value) override {
    ready(); Key k(key); values_.push_back(value);
    auto* old = art_insert_no_replace(&tree_, k.data, int(key.size() + 1), &values_.back());
    if (old) { values_.pop_back(); return InsertResult::already_exists; }
    return InsertResult::inserted;
  }
  UpdateResult update(std::string_view, Value) override { throw UnsupportedOperation("update unsupported"); }
  EraseResult erase(std::string_view) override { throw UnsupportedOperation("erase unsupported"); }
  std::vector<Row> scan(std::string_view lower, std::size_t limit) const override {
    ready(); cstring_key(lower, INT_MAX - 1);
    struct State { std::string_view lower; std::size_t limit; std::vector<Row> rows; } state{lower, limit, {}};
    if (!limit) return {};
    art_iter(const_cast<art_tree*>(&tree_), [](void* context, const unsigned char* k,
                                             std::uint32_t length, void* value) {
      auto& s = *static_cast<State*>(context);
      std::string_view key(reinterpret_cast<const char*>(k), length - 1);
      if (!blocklearn::less(key, s.lower)) s.rows.push_back({std::string(key), *static_cast<Value*>(value)});
      return s.rows.size() == s.limit ? 1 : 0;
    }, &state);
    return state.rows;
  }
  Value range_sum(std::string_view lower, std::string_view upper) const override {
    ready(); cstring_key(lower, INT_MAX - 1); cstring_key(upper, INT_MAX - 1);
    if (blocklearn::less(upper, lower)) return 0;
    struct State { std::string_view lower, upper; Value sum; } state{lower, upper, 0};
    art_iter(const_cast<art_tree*>(&tree_), [](void* context, const unsigned char* k,
                                             std::uint32_t length, void* value) {
      auto& s = *static_cast<State*>(context);
      std::string_view key(reinterpret_cast<const char*>(k), length - 1);
      if (blocklearn::less(s.upper, key)) return 1;
      if (!blocklearn::less(key, s.lower)) s.sum += *static_cast<Value*>(value);
      return 0;
    }, &state);
    return state.sum;
  }
  MemoryUsage memory_usage() const override {
    ready();
    return {bytes(tree_.root), sizeof(*this) + values_.size() * sizeof(Value), input_bytes(*input_),
      "ART logical allocated node/leaf sizes; adapter capacity estimate. Input "
      "retained in full for borrowed values. Excludes allocator/deque overhead; not peak RSS."};
  }
};
#else
class ArtIndex final : public Index {
 public:
  Descriptor descriptor() const override {
    Descriptor d; d.implementation = "art-libart-unavailable";
    d.implementation_version = "source-not-configured"; d.adapter_version = "blocklearn-baselines-v1";
    d.adaptation_notes = "ART_SOURCE_DIR missing; no replacement implementation.";
    return d;
  }
  void bulk_load(const std::vector<Row>&) override { throw UnsupportedOperation("ART source unavailable"); }
  std::optional<Value> find(std::string_view) const override { throw UnsupportedOperation("ART unavailable"); }
  InsertResult insert(std::string_view, Value) override { throw UnsupportedOperation("ART unavailable"); }
  UpdateResult update(std::string_view, Value) override { throw UnsupportedOperation("ART unavailable"); }
  EraseResult erase(std::string_view) override { throw UnsupportedOperation("ART unavailable"); }
  std::vector<Row> scan(std::string_view, std::size_t) const override { throw UnsupportedOperation("ART unavailable"); }
  Value range_sum(std::string_view, std::string_view) const override { throw UnsupportedOperation("ART unavailable"); }
  MemoryUsage memory_usage() const override { return {}; }
};
#endif
}
std::unique_ptr<string_index_bench::v1::Index> make_art_baseline(
    const string_index_bench::v1::Parameters& params) {
  no_parameters(params); return std::make_unique<ArtIndex>();
}
