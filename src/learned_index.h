#pragma once

#include <index_contract.h>
#include <deque>
#include <memory>
#include <new>
#include "block_model.h"
#include "residual_tags.h"

namespace blocklearn {

enum class Verification { full_key, hash, residual };
struct Options {
  Verification verification = Verification::residual;
  bool allow_insert = true;
  bool instrument = false;
  std::size_t region_size = 1024;
  std::size_t directory_fanout = 64;
  std::size_t delta_capacity = 16;
  std::size_t max_segments = 8;
  std::size_t max_block_error = 1;
  unsigned tag_features = 2;
};
struct Statistics {
  std::uint64_t queries = 0, full_key_comparisons = 0, tag_tests = 0;
  std::uint64_t directory_fallbacks = 0, merges = 0, leaf_splits = 0;
  std::uint64_t directory_splits = 0, regions_trained = 0;
  std::size_t leaves = 0, directory_nodes = 0, height = 0, delta_rows = 0;
};

template<class T> struct CacheAlignedAllocator {
  using value_type = T;
  CacheAlignedAllocator() noexcept = default;
  template<class U> CacheAlignedAllocator(const CacheAlignedAllocator<U>&) noexcept {}
  T* allocate(std::size_t n) {
    if (n > std::size_t(-1) / sizeof(T)) throw std::bad_array_new_length();
    return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(64)));
  }
  void deallocate(T* p, std::size_t) noexcept {
    ::operator delete(p, std::align_val_t(64));
  }
  template<class U> bool operator==(const CacheAlignedAllocator<U>&) const { return true; }
  template<class U> bool operator!=(const CacheAlignedAllocator<U>&) const { return false; }
};

class LearnedIndex final : public string_index_bench::v1::Index {
 public:
  using Row = string_index_bench::v1::Row;
  using Value = string_index_bench::v1::Value;
  explicit LearnedIndex(Options options = {});
  string_index_bench::v1::Descriptor descriptor() const override;
  void bulk_load(const std::vector<Row>& rows) override;
  std::optional<Value> find(std::string_view key) const override;
  string_index_bench::v1::InsertResult insert(std::string_view key, Value value) override;
  string_index_bench::v1::UpdateResult update(std::string_view, Value) override;
  string_index_bench::v1::EraseResult erase(std::string_view) override;
  std::vector<Row> scan(std::string_view lower, std::size_t limit) const override;
  Value range_sum(std::string_view lower, std::string_view upper) const override;
  string_index_bench::v1::MemoryUsage memory_usage() const override;
  Statistics statistics() const;
  // Expensive semantic audit, never part of the timed query path.
  void validate() const;

 private:
  using Id = std::uint32_t;
  struct DeltaEntry { Id id; std::uint16_t tag; };
  struct Node {
    bool leaf = true;
    std::vector<Id> ids;
    std::vector<std::uint16_t, CacheAlignedAllocator<std::uint16_t>> tags;
    std::vector<DeltaEntry> delta;
    std::vector<std::unique_ptr<Node>> children;
    std::vector<std::string_view> fences;
    BlockModel model;
    TagPlan tag_plan;
    std::size_t merges_since_features = 0;
    Node* next = nullptr;
  };
  Options options_;
  const std::vector<Row>* input_ = nullptr;
  std::deque<Row> inserted_;  // stable owning storage; no per-key heap pointer
  std::unique_ptr<Node> root_;
  bool loaded_ = false;
  mutable Statistics counters_;

  void require_loaded() const;
  const Row& row(Id id) const;
  std::string_view first_key(const Node&) const;
  std::uint16_t tag(const Node&, std::string_view) const;
  void train_leaf(Node&, bool reuse_features = false);
  void train_directory(Node&);
  void bulk_leaves(std::vector<Id>, std::vector<std::unique_ptr<Node>>&);
  std::size_t child_position(const Node&, std::string_view) const;
  const Node* leaf_for(std::string_view) const;
  std::optional<Value> find_leaf(const Node&, std::string_view) const;
  std::unique_ptr<Node> insert_into(Node&, Id);
  std::vector<Id> ordered_ids(const Node&) const;
  std::size_t node_bytes(const Node&) const;
};

}  // namespace blocklearn
