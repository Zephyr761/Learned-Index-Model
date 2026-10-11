#include "learned_index.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include "key_bytes.h"
#include "tag_probe.h"

namespace blocklearn {
using namespace string_index_bench::v1;

LearnedIndex::LearnedIndex(Options options) : options_(options) {
  if (options.region_size < 16 || options.region_size > (1U << 20) ||
      options.directory_fanout < 4 || options.directory_fanout > 1024 ||
      !options.delta_capacity || options.delta_capacity > options.region_size ||
      !options.max_segments || options.max_segments > 32 ||
      options.max_block_error > 16 || options.tag_features > 4)
    throw std::invalid_argument("invalid learned-index configuration");
}
Descriptor LearnedIndex::descriptor() const {
  Descriptor d;
  d.implementation = options_.verification == Verification::full_key ? "block-plain"
      : options_.verification == Verification::hash ? "block-hash"
      : options_.allow_insert ? "block-residual-delta" : "block-residual";
  d.implementation_version = "blocklearn-prototype-v1";
  d.adapter_version = "string-index-v1-2026-10";
  auto& c = d.capabilities;
  c.bulk_load = c.find = c.scan = c.range_sum = Support::native;
  c.insert = options_.allow_insert ? Support::native : Support::unsupported;
  d.adaptation_notes = "Single-threaded. Borrowed immutable bulk rows, stable owned "
      "insert rows. Exact unsigned-byte semantics. No neural network or hotspot cache. "
      "32-bit record IDs limit total records to UINT32_MAX.";
  return d;
}
void LearnedIndex::require_loaded() const {
  if (!loaded_) throw std::logic_error("bulk_load required");
}
const LearnedIndex::Row& LearnedIndex::row(Id id) const {
  return id < input_->size() ? (*input_)[id] : inserted_[id - input_->size()];
}
std::string_view LearnedIndex::first_key(const Node& n) const {
  if (!n.leaf) return n.fences.front();
  std::string_view result;
  bool set = false;
  if (!n.ids.empty()) { result = row(n.ids.front()).key; set = true; }
  for (auto d : n.delta)
    if (!set || less(row(d.id).key, result)) { result = row(d.id).key; set = true; }
  return result;
}
std::uint16_t LearnedIndex::tag(const Node& n, std::string_view key) const {
  return options_.verification == Verification::hash
      ? hash_tag(key, n.model.prefix_size()) : n.tag_plan.tag(key);
}
void LearnedIndex::train_leaf(Node& n, bool reuse_features) {
  ++counters_.regions_trained;
  std::vector<std::string_view> keys;
  keys.reserve(n.ids.size());
  for (auto id : n.ids) keys.push_back(row(id).key);
  if (keys.empty()) { n.model = BlockModel(); n.tags.clear(); return; }
  // Normal merges do not need a fresh grid search. Refit the existing small
  // model and re-certify every base key. Periodically audit feature quality;
  // exactness does not depend on collision rate or on this heuristic schedule.
  if (reuse_features && n.model.certified() && n.model.segment_count()) {
    auto refit = BlockModel::fit(keys, n.model.block_size(), n.model.encoding_bytes(),
                                n.model.segment_count(), options_.max_block_error);
    if (refit.certified()) {
      const bool degraded = refit.average_blocks() > n.model.average_blocks() * 1.25;
      n.model = std::move(refit);
      ++n.merges_since_features;
      if (degraded || n.merges_since_features >= 8) {
        n.tag_plan.candidate_matches = evaluate_tags(keys, n.model, n.tag_plan);
        if (n.tag_plan.candidate_matches > 3)
          n.tag_plan = learn_tags(keys, n.model, options_.tag_features);
        n.merges_since_features = 0;
      }
      n.tags.clear();
      if (options_.verification != Verification::full_key) {
        n.tags.reserve(keys.size());
        for (auto key : keys) n.tags.push_back(tag(n, key));
      }
      return;
    }
  }
  std::vector<BlockModel> candidates;
  // The planner is shared by all ablations: only verification/write handling
  // differs. Keep the two cheapest certified plans, then evaluate residual tags.
  for (std::size_t b : {16U, 32U, 64U})
    for (unsigned bytes : {2U, 4U, 7U})
      for (std::size_t s = 1; s <= options_.max_segments; s *= 2) {
        auto m = BlockModel::fit(keys, b, bytes, s, options_.max_block_error);
        if (m.certified()) candidates.push_back(std::move(m));
      }
  if (candidates.empty()) {
    n.model = BlockModel::fit(keys, 32, 7, options_.max_segments,
                             options_.max_block_error);
    n.tags.clear();
    return;  // caller splits difficult regions; exact binary fallback is safe
  }
  std::stable_sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
    return a.planning_cost(keys.size()) < b.planning_cost(keys.size());
  });
  double best_cost = std::numeric_limits<double>::infinity();
  BlockModel best_model;
  TagPlan best_tags;
  for (std::size_t i = 0; i < std::min<std::size_t>(2, candidates.size()); ++i) {
    auto plan = reuse_features ? n.tag_plan
        : learn_tags(keys, candidates[i], options_.tag_features);
    if (reuse_features) {
      plan.candidate_matches = evaluate_tags(keys, candidates[i], plan);
      plan.cost = plan.candidate_matches + 0.04 * plan.count;
      if (plan.candidate_matches > 3)
        plan = learn_tags(keys, candidates[i], options_.tag_features);
    }
    const auto cost = candidates[i].planning_cost(keys.size()) + plan.cost;
    if (cost < best_cost) {
      best_cost = cost; best_model = candidates[i]; best_tags = plan;
    }
  }
  n.model = std::move(best_model);
  n.tag_plan = best_tags;
  n.merges_since_features = 0;
  n.tags.clear();
  if (options_.verification != Verification::full_key) {
    n.tags.reserve(n.ids.size());
    for (auto key : keys) n.tags.push_back(tag(n, key));
  }
}
void LearnedIndex::train_directory(Node& n) {
  n.fences.clear();
  for (const auto& child : n.children) n.fences.push_back(first_key(*child));
  double cost = std::numeric_limits<double>::infinity();
  BlockModel best;
  for (unsigned bytes : {2U, 4U, 7U})
    for (std::size_t s = 1; s <= options_.max_segments; s *= 2) {
      auto m = BlockModel::fit(n.fences, 1, bytes, s, 2);
      if (m.certified() && m.planning_cost(n.fences.size()) < cost) {
        cost = m.planning_cost(n.fences.size()); best = std::move(m);
      }
    }
  n.model = std::move(best);  // uncertified empty model => exact ordered directory
}
void LearnedIndex::bulk_leaves(std::vector<Id> ids,
                               std::vector<std::unique_ptr<Node>>& leaves) {
  auto n = std::make_unique<Node>();
  n->ids = std::move(ids);
  train_leaf(*n);
  if (!n->model.certified() && n->ids.size() > 64) {
    const auto middle = n->ids.begin() + n->ids.size() / 2;
    std::vector<Id> right(middle, n->ids.end());
    n->ids.erase(middle, n->ids.end());
    bulk_leaves(std::move(n->ids), leaves);
    bulk_leaves(std::move(right), leaves);
    return;
  }
  if (!leaves.empty()) leaves.back()->next = n.get();
  leaves.push_back(std::move(n));
}
void LearnedIndex::bulk_load(const std::vector<Row>& rows) {
  if (loaded_) throw std::logic_error("bulk_load called twice");
  if (rows.size() > std::numeric_limits<Id>::max())
    throw std::length_error("32-bit record ID capacity exceeded");
  for (std::size_t i = 1; i < rows.size(); ++i)
    if (!less(rows[i - 1].key, rows[i].key))
      throw std::invalid_argument("bulk rows must be strictly byte-sorted and unique");
  input_ = &rows;
  std::vector<std::unique_ptr<Node>> level;
  for (std::size_t begin = 0; begin < rows.size(); begin += options_.region_size) {
    std::vector<Id> ids;
    const auto end = std::min(rows.size(), begin + options_.region_size);
    ids.reserve(end - begin);
    for (auto i = begin; i < end; ++i) ids.push_back(static_cast<Id>(i));
    bulk_leaves(std::move(ids), level);
  }
  while (level.size() > 1) {
    std::vector<std::unique_ptr<Node>> parents;
    for (std::size_t i = 0; i < level.size();) {
      auto n = std::make_unique<Node>(); n->leaf = false;
      const auto end = std::min(level.size(), i + options_.directory_fanout);
      for (; i < end; ++i) n->children.push_back(std::move(level[i]));
      train_directory(*n);
      parents.push_back(std::move(n));
    }
    level = std::move(parents);
  }
  root_ = level.empty() ? std::make_unique<Node>() : std::move(level.front());
  loaded_ = true;
}
std::size_t LearnedIndex::child_position(const Node& n, std::string_view key) const {
  auto ordered_less = [&](std::string_view a, std::string_view b) {
    if (options_.instrument) ++counters_.full_key_comparisons;
    return less(a, b);
  };
  auto upper = [&](std::size_t lo, std::size_t hi) {
    return std::size_t(std::upper_bound(n.fences.begin() + lo,
        n.fences.begin() + hi, key,
        [&](std::string_view a, std::string_view b) { return ordered_less(a, b); })
        - n.fences.begin());
  };
  if (n.model.certified() && n.model.accepts_prefix(key)) {
    const auto w = n.model.window(key);
    const auto at = upper(w.first, w.last);
    const auto p = at ? at - 1 : 0;
    // A stored-separator certificate alone is not assumed to cover arbitrary
    // unseen queries. Validate the true predecessor bracket before trusting it.
    if ((p == 0 || !ordered_less(key, n.fences[p])) &&
        (p + 1 == n.fences.size() || ordered_less(key, n.fences[p + 1]))) return p;
  }
  if (options_.instrument) ++counters_.directory_fallbacks;
  const auto at = upper(0, n.fences.size());
  return at ? at - 1 : 0;
}
const LearnedIndex::Node* LearnedIndex::leaf_for(std::string_view key) const {
  auto n = root_.get();
  while (!n->leaf) n = n->children[child_position(*n, key)].get();
  return n;
}
std::optional<LearnedIndex::Value> LearnedIndex::find_leaf(
    const Node& n, std::string_view key) const {
  const bool filter = options_.verification != Verification::full_key;
  const auto fingerprint = filter ? tag(n, key) : 0;
  auto matches = [&](Id id) {
    if (options_.instrument) ++counters_.full_key_comparisons;
    return row(id).key == key;
  };
  // Delta is checked independently of the old base prefix/model certificate.
  // Newly inserted keys may shorten the common prefix or be out of model range.
  for (auto d : n.delta) {
    if (options_.instrument && filter) ++counters_.tag_tests;
    if ((!filter || d.tag == fingerprint) && matches(d.id)) return row(d.id).value;
  }
  if (n.ids.empty()) return std::nullopt;
  if (!n.model.certified()) {
    auto it = std::lower_bound(n.ids.begin(), n.ids.end(), key,
        [&](Id id, std::string_view k) {
          if (options_.instrument) ++counters_.full_key_comparisons;
          return less(row(id).key, k);
        });
    return it != n.ids.end() && matches(*it)
        ? std::optional<Value>(row(*it).value) : std::nullopt;
  }
  if (!n.model.accepts_prefix(key)) return std::nullopt;
  const auto w = n.model.window(key);
  const auto lo = w.first * n.model.block_size();
  const auto hi = std::min(n.ids.size(), w.last * n.model.block_size());
  if (!filter) {
    auto it = std::lower_bound(n.ids.begin() + lo, n.ids.begin() + hi, key,
        [&](Id id, std::string_view k) {
          if (options_.instrument) ++counters_.full_key_comparisons;
          return less(row(id).key, k);
        });
    return it != n.ids.begin() + hi && matches(*it)
        ? std::optional<Value>(row(*it).value) : std::nullopt;
  }
  for (auto i = lo; i < hi; i += 16) {
    const auto count = std::min<std::size_t>(16, hi - i);
    if (options_.instrument) counters_.tag_tests += count;
    auto mask = tag_mask(n.tags.data() + i, count, fingerprint);
    while (mask) {
      const auto lane = unsigned(__builtin_ctz(mask));
      const auto id = n.ids[i + lane];
      if (matches(id)) return row(id).value;
      mask &= mask - 1;
    }
  }
  return std::nullopt;
}
std::optional<LearnedIndex::Value> LearnedIndex::find(std::string_view key) const {
  require_loaded();
  if (options_.instrument) ++counters_.queries;
  return find_leaf(*leaf_for(key), key);
}
std::vector<LearnedIndex::Id> LearnedIndex::ordered_ids(const Node& n) const {
  auto result = n.ids;
  for (auto d : n.delta) result.push_back(d.id);
  if (!n.delta.empty()) std::sort(result.begin(), result.end(),
      [&](Id a, Id b) { return less(row(a).key, row(b).key); });
  return result;
}
std::unique_ptr<LearnedIndex::Node> LearnedIndex::insert_into(Node& n, Id id) {
  if (n.leaf) {
    n.delta.push_back({id, tag(n, row(id).key)});
    if (n.delta.size() < options_.delta_capacity) return {};
    ++counters_.merges;
    n.ids = ordered_ids(n); n.delta.clear();
    if (n.ids.size() <= options_.region_size) train_leaf(n, true);
    if (n.ids.size() <= options_.region_size &&
        (n.model.certified() || n.ids.size() <= 64)) return {};
    auto sibling = std::make_unique<Node>();
    const auto middle = n.ids.begin() + n.ids.size() / 2;
    sibling->ids.assign(middle, n.ids.end()); n.ids.erase(middle, n.ids.end());
    sibling->next = n.next; n.next = sibling.get();
    train_leaf(n, true); train_leaf(*sibling);
    ++counters_.leaf_splits;
    return sibling;
  }
  const auto p = child_position(n, row(id).key);
  const auto old_first = first_key(*n.children[p]);
  auto sibling = insert_into(*n.children[p], id);
  const bool split = bool(sibling);
  const bool changed = old_first != first_key(*n.children[p]);
  if (sibling) n.children.insert(n.children.begin() + p + 1, std::move(sibling));
  if (split || changed) train_directory(n);
  if (n.children.size() <= options_.directory_fanout) return {};
  auto right = std::make_unique<Node>(); right->leaf = false;
  const auto middle = n.children.size() / 2;
  for (auto i = middle; i < n.children.size(); ++i)
    right->children.push_back(std::move(n.children[i]));
  n.children.resize(middle);
  train_directory(n); train_directory(*right);
  ++counters_.directory_splits;
  return right;
}
InsertResult LearnedIndex::insert(std::string_view key, Value value) {
  require_loaded();
  if (!options_.allow_insert) throw UnsupportedOperation("static ablation: insert unsupported");
  if (find(key).has_value()) return InsertResult::already_exists;
  const auto total = input_->size() + inserted_.size();
  if (total >= std::numeric_limits<Id>::max()) throw std::length_error("record IDs exhausted");
  inserted_.push_back({std::string(key), value});
  auto sibling = insert_into(*root_, static_cast<Id>(total));
  if (sibling) {
    auto new_root = std::make_unique<Node>(); new_root->leaf = false;
    new_root->children.push_back(std::move(root_));
    new_root->children.push_back(std::move(sibling));
    train_directory(*new_root); root_ = std::move(new_root);
  }
  return InsertResult::inserted;
}
UpdateResult LearnedIndex::update(std::string_view, Value) {
  throw UnsupportedOperation("update not implemented");
}
EraseResult LearnedIndex::erase(std::string_view) {
  throw UnsupportedOperation("erase not implemented");
}
std::vector<LearnedIndex::Row> LearnedIndex::scan(std::string_view lower,
                                               std::size_t limit) const {
  require_loaded();
  std::vector<Row> result;
  for (auto n = leaf_for(lower); n && result.size() < limit; n = n->next) {
    const auto ids = ordered_ids(*n);
    auto it = std::lower_bound(ids.begin(), ids.end(), lower,
        [&](Id id, std::string_view k) { return less(row(id).key, k); });
    for (; it != ids.end() && result.size() < limit; ++it) result.push_back(row(*it));
  }
  return result;
}
LearnedIndex::Value LearnedIndex::range_sum(std::string_view lower,
                                           std::string_view upper) const {
  require_loaded();
  if (less(upper, lower)) return 0;
  Value total = 0;
  for (auto n = leaf_for(lower); n; n = n->next) {
    for (auto id : ordered_ids(*n)) {
      const auto& r = row(id);
      if (less(r.key, lower)) continue;
      if (less(upper, r.key)) return total;
      total += r.value;
    }
  }
  return total;
}
std::size_t LearnedIndex::node_bytes(const Node& n) const {
  std::size_t bytes = sizeof(Node) + n.ids.capacity() * sizeof(Id)
      + n.tags.capacity() * sizeof(std::uint16_t)
      + n.delta.capacity() * sizeof(DeltaEntry)
      + n.children.capacity() * sizeof(std::unique_ptr<Node>)
      + n.fences.capacity() * sizeof(std::string_view) + n.model.owned_bytes();
  for (const auto& child : n.children) bytes += node_bytes(*child);
  return bytes;
}
// Approximate string allocations without double-counting inline SSO storage.
static std::size_t heap_bytes(const std::string& s) {
  const auto data = reinterpret_cast<std::uintptr_t>(s.data());
  const auto object = reinterpret_cast<std::uintptr_t>(&s);
  return data >= object && data < object + sizeof(s) ? 0 : s.capacity() + 1;
}
MemoryUsage LearnedIndex::memory_usage() const {
  require_loaded();
  std::size_t owned = sizeof(*this) + node_bytes(*root_) + inserted_.size() * sizeof(Row);
  for (const auto& r : inserted_) owned += heap_bytes(r.key);
  std::size_t retained = input_->capacity() * sizeof(Row);
  for (const auto& r : *input_) retained += heap_bytes(r.key);
  return {owned, 0, retained,
      "Capacity-based estimate; borrowed input counted in full. Excludes allocator "
      "headers, deque spare blocks/map, transient training/merge allocations. "
      "Not peak RSS; IDs/tags are owned, payloads of bulk rows are borrowed."};
}
Statistics LearnedIndex::statistics() const {
  auto s = counters_;
  auto visit = [&](auto&& self, const Node& n, std::size_t depth) -> void {
    s.height = std::max(s.height, depth);
    if (n.leaf) { ++s.leaves; s.delta_rows += n.delta.size(); }
    else { ++s.directory_nodes; for (const auto& c : n.children) self(self, *c, depth + 1); }
  };
  if (root_) visit(visit, *root_, 1);
  return s;
}
void LearnedIndex::validate() const {
  require_loaded();
  std::vector<const Node*> leaves;
  std::vector<bool> seen(input_->size() + inserted_.size(), false);
  auto check = [&](auto&& self, const Node& n) -> void {
    if (!n.leaf) {
      if (n.children.size() != n.fences.size() || n.children.empty())
        throw std::logic_error("bad directory sizes");
      for (std::size_t i = 0; i < n.children.size(); ++i) {
        if (n.fences[i] != first_key(*n.children[i])) throw std::logic_error("stale fence");
        if (i && !less(n.fences[i - 1], n.fences[i])) throw std::logic_error("unordered fences");
        self(self, *n.children[i]);
      }
      return;
    }
    leaves.push_back(&n);
    if (n.model.certified()) for (std::size_t i = 0; i < n.ids.size(); ++i) {
      const auto w = n.model.window(row(n.ids[i]).key);
      if (i / n.model.block_size() < w.first || i / n.model.block_size() >= w.last)
        throw std::logic_error("prediction omitted existing key");
    }
    if (options_.verification != Verification::full_key && n.model.certified()) {
      if (n.tags.size() != n.ids.size()) throw std::logic_error("bad tag count");
      for (std::size_t i = 0; i < n.ids.size(); ++i)
        if (n.tags[i] != tag(n, row(n.ids[i]).key)) throw std::logic_error("stale base tag");
    }
    for (auto d : n.delta) if (d.tag != tag(n, row(d.id).key))
      throw std::logic_error("stale delta tag");
    for (auto id : ordered_ids(n)) {
      if (id >= seen.size() || seen[id]) throw std::logic_error("duplicate/missing record ID");
      seen[id] = true;
    }
  };
  check(check, *root_);
  std::string_view previous;
  bool have_previous = false;
  for (std::size_t i = 0; i < leaves.size(); ++i) {
    if (leaves[i]->next != (i + 1 < leaves.size() ? leaves[i + 1] : nullptr))
      throw std::logic_error("bad leaf chain");
    for (auto id : ordered_ids(*leaves[i])) {
      if (have_previous && !less(previous, row(id).key)) throw std::logic_error("global order broken");
      previous = row(id).key; have_previous = true;
      if (leaf_for(previous) != leaves[i]) throw std::logic_error("range routing omitted key");
    }
  }
  if (std::find(seen.begin(), seen.end(), false) != seen.end()) throw std::logic_error("lost record");
}

}  // namespace blocklearn
