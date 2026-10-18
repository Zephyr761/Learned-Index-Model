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
  if (options.predicted_first && options.verification == Verification::full_key)
    throw std::invalid_argument("predicted_first requires tag verification");
  if ((options.verification == Verification::adaptive && !options.workload_planner) ||
      (options.workload_planner && options.verification == Verification::full_key))
    throw std::invalid_argument("adaptive tags require the work planner; full-key binary search uses the legacy planner");
  if (options.cost_weights) options.cost_weights->validate();
}
Descriptor LearnedIndex::descriptor() const {
  Descriptor d;
  d.implementation = options_.verification == Verification::adaptive ? "block-adaptive-delta"
      : options_.verification == Verification::full_key ? "block-plain"
      : options_.verification == Verification::hash ? "block-hash"
      : options_.allow_insert ? "block-residual-delta" : "block-residual";
  if (options_.short_directory) d.implementation += "-shortdir";
  if (options_.predicted_first) d.implementation += "-first";
  if (options_.workload_planner && options_.verification != Verification::adaptive) d.implementation += "-cost";
  d.implementation_version = "blocklearn-prototype-v3";
  d.adapter_version = "string-index-v1-2026-10";
  auto& c = d.capabilities;
  c.bulk_load = c.find = c.scan = c.range_sum = Support::native;
  c.insert = options_.allow_insert ? Support::native : Support::unsupported;
  d.adaptation_notes = "Single-threaded. Borrowed immutable bulk rows, stable owned "
      "insert rows. Exact unsigned-byte semantics. No neural network or hotspot cache. "
      "32-bit record IDs limit total records to UINT32_MAX.";
  d.adaptation_notes += std::string(" Short-directory routing=") +
      (options_.short_directory ? "on" : "off") + "; predicted-block-first=" +
      (options_.verification == Verification::adaptive && options_.predicted_first
          ? "allowed-per-region." : options_.predicted_first ? "on." : "off.");
  d.adaptation_notes += options_.workload_planner
      ? " Calibrated actual-batch/early-return planning; cost profile required at bulk_load; no benchmark queries used."
      : " Original legacy planner retained as control.";
  if (options_.verification == Verification::adaptive)
    d.adaptation_notes += " Per-region hash/residual tags and allowed probe order chosen on held-out stored-key samples.";
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
      || (options_.verification == Verification::adaptive && n.tag_kind == TagKind::hash)
      ? hash_tag(key, n.model.prefix_size()) : n.tag_plan.tag(key);
}
void LearnedIndex::train_work_leaf(Node& n, const std::vector<std::string_view>& keys, bool reuse) {
  if (!n.delta.empty()) throw std::logic_error("cannot reconfigure tags with unmerged delta");
  const auto& weights = *options_.cost_weights;
  RegionPlanningOptions settings;
  settings.max_segments = options_.max_segments; settings.max_block_error = options_.max_block_error;
  settings.max_features = options_.tag_features;
  settings.allow_hash = options_.verification != Verification::residual;
  settings.allow_residual = options_.verification != Verification::hash;
  settings.predicted_first = options_.predicted_first;
  settings.choose_order = options_.verification == Verification::adaptive;
  RegionPlan plan;
  bool refitted = false;
  if (reuse && n.model.certified() && n.model.segment_count()) {
    auto model = BlockModel::fit(keys, n.model.block_size(), n.model.encoding_bytes(),
                                 n.model.segment_count(), options_.max_block_error);
    if (model.certified()) {
      TagStrategy strategy; strategy.kind = n.tag_kind; strategy.learned = n.tag_plan;
      strategy.predicted_first = n.predicted_first;
      const auto score = evaluate_query_work(keys, model, strategy, weights, planning_samples(keys, true));
      ++n.merges_since_features;
      if (n.merges_since_features < 8 && score.cost <= n.planning_cost * 1.25)
        plan = {std::move(model), strategy, score};
      else {
        plan = plan_model(keys, model, weights, settings);
        n.merges_since_features = 0;
      }
      refitted = true;
    }
  }
  if (!refitted) {
    plan = plan_region(keys, weights, settings);
    n.merges_since_features = 0;
  }
  n.model = std::move(plan.model); n.tag_plan = plan.strategy.learned;
  n.tag_kind = plan.strategy.kind; n.predicted_first = plan.strategy.predicted_first;
  n.planning_cost = plan.evaluation.cost;
  n.tags.clear();
  if (n.model.certified()) {
    n.tags.reserve(keys.size());
    for (auto key : keys) n.tags.push_back(tag(n, key));
  }
}
void LearnedIndex::train_leaf(Node& n, bool reuse_features) {
  ++counters_.regions_trained;
  std::vector<std::string_view> keys;
  keys.reserve(n.ids.size());
  for (auto id : n.ids) keys.push_back(row(id).key);
  if (keys.empty()) { n.model = BlockModel(); n.tags.clear(); return; }
  if (options_.workload_planner) { train_work_leaf(n, keys, reuse_features); return; }
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
    return a.legacy_planning_cost(keys.size()) < b.legacy_planning_cost(keys.size());
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
    const auto cost = candidates[i].legacy_planning_cost(keys.size()) + plan.cost;
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
      if (m.certified() && m.legacy_planning_cost(n.fences.size()) < cost) {
        cost = m.legacy_planning_cost(n.fences.size()); best = std::move(m);
      }
    }
  n.model = std::move(best);  // uncertified empty model => exact ordered directory
  if (options_.short_directory) n.short_keys.rebuild(n.fences);
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
  if (options_.workload_planner && !options_.cost_weights)
    options_.cost_weights = environment_cost_profile().weights;
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
  NoLookupTrace trace;
  return child_position_traced(n, key, trace);
}
template<class Trace>
std::size_t LearnedIndex::child_position_traced(const Node& n, std::string_view key,
                                               Trace& trace) const {
  auto ordered_less = [&](std::string_view a, std::string_view b) {
    if (options_.instrument) ++counters_.full_key_comparisons;
    return less(a, b);
  };
  if (options_.short_directory) {
    const auto p = n.short_keys.predecessor(n.fences, key, ordered_less);
    if (p) return *p;  // optional presence, NOT whether the child index is nonzero
  }
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
  typename Trace::Scope fallback(trace, LookupStage::exact_fallback);
  if (options_.instrument) ++counters_.directory_fallbacks;
  const auto at = upper(0, n.fences.size());
  return at ? at - 1 : 0;
}
const LearnedIndex::Node* LearnedIndex::leaf_for(std::string_view key) const {
  NoLookupTrace trace;
  return leaf_for_traced(key, trace);
}
template<class Trace>
const LearnedIndex::Node* LearnedIndex::leaf_for_traced(std::string_view key,
                                                       Trace& trace) const {
  auto n = root_.get();
  if (n->leaf) return n;
  typename Trace::Scope routing(trace, LookupStage::directory_route);
  while (!n->leaf) n = n->children[child_position_traced(*n, key, trace)].get();
  return n;
}
template<class Trace>
std::optional<LearnedIndex::Value> LearnedIndex::find_leaf_traced(
    const Node& n, std::string_view key, Trace& trace) const {
  const bool filter = options_.verification != Verification::full_key;
  std::uint16_t fingerprint = 0;
  if (filter) {
    typename Trace::Scope generation(trace, LookupStage::tag_generate);
    fingerprint = tag(n, key);
  }
  auto matches = [&](Id id) {
    typename Trace::Scope verification(trace, LookupStage::key_verify);
    if (options_.instrument) ++counters_.full_key_comparisons;
    return row(id).key == key;
  };
  // Delta is checked independently of the old base prefix/model certificate.
  // Newly inserted keys may shorten the common prefix or be out of model range.
  if (!n.delta.empty()) {
    typename Trace::Scope delta(trace, LookupStage::delta_scan);
    for (auto d : n.delta) {
      if (options_.instrument && filter) ++counters_.tag_tests;
      if ((!filter || d.tag == fingerprint) && matches(d.id)) return row(d.id).value;
    }
  }
  if (n.ids.empty()) return std::nullopt;
  if (!n.model.certified()) {
    typename Trace::Scope fallback(trace, LookupStage::exact_fallback);
    auto it = std::lower_bound(n.ids.begin(), n.ids.end(), key,
        [&](Id id, std::string_view k) {
          if (options_.instrument) ++counters_.full_key_comparisons;
          return less(row(id).key, k);
        });
    return it != n.ids.end() && matches(*it)
        ? std::optional<Value>(row(*it).value) : std::nullopt;
  }
  std::size_t lo, hi, predicted;
  {
    typename Trace::Scope prediction(trace, LookupStage::leaf_model);
    if (!n.model.accepts_prefix(key)) return std::nullopt;
    const auto location = n.model.locate(key);
    lo = location.window.first * n.model.block_size();
    hi = std::min(n.ids.size(), location.window.last * n.model.block_size());
    predicted = location.predicted * n.model.block_size();
  }
  if (!filter) {
    typename Trace::Scope verification(trace, LookupStage::key_verify);
    auto it = std::lower_bound(n.ids.begin() + lo, n.ids.begin() + hi, key,
        [&](Id id, std::string_view k) {
          if (options_.instrument) ++counters_.full_key_comparisons;
          return less(row(id).key, k);
        });
    return it != n.ids.begin() + hi && matches(*it)
        ? std::optional<Value>(row(*it).value) : std::nullopt;
  }
  auto probe = [&](std::size_t begin, std::size_t end) -> std::optional<Value> {
    for (auto i = begin; i < end; i += 16) {
      std::uint32_t mask;
      {
        typename Trace::Scope filtering(trace, LookupStage::tag_filter);
        const auto count = std::min<std::size_t>(16, end - i);
        if (options_.instrument) { counters_.tag_tests += count; ++counters_.tag_batches; }
        mask = tag_mask(n.tags.data() + i, count, fingerprint);
      }
      while (mask) {
        const auto lane = unsigned(__builtin_ctz(mask));
        const auto id = n.ids[i + lane];
        if (matches(id)) return row(id).value;
        mask &= mask - 1;
      }
    }
    return std::nullopt;
  };
  if (options_.workload_planner ? n.predicted_first : options_.predicted_first) {
    const auto end = std::min(n.ids.size(), predicted + n.model.block_size());
    if (auto value = probe(predicted, end)) return value;
    if (auto value = probe(lo, predicted)) return value;
    return probe(end, hi);
  }
  return probe(lo, hi);
}
std::optional<LearnedIndex::Value> LearnedIndex::find(std::string_view key) const {
  NoLookupTrace trace;
  return lookup_traced(key, trace);
}
std::optional<LearnedIndex::Value> LearnedIndex::profile_find(
    std::string_view key, LookupProfile& profile) const {
  TimedLookupTrace trace(profile);
  return lookup_traced(key, trace);
}
template<class Trace>
std::optional<LearnedIndex::Value> LearnedIndex::lookup_traced(std::string_view key,
                                                            Trace& trace) const {
  require_loaded();
  if (options_.instrument) ++counters_.queries;
  return find_leaf_traced(*leaf_for_traced(key, trace), key, trace);
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
      + n.fences.capacity() * sizeof(std::string_view) + n.model.owned_bytes()
      + n.short_keys.owned_bytes();
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
    if (n.leaf) {
      ++s.leaves; s.delta_rows += n.delta.size();
      if (n.model.certified() && !n.ids.empty() && options_.verification != Verification::full_key) {
        if (options_.verification == Verification::hash ||
            (options_.verification == Verification::adaptive && n.tag_kind == TagKind::hash)) ++s.hash_leaves;
        else ++s.residual_leaves;
        if (options_.workload_planner ? n.predicted_first : options_.predicted_first) ++s.predicted_first_leaves;
      }
    }
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
      if (options_.short_directory && !n.short_keys.valid_for(n.fences))
        throw std::logic_error("stale short directory codes/prefix");
      for (std::size_t i = 0; i < n.children.size(); ++i) {
        if (n.fences[i] != first_key(*n.children[i])) throw std::logic_error("stale fence");
        if (i && !less(n.fences[i - 1], n.fences[i])) throw std::logic_error("unordered fences");
        self(self, *n.children[i]);
      }
      return;
    }
    leaves.push_back(&n);
    if (options_.workload_planner && n.predicted_first && !options_.predicted_first)
      throw std::logic_error("region enabled a disallowed probe order");
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
