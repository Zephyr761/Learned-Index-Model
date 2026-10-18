#include "region_planner.h"
#include "key_bytes.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

namespace blocklearn {
PlanningSamples planning_samples(const std::vector<std::string_view>& keys, bool validation) {
  PlanningSamples result;
  if (keys.empty()) return result;
  const auto count = std::min<std::size_t>(keys.size(), 128);
  std::vector<std::size_t> positions;
  for (std::size_t i = 0; i < count; ++i) positions.push_back(i * keys.size() / count);
  std::mt19937_64 rng(0x424c4b504c414eULL);
  std::shuffle(positions.begin(), positions.end(), rng);
  for (std::size_t i = validation ? 1 : 0; i < count; i += 2) result.hits.push_back(positions[i]);
  if (result.hits.empty()) result.hits.push_back(positions.front());  // singleton only
  const auto prefix = common_prefix(keys.front(), keys.back());
  for (auto id : result.hits) {
    if (result.misses.size() == 16) break;
    for (unsigned trial = 0; trial < 8; ++trial) {
      std::string candidate(keys[id]);
      if (candidate.size() > prefix && trial < 4) {
        const auto offset = prefix + rng() % (candidate.size() - prefix);
        candidate[offset] = char(static_cast<unsigned char>(candidate[offset]) ^ (1U << trial));
      } else candidate.push_back(char(rng()));
      if (!less(candidate, keys.front()) && !less(keys.back(), candidate) &&
          !std::binary_search(keys.begin(), keys.end(), std::string_view(candidate), less)) {
        result.misses.push_back(std::move(candidate)); break;
      }
    }
  }
  return result;
}
namespace {
std::uint16_t fingerprint(const TagStrategy& strategy, const BlockModel& model, std::string_view key) {
  return strategy.kind == TagKind::hash ? hash_tag(key, model.prefix_size()) : strategy.learned.tag(key);
}
QueryWork query_work(const std::vector<std::string_view>& keys, const BlockModel& model,
                    const TagStrategy& strategy, const std::vector<std::uint64_t>* entries,
                    std::string_view key, std::size_t target) {
  QueryWork result;
  if (strategy.kind == TagKind::hash) {
    result.hash_calls = 1;
    result.hash_bytes = key.size() > model.prefix_size() ? key.size() - model.prefix_size() : 0;
  } else {
    result.residual_calls = 1;
    result.feature_reads = strategy.learned.count;  // includes EOS branch/mix work
    std::set<std::size_t> lines;
    for (unsigned i = 0; i < strategy.learned.count; ++i)
      if (strategy.learned.offsets[i] < key.size()) lines.insert(strategy.learned.offsets[i] / 64);
    result.feature_lines = lines.empty() ? 0 : lines.size() - 1;
  }
  // The generated planning misses retain the common region prefix, just like
  // hits, so all planning samples enter the same model inference path.
  if (!model.accepts_prefix(key)) throw std::logic_error("planning query lost region prefix");
  const auto location = model.locate(key);
  const auto lo = location.window.first * model.block_size();
  const auto hi = std::min(keys.size(), location.window.last * model.block_size());
  const auto at = location.predicted * model.block_size();
  const auto end = std::min(keys.size(), at + model.block_size());
  const std::array<std::pair<std::size_t, std::size_t>, 3> ranges = strategy.predicted_first
      ? std::array<std::pair<std::size_t, std::size_t>, 3>{{{at, end}, {lo, at}, {end, hi}}}
      : std::array<std::pair<std::size_t, std::size_t>, 3>{{{lo, hi}, {hi, hi}, {hi, hi}}};
  const auto wanted = fingerprint(strategy, model, key);
  bool found = false;
  for (auto range : ranges) {
    const auto first = range.first, last = range.second;
    if (first == last) continue;
    const bool contains = first <= target && target < last;
    const auto scanned_end = contains ? std::min(last, first + ((target - first) / 16 + 1) * 16) : last;
    const auto records = scanned_end - first;
    result.tag_batches += (records + 15) / 16;
    result.tag_tests += records;
    result.scalar_tags += records % 16;
    if (entries) {
      // Tags after the target lane were loaded in the same SIMD batch, but
      // MUST NOT count as full-key verifications after an early successful hit.
      const auto verified_end = contains ? target + 1 : last;
      const auto token = std::uint64_t(wanted) << 32;
      auto begin = std::lower_bound(entries->begin(), entries->end(), token | first);
      auto finish = std::lower_bound(begin, entries->end(), token | verified_end);
      for (; begin != finish; ++begin) {
        const auto id = std::size_t(*begin & 0xffffffffULL);
        ++result.key_verifications;
        if (keys[id].size() == key.size())
          result.compared_bytes += std::min(key.size(), common_prefix(keys[id], key) + 1);
      }
    } else if (contains) {
      result.key_verifications = 1; result.compared_bytes = key.size();
    }
    if (contains) { found = true; break; }
  }
  if (target != keys.size() && !found) throw std::logic_error("planner omitted certified hit");
  return result;
}
WorkEvaluation evaluate(const std::vector<std::string_view>& keys, const BlockModel& model,
                        const TagStrategy& strategy, const QueryCostWeights& weights,
                        const PlanningSamples& samples, bool collisions) {
  WorkEvaluation result;
  if (keys.empty()) return result;
  if (!model.certified()) throw std::invalid_argument("work evaluation requires a certified model");
  std::vector<std::uint64_t> entries;
  if (collisions) {
    entries.reserve(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i)
      entries.push_back((std::uint64_t(fingerprint(strategy, model, keys[i])) << 32) | i);
    std::sort(entries.begin(), entries.end());
  }
  for (auto id : samples.hits) {
    if (id >= keys.size()) throw std::invalid_argument("planning sample out of bounds");
    result.hits += query_work(keys, model, strategy, collisions ? &entries : nullptr, keys[id], id);
  }
  for (const auto& key : samples.misses)
    result.misses += query_work(keys, model, strategy, collisions ? &entries : nullptr, key, keys.size());
  result.hit_samples = samples.hits.size(); result.miss_samples = samples.misses.size();
  if (result.hit_samples) result.hits = result.hits.scaled(1.0 / result.hit_samples);
  if (result.miss_samples) result.misses = result.misses.scaled(1.0 / result.miss_samples);
  // Frozen prior, never read from the benchmark's miss_ratio/query file.
  result.expected = result.hits.scaled(result.miss_samples ? 0.9 : 1.0);
  if (result.miss_samples) result.expected += result.misses.scaled(0.1);
  result.cost = model.planning_cost(keys.size(), weights, result.expected.tag_batches)
      + result.expected.verification_cost(weights)
      + weights.space_byte * (6.0 + double(sizeof(TagStrategy)) / keys.size());
  return result;
}
TagPlan learn(const std::vector<std::string_view>& keys, const BlockModel& model,
              const QueryCostWeights& weights, const PlanningSamples& train,
              bool first, unsigned max_features) {
  TagStrategy best; best.predicted_first = first;
  auto evaluation = evaluate(keys, model, best, weights, train, true);
  best.learned.cost = evaluation.cost;
  best.learned.candidate_matches = evaluation.expected.key_verifications;
  std::size_t max_length = 0;
  for (auto key : keys) max_length = std::max(max_length, key.size());
  const auto start = model.prefix_size();
  std::set<std::size_t> positions;
  for (std::size_t i = 0; i < 8 && start + i < max_length; ++i) positions.insert(start + i);
  for (std::size_t q = 1; q <= 8 && max_length > start; ++q)
    positions.insert(start + (max_length - start - 1) * q / 8);
  for (unsigned step = 0; step < std::min(max_features, 4U); ++step) {
    auto next = best;
    for (auto position : positions) {
      if (std::find(best.learned.offsets.begin(), best.learned.offsets.begin() + best.learned.count, position)
          != best.learned.offsets.begin() + best.learned.count) continue;
      auto trial = best;
      trial.learned.offsets[trial.learned.count++] = position;
      const auto score = evaluate(keys, model, trial, weights, train, true);
      trial.learned.cost = score.cost;
      trial.learned.candidate_matches = score.expected.key_verifications;
      if (trial.learned.cost < next.learned.cost) next = trial;
    }
    if (next.learned.count == best.learned.count) break;
    best = next;
  }
  return best.learned;
}
std::vector<bool> orders(const RegionPlanningOptions& options) {
  return options.choose_order && options.predicted_first ? std::vector<bool>{false, true}
                                                        : std::vector<bool>{options.predicted_first};
}
}
WorkEvaluation evaluate_query_work(const std::vector<std::string_view>& keys, const BlockModel& model,
    const TagStrategy& strategy, const QueryCostWeights& weights, const PlanningSamples& samples) {
  return evaluate(keys, model, strategy, weights, samples, true);
}
RegionPlan plan_model(const std::vector<std::string_view>& keys, const BlockModel& model,
                      const QueryCostWeights& weights, const RegionPlanningOptions& options) {
  RegionPlan best; best.model = model; best.evaluation.cost = std::numeric_limits<double>::infinity();
  if (!model.certified() || keys.empty()) return best;
  const auto train = planning_samples(keys, false), validation = planning_samples(keys, true);
  for (bool first : orders(options))
    for (TagKind kind : {TagKind::residual, TagKind::hash}) {
      if ((kind == TagKind::hash && !options.allow_hash) || (kind == TagKind::residual && !options.allow_residual)) continue;
      TagStrategy strategy; strategy.kind = kind; strategy.predicted_first = first;
      if (kind == TagKind::residual) strategy.learned = learn(keys, model, weights, train, first, options.max_features);
      const auto score = evaluate(keys, model, strategy, weights, validation, true);
      if (score.cost < best.evaluation.cost) {
        best.strategy = strategy; best.evaluation = score;
      }
    }
  if (!std::isfinite(best.evaluation.cost)) throw std::invalid_argument("planner has no allowed tag strategy");
  return best;
}
RegionPlan plan_region(const std::vector<std::string_view>& keys, const QueryCostWeights& weights,
                       const RegionPlanningOptions& options) {
  weights.validate();
  std::vector<std::pair<double, BlockModel>> candidates;
  const auto samples = planning_samples(keys, false);
  for (std::size_t b : {16U, 32U, 64U})
    for (unsigned bytes : {2U, 4U, 7U})
      for (std::size_t s = 1; s <= options.max_segments; s *= 2) {
        auto model = BlockModel::fit(keys, b, bytes, s, options.max_block_error);
        if (!model.certified()) continue;
        double lower_cost = std::numeric_limits<double>::infinity();
        for (bool first : orders(options))
          for (auto kind : {TagKind::residual, TagKind::hash}) {
            if ((kind == TagKind::hash && !options.allow_hash) || (kind == TagKind::residual && !options.allow_residual)) continue;
            TagStrategy strategy; strategy.kind = kind; strategy.predicted_first = first;
            lower_cost = std::min(lower_cost, evaluate(keys, model, strategy, weights, samples, false).cost);
          }
        candidates.emplace_back(lower_cost, std::move(model));
      }
  if (candidates.empty()) {
    RegionPlan result;
    result.model = BlockModel::fit(keys, 32, 7, options.max_segments, options.max_block_error);
    if (result.model.certified() && !keys.empty()) return plan_model(keys, result.model, weights, options);
    return result;  // exact search or caller splits; no unverified fast path
  }
  std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  RegionPlan best; best.evaluation.cost = std::numeric_limits<double>::infinity();
  // Explicit bounded planning budget: exact strategy evaluation of four models,
  // not a claim of finding the global optimum over every model/tag combination.
  for (std::size_t i = 0; i < std::min<std::size_t>(4, candidates.size()); ++i) {
    auto trial = plan_model(keys, candidates[i].second, weights, options);
    if (trial.evaluation.cost < best.evaluation.cost) best = std::move(trial);
  }
  return best;
}
}  // namespace blocklearn
