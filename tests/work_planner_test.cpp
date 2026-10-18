#include "learned_index.h"
#include "key_bytes.h"
#include "tag_probe.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

using namespace blocklearn;
using namespace string_index_bench::v1;
static void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
static bool same(double a, double b) { return std::abs(a - b) < 1e-8; }
static std::vector<std::string_view> views(const std::vector<Row>& rows) {
  std::vector<std::string_view> keys;
  for (const auto& row : rows) keys.push_back(row.key);
  return keys;
}
static QueryCostWeights weights() {
  QueryCostWeights result;
  result.hash_base = result.hash_byte = .05;
  result.residual_base = 1; result.feature_read = 2; result.feature_line = .01;
  result.verify_byte = .001; result.space_byte = result.encode_byte = result.segment_step = .01;
  return result;
}
static void costs_and_batches() {
  CostProfile profile; profile.weights = weights(); profile.seed = 7; profile.iterations = 1024;
  profile.origin = "independent-synthetic-v1";
  std::ostringstream output; write_cost_profile(output, profile);
  std::istringstream input(output.str()); const auto restored = read_cost_profile(input);
  check(restored.seed == 7 && same(restored.weights.hash_byte, .05), "cost profile round trip");
  for (const auto& bad : {output.str() + "hash_byte 2\n", output.str() + "unknown 3\n",
                          std::string("blocklearn-query-cost/v1\norigin benchmark-queries\n")}) {
    bool rejected = false; try { std::istringstream stream(bad); read_cost_profile(stream); }
    catch (const std::exception&) { rejected = true; }
    check(rejected, "invalid/missing/duplicate/non-independent profile accepted");
  }
  std::vector<Row> rows;
  for (unsigned i = 0; i < 128; ++i) rows.push_back({"key/" + std::to_string(10000 + i), i});
  const auto keys = views(rows);
  const auto small = BlockModel::fit(keys, 16, 2, 1, 16);
  const auto large = BlockModel::fit(keys, 64, 2, 1, 16);
  check(large.average_blocks() < small.average_blocks() && large.average_tag_batches() > small.average_tag_batches(),
        "block count still confused with real 16-tag batches");
  const auto model = BlockModel::fit(keys, 64, 7, 1, 16);
  TagStrategy strategy; // all keys same length => deliberately all tags collide
  PlanningSamples sample; sample.hits = {0};
  const auto first = evaluate_query_work(keys, model, strategy, weights(), sample);
  check(first.hits.key_verifications == 1 && first.hits.tag_tests == 16 && first.hits.tag_batches == 1,
        "collisions after successful lane counted as key verifications");
  auto train = planning_samples(keys, false), validation = planning_samples(keys, true);
  for (auto id : train.hits)
    check(std::find(validation.hits.begin(), validation.hits.end(), id) == validation.hits.end(),
          "feature fit and strategy selection samples overlap");
  check(!validation.misses.empty(), "independent local misses were not generated");
  for (const auto& miss : validation.misses)
    check(!std::binary_search(keys.begin(), keys.end(), std::string_view(miss), less) &&
          !less(miss, keys.front()) && !less(keys.back(), miss), "planning miss is not absent/in region");
}
static void exact_work_counts() {
  std::vector<Row> rows;
  for (unsigned i = 0; i < 173; ++i) rows.push_back({"common/" + std::to_string(10000 + i), i});
  const auto keys = views(rows);
  const auto w = weights();
  RegionPlanningOptions settings;
  const auto planned = plan_region(keys, w, settings);
  Options options; options.verification = Verification::adaptive; options.workload_planner = true;
  options.predicted_first = true; options.cost_weights = w; options.instrument = true;
  LearnedIndex index(options); index.bulk_load(rows);
  const auto samples = planning_samples(keys, true);
  const auto score = evaluate_query_work(keys, planned.model, planned.strategy, w, samples);
  auto before = index.statistics();
  for (auto id : samples.hits) check(index.find(keys[id]) == rows[id].value, "planned hit lost");
  auto after = index.statistics();
  check(same(double(after.tag_tests - before.tag_tests), score.hits.tag_tests * samples.hits.size()) &&
        same(double(after.tag_batches - before.tag_batches), score.hits.tag_batches * samples.hits.size()) &&
        same(double(after.full_key_comparisons - before.full_key_comparisons), score.hits.key_verifications * samples.hits.size()),
        "simulated hit work differs from actual SIMD/verification path");
  before = after;
  for (const auto& key : samples.misses) check(!index.find(key), "planned miss became hit");
  after = index.statistics();
  check(same(double(after.tag_tests - before.tag_tests), score.misses.tag_tests * samples.misses.size()) &&
        same(double(after.tag_batches - before.tag_batches), score.misses.tag_batches * samples.misses.size()) &&
        same(double(after.full_key_comparisons - before.full_key_comparisons), score.misses.key_verifications * samples.misses.size()),
        "simulated miss work differs from full candidate scan");
  index.validate();
}
static void all_orders_and_tails() {
  std::vector<Row> rows;
  for (unsigned i = 0; i < 173; ++i) rows.push_back({"prefix/" + std::to_string(10000 + i), i});
  const auto keys = views(rows);
  for (std::size_t block : {16U, 32U, 64U})
   for (auto kind : {TagKind::hash, TagKind::residual})
    for (bool first : {false, true}) {
      const auto model = BlockModel::fit(keys, block, 7, 4, 16);
      TagStrategy strategy; strategy.kind = kind; strategy.predicted_first = first;
      std::vector<std::uint16_t> tags;
      for (auto key : keys) tags.push_back(kind == TagKind::hash ? hash_tag(key, model.prefix_size()) : strategy.learned.tag(key));
      for (std::size_t target : {0U, 15U, 16U, 31U, 78U, 160U, 172U}) {
        PlanningSamples sample; sample.hits = {target};
        const auto simulated = evaluate_query_work(keys, model, strategy, weights(), sample).hits;
        const auto predicted = model.locate(keys[target]);
        const auto lo = predicted.window.first * block, hi = std::min(keys.size(), predicted.window.last * block);
        const auto at = predicted.predicted * block, end = std::min(keys.size(), at + block);
        std::vector<std::pair<std::size_t, std::size_t>> ranges = first
            ? std::vector<std::pair<std::size_t, std::size_t>>{{at, end}, {lo, at}, {end, hi}}
            : std::vector<std::pair<std::size_t, std::size_t>>{{lo, hi}};
        QueryWork reference; bool found = false;
        for (auto range : ranges) {
          for (auto i = range.first; i < range.second && !found; i += 16) {
            const auto count = std::min<std::size_t>(16, range.second - i);
            ++reference.tag_batches; reference.tag_tests += count;
            if (count != 16) reference.scalar_tags += count;
            auto mask = tag_mask(tags.data() + i, count, tags[target]);
            while (mask) {
              const auto lane = unsigned(__builtin_ctz(mask));
              ++reference.key_verifications;
              if (i + lane == target) { found = true; break; }
              mask &= mask - 1;
            }
          }
          if (found) break;
        }
        check(found && simulated.tag_tests == reference.tag_tests && simulated.tag_batches == reference.tag_batches &&
              simulated.scalar_tags == reference.scalar_tags && simulated.key_verifications == reference.key_verifications,
              "simulation diverged for a scan order/block size/partial tail");
      }
    }
}
static void region_policies_and_retag() {
  auto w = weights();
  std::vector<Row> rows;
  for (unsigned i = 1; i <= 64; ++i) rows.push_back({std::string("A") + char(i), i % 2 ? i : 0});
  for (unsigned i = 1; i <= 64; ++i) rows.push_back({std::string("B") + char(i) + std::string(200, 'x'), i});
  Options options; options.verification = Verification::adaptive; options.workload_planner = true;
  options.short_directory = options.predicted_first = true; options.cost_weights = w;
  options.region_size = 64; options.directory_fanout = 4; options.delta_capacity = 4;
  LearnedIndex index(options); index.bulk_load(rows);
  const auto initial = index.statistics();
  check(initial.hash_leaves > 0 && initial.residual_leaves > 0, "regions failed to choose different tag kinds");
  for (const auto& row : rows) check(index.find(row.key) == row.value, "mixed-mode region hit lost");
  index.validate();
  std::vector<Row> base(rows.begin(), rows.begin() + 32);
  options.region_size = 128;
  LearnedIndex changing(options); changing.bulk_load(base);
  check(changing.statistics().hash_leaves == 1, "retag fixture did not start as hash");
  for (unsigned i = 65; i < 129; ++i) {
    auto key = std::string("A") + char(i) + std::string(200, 'x');
    check(changing.insert(key, i) == InsertResult::inserted, "retag insert failed");
    check(changing.insert(key, 999) == InsertResult::already_exists && changing.find(key) == i, "retag duplicate overwrote");
    for (const auto& row : base) check(changing.find(row.key) == row.value, "retag lost old base hit");
    changing.validate();
  }
  check(changing.statistics().leaves == 1 && changing.statistics().residual_leaves == 1,
        "hash-to-residual reconfiguration not exercised");
  // Dynamic binary keys, shortened prefixes, and leaf/directory splits.
  for (unsigned i = 0; i < 240; ++i) {
    const std::string key = i % 2 ? std::string(i / 2, '\0') : std::string(1, char(255)) + std::to_string(i);
    check(index.insert(key, i) == InsertResult::inserted, "adaptive binary insert");
    check(index.find(key) == i, "adaptive delta/binary hit lost");
    if (i % 47 == 0) index.validate();
  }
  check(index.statistics().leaf_splits && index.statistics().directory_splits, "adaptive multi-level splits not exercised");
  index.validate();
  std::vector<Row> empty;
  LearnedIndex fresh(options); fresh.bulk_load(empty);
  check(fresh.insert("", 0) == InsertResult::inserted && fresh.find("") == 0, "adaptive empty/zero semantics");
}
int main() {
  try {
    costs_and_batches(); exact_work_counts(); all_orders_and_tails(); region_policies_and_retag();
    std::cout << "calibrated costs, real batches/early return, split samples, mixed labels, retagging and dynamic semantics passed\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
