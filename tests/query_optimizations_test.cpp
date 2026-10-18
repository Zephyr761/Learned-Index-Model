#include "learned_index.h"
#include "benchmark/model_registry.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>

using namespace blocklearn;
using namespace string_index_bench::v1;
static void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
struct ByteLess {
  bool operator()(std::string_view a, std::string_view b) const { return less(a, b); }
};
using Oracle = std::map<std::string, Value, ByteLess>;
static std::optional<Value> expected(const Oracle& oracle, std::string_view key) {
  auto it = oracle.find(std::string(key));
  return it == oracle.end() ? std::optional<Value>{} : it->second;
}
static void short_routes() {
  std::mt19937_64 rng(20261018);
  std::vector<std::vector<std::string>> fixtures{
    {}, {"only-child"}, {"aa", "cc", "ee", "gg"},
    {"", std::string(1, '\0'), std::string("a\0", 2), "aa", std::string(1, char(255))},
    {"prefix/a0000000X", "prefix/a0000000Y", "prefix/b0000000Z"},
    {std::string(300, 'p'), std::string(300, 'p') + "a", std::string(300, 'p') + "b"}};
  for (unsigned trial = 0; trial < 30; ++trial) {
    Oracle unique;
    for (unsigned i = 0; i < 96; ++i) {
      std::string key = trial % 2 ? std::string(50, 'p') : "";
      for (unsigned j = 0, n = rng() % 24; j < n; ++j) key.push_back(char(rng()));
      unique.emplace(key, 0);
    }
    std::vector<std::string> keys;
    for (const auto& entry : unique) keys.push_back(entry.first);
    fixtures.push_back(std::move(keys));
  }
  bool saw_collision_comparison = false;
  for (auto& stored : fixtures) {
    std::sort(stored.begin(), stored.end(), ByteLess{});
    std::vector<std::string_view> fences;
    for (const auto& key : stored) fences.push_back(key);
    ShortDirectory directory; directory.rebuild(fences);
    check(directory.valid_for(fences), "short directory invalid after build");
    auto queries = stored;
    queries.insert(queries.end(), {"", "!", "dd", "prefix/a0000000W", "prefix/a0000000Z", "prefix/a0000000X-extra"});
    for (unsigned j = 0; j < 300; ++j) {
      std::string key;
      for (unsigned k = 0, n = rng() % 40; k < n; ++k) key.push_back(char(rng()));
      queries.push_back(std::move(key));
    }
    for (const auto& key : queries) {
      std::size_t comparisons = 0;
      const auto got = directory.predecessor(fences, key,
          [&](std::string_view a, std::string_view b) { ++comparisons; return less(a, b); });
      if (!directory.useful()) { check(!got, "disabled directory produced a route"); continue; }
      const auto at = std::upper_bound(fences.begin(), fences.end(), std::string_view(key), ByteLess{}) - fences.begin();
      check(got && *got == (at ? std::size_t(at - 1) : 0), "short routing changed predecessor");
      saw_collision_comparison = saw_collision_comparison || comparisons > 0;
      if (stored == std::vector<std::string>{"aa", "cc", "ee", "gg"} && key == "dd")
        check(comparisons == 0, "distinct short codes still read full fences");
    }
  }
  check(saw_collision_comparison, "equal-code full comparison not exercised");
  std::vector<std::string> poor;
  for (unsigned i = 0; i < 64; ++i)
    poor.push_back(std::string(i < 32 ? "a" : "b") + std::string(12, 'x') + char(i));
  std::vector<std::string_view> fences;
  for (const auto& key : poor) fences.push_back(key);
  ShortDirectory directory; directory.rebuild(fences);
  check(!directory.useful() && directory.valid_for(fences), "low-discrimination fallback missing");
  std::vector<std::string_view> changed{"!new-min", "other"};
  check(!directory.valid_for(changed), "stale metadata not detected");
  directory.rebuild(changed);
  check(directory.valid_for(changed), "metadata not refreshed");
}
static void audit(LearnedIndex& index, const Oracle& oracle, std::mt19937_64& rng) {
  for (const auto& [key, value] : oracle) check(index.find(key) == value, "lost optimized hit");
  for (unsigned i = 0; i < 180; ++i) {
    std::string key;
    for (unsigned j = 0, n = rng() % 35; j < n; ++j) key.push_back(char(rng()));
    check(index.find(key) == expected(oracle, key), "optimized false positive/negative");
  }
  for (std::string lower : {std::string(), std::string("shared/"), std::string(1, char(255))}) {
    auto scan = index.scan(lower, 23);
    auto at = oracle.lower_bound(lower);
    std::size_t i = 0;
    for (; at != oracle.end() && i < 23; ++at, ++i)
      check(i < scan.size() && scan[i].key == at->first && scan[i].value == at->second, "optimized scan mismatch");
    check(scan.size() == i, "optimized scan size");
    Value sum = 0;
    const std::string upper(2, char(255));
    for (at = oracle.lower_bound(lower); at != oracle.end() && !less(upper, at->first); ++at) sum += at->second;
    check(index.range_sum(lower, upper) == sum, "optimized range sum");
  }
  LookupProfile profile;
  check(index.profile_find("", profile) == expected(oracle, ""), "optimized profiling differs");
  double sum = 0; for (const auto& s : profile.stages) sum += s.ns;
  check(std::abs(sum - profile.total_ns) < .01, "optimized profile double counted");
  index.validate();
}
static void dynamic_semantics() {
  std::vector<Row> rows;
  for (unsigned i = 0; i < 600; ++i)
    rows.push_back({"shared/" + std::to_string(100000 + i), i % 5 ? i : 0});
  for (auto verification : {Verification::hash, Verification::residual})
   for (unsigned flags = 0; flags < 4; ++flags)
    for (unsigned features : {0U, 2U}) {
      Options options; options.verification = verification;
      options.short_directory = flags & 1; options.predicted_first = flags & 2;
      options.region_size = 64; options.directory_fanout = 4;
      options.delta_capacity = 4; options.tag_features = features;
      LearnedIndex index(options); index.bulk_load(rows);
      Oracle oracle;
      for (const auto& row : rows) oracle.emplace(row.key, row.value);
      std::mt19937_64 rng(19);
      audit(index, oracle, rng);
      // Shorten the first directory prefixes, then force local merges, leaf
      // splits and upward directory splits. Include arbitrary byte keys.
      for (unsigned i = 0; i < 220; ++i) {
        std::string key = i % 3 ? "shared/new/" + std::to_string(i) : std::string(i / 3, '\0');
        if (i % 7 == 0) key = std::string(1, char(255)) + std::to_string(i);
        const Value value = i % 5 ? i : UINT64_MAX;
        const bool fresh = oracle.emplace(key, value).second;
        check(index.insert(key, value) == (fresh ? InsertResult::inserted : InsertResult::already_exists), "optimized insert result");
        check(index.insert(key, 123) == InsertResult::already_exists, "optimized duplicate");
        check(index.find(key) == oracle.at(key), "duplicate overwrote value");
        if (i % 53 == 0) audit(index, oracle, rng);
      }
      audit(index, oracle, rng);
      const auto stats = index.statistics();
      check(stats.merges && stats.leaf_splits && stats.directory_splits, "optimized structural maintenance not exercised");
    }
}
static void probe_work() {
  std::vector<Row> rows;
  for (unsigned i = 0; i < 1024; ++i)
    rows.push_back({"prefix/" + std::to_string(100000 + i), i});
  Options options; options.instrument = true; options.tag_features = 0;
  options.region_size = 1024;
  LearnedIndex control(options); control.bulk_load(rows);
  options.predicted_first = true;
  LearnedIndex first(options); first.bulk_load(rows);
  bool saved_batch = false;
  for (const auto& row : rows) {
    auto before_c = control.statistics(), before_f = first.statistics();
    check(control.find(row.key) == first.find(row.key), "probe hit mismatch");
    saved_batch = saved_batch || first.statistics().tag_tests - before_f.tag_tests <
                                 control.statistics().tag_tests - before_c.tag_tests;
  }
  check(saved_batch, "prediction-first did not skip any preceding tag batch");
  for (unsigned i = 0; i < rows.size(); i += 11) {
    auto key = rows[i].key; key.back() = ':';  // same length/prefix, guaranteed absent
    auto before_c = control.statistics(), before_f = first.statistics();
    check(!control.find(key) && !first.find(key), "probe miss became hit");
    check(first.statistics().tag_tests - before_f.tag_tests ==
          control.statistics().tag_tests - before_c.tag_tests, "miss skipped/repeated candidate tags");
  }
}
int main() {
  try {
    short_routes(); dynamic_semantics(); probe_work();
    for (const auto& model : bench::model_factories()) {
      auto index = model.factory({});
      check(index->descriptor().implementation == model.id, "variant identity lost optimization flags");
    }
    bool rejected = false;
    try { make_block_hash_shortdir({{"short_directory", "0"}}); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "fixed ablation flags silently overridden");
    rejected = false;
    try { make_block_hash({{"predicted_first", "2"}}); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "invalid optimization parameter accepted");
    std::cout << "short-code collisions/EOS/extremes, all switch combinations, inserts/splits, exhaustive misses and probe savings passed\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
