#include "learned_index.h"
#include "key_bytes.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace blocklearn;
using Row = LearnedIndex::Row;
static void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
static const StageTime& stage(const LookupProfile& p, LookupStage s) {
  return p.stages[static_cast<std::size_t>(s)];
}
static void conserved(const LookupProfile& p) {
  double sum = 0;
  for (auto s : p.stages) { check(s.ns >= 0, "negative time"); sum += s.ns; }
  check(std::abs(sum - p.total_ns) < 0.01, "nested stages double-counted/lost time");
}
int main() {
  try {
    std::vector<Row> rows{{"", 0}, {std::string("a\0b", 3), std::numeric_limits<std::uint64_t>::max()},
        {std::string(1, char(255)), 9}, {std::string(600, 'x'), 7}};
    for (unsigned i = 0; i < 500; ++i)
      rows.push_back({"https://example.org/" + std::to_string(100000 + i), i});
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return less(a.key, b.key); });
    for (auto verification : {Verification::full_key, Verification::hash, Verification::residual}) {
      Options o; o.verification = verification; o.instrument = true;
      o.region_size = 64; o.directory_fanout = 4; o.delta_capacity = 4;
      LearnedIndex index(o); index.bulk_load(rows);
      LookupProfile profile;
      for (const auto& row : rows) {
        check(index.profile_find(row.key, profile) == index.find(row.key), "profile changed hit result");
      }
      for (auto key : {std::string("absent"), std::string(2, char(255)), std::string("https://example.org/100001-missing")})
        check(index.profile_find(key, profile) == index.find(key), "profile changed miss result");
      check(profile.queries == rows.size() + 3, "profile query count");
      check(stage(profile, LookupStage::directory_route).calls > 0, "directory not measured");
      check(stage(profile, LookupStage::leaf_model).calls > 0, "leaf model not measured");
      check(stage(profile, LookupStage::key_verify).calls > 0, "key verification not measured");
      check(stage(profile, LookupStage::delta_scan).calls == 0, "empty delta measured");
      if (verification == Verification::full_key) {
        check(stage(profile, LookupStage::tag_generate).calls == 0, "plain generated tags");
        check(stage(profile, LookupStage::tag_filter).calls == 0, "plain filtered tags");
      } else {
        check(stage(profile, LookupStage::tag_generate).calls == profile.queries, "missing tag generation");
        check(stage(profile, LookupStage::tag_filter).calls > 0, "tag filter not measured");
      }
      conserved(profile);
      // A prefix-changing insertion is still checked independently of the base
      // model. Its exact key verification must be exclusive of delta scanning.
      const std::string inserted("\0new", 4);
      index.insert(inserted, 0);
      LookupProfile delta;
      check(index.profile_find(inserted, delta) == 0, "profile lost delta key/zero value");
      check(stage(delta, LookupStage::delta_scan).calls == 1, "delta not measured");
      check(stage(delta, LookupStage::key_verify).calls == 1, "delta key check not measured");
      check(stage(delta, LookupStage::leaf_model).calls == 0, "delta hit unnecessarily predicted base");
      conserved(delta);
      for (unsigned i = 0; i < 100; ++i) index.insert("new-" + std::to_string(i), i);
      LookupProfile merged;
      for (unsigned i = 0; i < 100; ++i)
        check(index.profile_find("new-" + std::to_string(i), merged) == i, "profile lost merged key");
      conserved(merged); index.validate();
    }
    std::vector<Row> common;
    for (unsigned i = 0; i < 400; ++i)
      common.push_back({"https://common/" + std::to_string(100000 + i), i});
    Options fallback_options; fallback_options.region_size = 64;
    LearnedIndex fallback_index(fallback_options); fallback_index.bulk_load(common);
    LookupProfile fallback;
    check(!fallback_index.profile_find("!", fallback), "fallback false positive");
    check(stage(fallback, LookupStage::exact_fallback).calls > 0, "directory fallback not measured");
    conserved(fallback);
    std::vector<Row> empty;
    LearnedIndex index; index.bulk_load(empty);
    LookupProfile profile;
    check(!index.profile_find("", profile), "empty profile semantics");
    check(profile.queries == 1 && stage(profile, LookupStage::directory_route).calls == 0,
          "single leaf incorrectly measured directory");
    conserved(profile);
    std::cout << "profile/ordinary semantic equivalence, exclusive stages, binary keys, delta and merges passed\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
