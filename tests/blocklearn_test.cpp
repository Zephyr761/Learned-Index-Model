#include "learned_index.h"
#include "benchmark/factories.h"
#include "key_bytes.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>

using namespace string_index_bench::v1;
struct ByteLess {
  bool operator()(const std::string& a, const std::string& b) const { return blocklearn::less(a, b); }
};
using Oracle = std::map<std::string, Value, ByteLess>;
static void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
static std::vector<Row> rows_of(const Oracle& oracle) {
  std::vector<Row> rows;
  for (const auto& [key, value] : oracle) rows.push_back({key, value});
  return rows;
}
static void audit(Index& index, const Oracle& oracle, std::mt19937_64& rng) {
  for (const auto& [key, value] : oracle) check(index.find(key) == value, "lost/wrong lookup");
  for (unsigned i = 0; i < 200; ++i) {
    std::string key(rng() % 40, '\0');
    for (char& c : key) c = char(rng());
    auto it = oracle.find(key);
    auto expected = it == oracle.end() ? std::optional<Value>{} : it->second;
    check(index.find(key) == expected, "false positive/negative");
  }
  for (std::string lower : {std::string(), std::string("a"), std::string("https://"), std::string(1, char(255))}) {
    auto got = index.scan(lower, 47);
    auto it = oracle.lower_bound(lower);
    std::size_t j = 0;
    for (; it != oracle.end() && j < 47; ++it, ++j)
      check(j < got.size() && got[j].key == it->first && got[j].value == it->second, "scan mismatch");
    check(got.size() == j, "scan size mismatch");
    Value sum = 0;
    const std::string upper(2, char(255));
    for (it = oracle.lower_bound(lower); it != oracle.end() && !ByteLess{}(upper, it->first); ++it)
      sum += it->second;
    check(index.range_sum(lower, upper) == sum, "range_sum mismatch");
    check(index.range_sum("z", "a") == 0, "reversed range");
  }
  check(index.scan("", 0).empty(), "zero scan");
  if (auto* own = dynamic_cast<blocklearn::LearnedIndex*>(&index)) own->validate();
}
int main() {
  try {
    std::mt19937_64 rng(42);
    Oracle oracle;
    for (std::string key : {std::string(), std::string("a"), std::string("aa"),
         std::string("a\0", 2), std::string("a\0b", 3), std::string(1, char(128)),
         std::string(1, char(255)), std::string(600, 'x')}) oracle[key] = rng();
    for (unsigned i = 0; i < 1600; ++i) {
      std::string key = i % 2 ? "https://example.org/common/prefix/" : "";
      for (unsigned j = 0, n = 1 + rng() % 40; j < n; ++j) key.push_back(char(rng()));
      oracle[key] = rng();
    }
    auto rows = rows_of(oracle);
    for (Factory factory : {make_block_plain, make_block_hash, make_block_residual, make_block_residual_delta}) {
      auto index = factory({{"region_size", "64"}, {"directory_fanout", "4"}, {"delta_capacity", "4"}});
      index->bulk_load(rows);
      audit(*index, oracle, rng);
      check(index->find(rows.front().key).has_value(), "zero value existence");
      if (factory != make_block_residual_delta) {
        bool threw = false;
        try { index->insert("new", 7); } catch (const UnsupportedOperation&) { threw = true; }
        check(threw, "static insert must throw");
        continue;
      }
      Oracle changed = oracle;
      for (unsigned i = 0; i < 320; ++i) {
        std::string key = i % 3 ? "https://example.org/new/" + std::to_string(i)
            : std::string(i / 3 + 1, '\0');
        const auto value = i % 7 == 0 ? 0 : rng();
        const bool inserted = changed.emplace(key, value).second;
        check(index->insert(key, value) == (inserted ? InsertResult::inserted : InsertResult::already_exists), "insert result");
        check(index->insert(key, 123) == InsertResult::already_exists, "duplicate insert");
        check(index->find(key) == changed.at(key), "duplicate overwrote value");
        if (i % 31 == 0) audit(*index, changed, rng);
      }
      audit(*index, changed, rng);
      auto* own = dynamic_cast<blocklearn::LearnedIndex*>(index.get());
      const auto s = own->statistics();
      check(s.height >= 3 && s.merges && s.leaf_splits && s.directory_splits, "structural operations not exercised");
    }
    for (Factory factory : {make_block_plain, make_block_hash, make_block_residual, make_block_residual_delta}) {
      std::vector<Row> empty;
      auto index = factory({}); index->bulk_load(empty);
      check(!index->find("") && index->scan("", 3).empty(), "empty semantics");
      if (factory == make_block_residual_delta) {
        check(index->insert("", 0) == InsertResult::inserted, "empty key insert");
        check(index->find("") == 0, "zero payload");
        for (unsigned i = 0; i < 180; ++i) index->insert(std::to_string(i), i);
        dynamic_cast<blocklearn::LearnedIndex*>(index.get())->validate();
      }
    }
    bool bad = false;
    try { make_block_residual({{"unknown", "1"}}); } catch (const std::invalid_argument&) { bad = true; }
    check(bad, "unknown parameters accepted");
    std::cout << "blocklearn: binary keys, all ablations, misses, scans, merges, multi-level splits passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
