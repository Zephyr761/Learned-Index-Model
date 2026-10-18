// This test MUST fail on the unavailable adapter; it never accepts a stub as HOT.
#include "benchmark/factories.h"
#include "key_bytes.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>

using namespace string_index_bench::v1;
struct ByteLess {
  bool operator()(std::string_view a, std::string_view b) const { return blocklearn::less(a, b); }
};
using Oracle = std::map<std::string, Value, ByteLess>;
static void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template<class Exception, class Operation>
static void rejects(Operation operation, const char* message) {
  try { operation(); }
  catch (const Exception&) { return; }
  throw std::runtime_error(message);
}
static void answer(Index& index, const Oracle& oracle, const std::string& key) {
  const auto found = oracle.find(key);
  const std::optional<Value> expected = found == oracle.end()
      ? std::optional<Value>{} : std::optional<Value>{found->second};
  check(index.find(key) == expected, "HOT lookup differs from unsigned-byte oracle");
  // The query is not a C string at the end of this view. The adapter must obey
  // string_view length rather than reading its backing string's suffix.
  const auto backing = "not-the-key:" + key + "/trailing-bytes";
  check(index.find(std::string_view(backing).substr(12, key.size())) == expected,
        "HOT ignored query string_view length");
}
static void ordered(Index& index, const Oracle& oracle, const std::string& lower) {
  for (std::size_t limit : {0U, 1U, 17U, 5000U}) {
    const auto actual = index.scan(lower, limit);
    auto expected = oracle.lower_bound(lower);
    std::size_t i = 0;
    for (; expected != oracle.end() && i < limit; ++expected, ++i)
      check(i < actual.size() && actual[i].key == expected->first && actual[i].value == expected->second,
            "HOT lower_bound/scan differs from oracle");
    check(actual.size() == i, "HOT scan limit/end mismatch");
  }
  for (const auto& upper : {std::string(), std::string("a"), std::string("shared/"),
                            std::string(254, char(255))}) {
    Value sum = 0;
    if (!blocklearn::less(upper, lower))
      for (auto at = oracle.lower_bound(lower); at != oracle.end() && !blocklearn::less(upper, at->first); ++at)
        sum += at->second;
    check(index.range_sum(lower, upper) == sum, "HOT inclusive range_sum/modulo mismatch");
  }
}
static void rejected_keys(Index& index) {
  for (const auto& bad : {std::string("a\0b", 3), std::string(255, 'p')}) {
    rejects<std::invalid_argument>([&] { index.find(bad); }, "HOT accepted invalid lookup key");
    rejects<std::invalid_argument>([&] { index.insert(bad, 7); }, "HOT accepted invalid insertion key");
    rejects<std::invalid_argument>([&] { index.scan(bad, 1); }, "HOT accepted invalid scan bound");
    rejects<std::invalid_argument>([&] { index.range_sum(bad, ""); }, "HOT accepted invalid reversed lower bound");
    rejects<std::invalid_argument>([&] { index.range_sum("", bad); }, "HOT accepted invalid upper bound");
  }
}
int main() {
  try {
    auto index = make_hot_baseline({});
    const auto descriptor = index->descriptor();
    const auto& capabilities = descriptor.capabilities;
    check(descriptor.implementation == "hot-single-threaded" &&
          capabilities.bulk_load != Support::unsupported && capabilities.find == Support::native &&
          capabilities.insert == Support::native, "native HOT is required: unavailable adapter is NOT a pass");
    check(!capabilities.keys.allows_nul && capabilities.keys.max_bytes == 254 &&
          capabilities.keys.allows_high_bit && capabilities.keys.allows_prefix_pairs &&
          capabilities.keys.min_bytes == 0 && capabilities.max_value == UINT64_MAX,
          "HOT descriptor does not match C-string adapter domain");
    rejects<std::invalid_argument>([] { make_hot_baseline({{"unsupported", "1"}}); }, "HOT ignored factory parameters");
    rejects<std::logic_error>([&] { index->find(""); }, "HOT lookup before bulk_load accepted");

    Oracle oracle{{"", 0}, {"a", UINT64_MAX}, {"aa", 2}, {"ab", 3}, {"b", 4}};
    std::mt19937_64 rng(20261018);
    for (unsigned i = 0; i < 1200; ++i) {
      std::string key;
      for (unsigned j = 0, count = 1 + rng() % 80; j < count; ++j)
        key.push_back(char(1 + rng() % 255));
      oracle.emplace(key, rng());
    }
    // Long prefix pairs and a distinction in the last allowed payload byte.
    oracle.emplace(std::string(253, 'p'), 6);
    oracle.emplace(std::string(254, 'p'), 0);
    oracle.emplace(std::string(253, 'p') + 'q', UINT64_MAX);
    for (unsigned i = 0; i < 256; ++i)
      oracle.emplace("shared/prefix/" + std::to_string(100000 + i), i);
    std::vector<Row> rows;
    for (const auto& [key, value] : oracle) rows.push_back({key, value});
    index->bulk_load(rows);
    rejects<std::logic_error>([&] { index->bulk_load(rows); }, "HOT bulk_load twice accepted");
    for (const auto& [key, value] : oracle) answer(*index, oracle, key);
    rejected_keys(*index);
    rejects<UnsupportedOperation>([&] { index->update("a", 9); }, "HOT update did not reject");
    rejects<UnsupportedOperation>([&] { index->erase("a"); }, "HOT erase did not reject");
    for (unsigned i = 0; i < 1400; ++i) {
      std::string temporary = "new/shared/" + std::to_string(100000 + i);
      const auto stable = temporary;
      const Value value = i % 3 ? rng() : 0;
      check(index->insert(temporary, value) == InsertResult::inserted, "HOT fresh insert result");
      oracle.emplace(stable, value);
      temporary.assign(temporary.size(), 'x');  // inserted key must be owned
      check(index->insert(stable, value + 1) == InsertResult::already_exists, "HOT duplicate result");
      answer(*index, oracle, stable);
    }
    for (const auto& [key, value] : oracle) answer(*index, oracle, key);
    for (unsigned i = 0; i < 500; ++i) {
      std::string absent;
      for (unsigned j = 0, count = rng() % 100; j < count; ++j) absent.push_back(char(1 + rng() % 255));
      answer(*index, oracle, absent);
    }
    for (const auto& lower : {std::string(), std::string("a"), std::string("aaa"),
                              std::string("new/shared/100050"), std::string(1, char(128)),
                              std::string(254, char(255))}) ordered(*index, oracle, lower);
    const auto memory = index->memory_usage();
    check(memory.index_owned_bytes.has_value() && memory.adapter_owned_bytes.has_value() &&
          memory.retained_input_bytes.has_value(), "HOT memory partitions unavailable");

    std::vector<Row> empty;
    auto singleton = make_hot_baseline({}); singleton->bulk_load(empty);
    check(!singleton->find("") && singleton->scan("", 1).empty() && singleton->range_sum("", "z") == 0,
          "HOT empty tree semantics");
    check(singleton->memory_usage().index_owned_bytes == 0, "HOT empty memory handling");
    check(singleton->insert("", 0) == InsertResult::inserted && singleton->find("") == 0,
          "HOT empty-to-singleton transition");
    check(singleton->insert("a", UINT64_MAX) == InsertResult::inserted && singleton->range_sum("", "a") == UINT64_MAX,
          "HOT singleton-to-node transition");
    for (const std::vector<Row>& invalid : {std::vector<Row>{{"a", 1}, {"a", 2}},
          std::vector<Row>{{"b", 1}, {"a", 2}}, std::vector<Row>{{std::string("a\0b", 3), 1}},
          std::vector<Row>{{std::string(255, 'p'), 1}}}) {
      auto fresh = make_hot_baseline({});
      rejects<std::invalid_argument>([&] { fresh->bulk_load(invalid); }, "HOT accepted invalid bulk rows");
      fresh->bulk_load(empty);  // invalid input must not partially build
      check(!fresh->find("a"), "HOT invalid bulk_load mutated contents");
    }
    std::cout << "Native HOT: descriptor, exact lookups, byte order, boundary keys, ownership, inserts, scans/ranges passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
