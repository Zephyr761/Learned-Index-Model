#include "benchmark/factories.h"
#include "key_bytes.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
using namespace string_index_bench::v1;
static void check(bool b, const char* message) { if (!b) throw std::runtime_error(message); }
int main() {
  try {
    for (Factory factory : {make_art_baseline, make_hot_baseline}) {
      auto index = factory({});
      if (index->descriptor().capabilities.bulk_load == Support::unsupported) {
        bool threw = false;
        try { index->bulk_load({}); } catch (const UnsupportedOperation&) { threw = true; }
        check(threw, "unavailable baseline did not throw");
        std::cout << index->descriptor().implementation << ": unsupported, no substitute\n";
        continue;
      }
      std::vector<Row> rows{{"", 0}, {"a", UINT64_MAX}, {"aa", 2}, {"ab", 3},
                            {"b", 4}, {std::string(1, char(255)), 5}};
      std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return blocklearn::less(a.key, b.key); });
      index->bulk_load(rows);
      for (const auto& r : rows) check(index->find(r.key) == r.value, "baseline lookup");
      check(!index->find("aaa"), "baseline false positive");
      check(index->insert("a", 42) == InsertResult::already_exists, "baseline duplicate result");
      check(index->find("a") == UINT64_MAX, "baseline duplicate overwrote value");
      check(index->insert("aab", 0) == InsertResult::inserted && index->find("aab") == 0, "baseline insertion");
      auto scan = index->scan("aa", 3);
      check(scan.size() == 3 && scan[0].key == "aa" && scan[1].key == "aab" && scan[2].key == "ab", "baseline scan");
      check(index->range_sum("a", "ab") == 4, "baseline range/modular sum");
      std::vector<Row> empty;
      auto e = factory({}); e->bulk_load(empty);
      check(!e->find("") && e->scan("", 5).empty(), "empty baseline");
      check(e->memory_usage().index_owned_bytes.has_value(), "empty baseline memory");
      check(e->insert("", 0) == InsertResult::inserted && e->find("") == 0, "baseline empty key insert");
      std::cout << index->descriptor().implementation << ": passed\n";
    }
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
