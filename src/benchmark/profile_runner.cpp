// Separate diagnostic executable. Never replace the teammate's throughput
// runner with the instrumented path, or interpret these raw times as QPS.
#include "factories.h"
#include "model_registry.h"
#include "learned_index.h"
#include "profile_interchange.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace string_index_bench::v1;
using namespace blocklearn;
using Clock = std::chrono::steady_clock;

struct TimerCalibration { double median_ns, p90_ns, zero_fraction; };
static TimerCalibration calibrate_clock() {
  std::vector<double> samples; samples.reserve(4096);
  for (unsigned i = 0; i < 4096; ++i) {
    const auto before = Clock::now();
    const auto after = Clock::now();
    samples.push_back(std::chrono::duration<double, std::nano>(after - before).count());
  }
  std::sort(samples.begin(), samples.end());
  const auto zero = std::count(samples.begin(), samples.end(), 0.0);
  return {samples[samples.size() / 2], samples[(samples.size() - 1) * 9 / 10],
          double(zero) / samples.size()};
}
static void verify(const std::optional<Value>& result, const bench::ProfileQuery& query) {
  if (result.has_value() != query.present || (result && *result != query.value))
    throw std::runtime_error("profile lookup differs from frozen query answer");
}
int main(int argc, char** argv) {
  try {
    std::string rows_path, queries_path, implementation;
    std::size_t maximum = 10000, repeats = 3;
    for (int i = 1; i < argc; ++i) {
      const std::string name = argv[i];
      if (name == "--help") {
        std::cout << "--rows SIBROW1 --queries SIBQRY1 --implementation MODEL-ID (including -shortdir/-first/-shortdir-first ablations) [--samples 10000 --repeats 3]\n";
        return 0;
      }
      if (++i == argc) throw std::invalid_argument("missing option value");
      const std::string value = argv[i];
      auto positive = [&]() {
        if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
          throw std::invalid_argument("invalid positive integer");
        const auto n = std::stoull(value);
        if (!n || n > std::numeric_limits<std::size_t>::max())
          throw std::invalid_argument("integer out of range");
        return static_cast<std::size_t>(n);
      };
      if (name == "--rows") rows_path = value;
      else if (name == "--queries") queries_path = value;
      else if (name == "--implementation") implementation = value;
      else if (name == "--samples") maximum = positive();
      else if (name == "--repeats") repeats = positive();
      else throw std::invalid_argument("unknown option: " + name);
    }
    Factory factory = nullptr;
    for (const auto& model : bench::model_factories())
      if (implementation == model.id) factory = model.factory;
    if (rows_path.empty() || queries_path.empty() || !factory)
      throw std::invalid_argument("rows, queries, and a BlockLearn implementation are required");
    const auto rows = bench::read_profile_rows(rows_path);
    const auto queries = bench::read_profile_queries(queries_path, maximum);
    if (rows.empty() || queries.sample.empty()) throw std::invalid_argument("empty profile workload");
    // Check the sampled frozen answers against the input independently of the
    // model, outside all timers (including missing keys and zero values).
    for (const auto& q : queries.sample) {
      auto it = std::lower_bound(rows.begin(), rows.end(), q.key,
          [](const Row& row, std::string_view key) { return less(row.key, key); });
      verify(it != rows.end() && it->key == q.key ? std::optional<Value>(it->value) : std::nullopt, q);
    }
    auto index = factory({{"instrument", "1"}});
    index->bulk_load(rows);
    auto& own = dynamic_cast<LearnedIndex&>(*index);
    const auto timer = calibrate_clock();
    std::cout << "implementation,status,repeat,outcome,source_operations,sampled_queries,stage,total_ns,ns_per_query,percent_of_profiled_total,scope_calls,calls_per_query,clock_pair_median_ns,clock_pair_p90_ns,clock_pair_zero_fraction,checksum,full_key_comparisons,tag_tests,directory_fallbacks,tag_batches,hash_leaves,residual_leaves,predicted_first_leaves\n";
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
      std::array<LookupProfile, 2> groups;
      std::array<Value, 2> checksums{};
      const auto before = own.statistics();
      for (const auto& q : queries.sample) {
        auto result = own.profile_find(q.key, groups[q.present ? 1 : 0]);
        verify(result, q); checksums[q.present ? 1 : 0] += result.value_or(0) + result.has_value();
      }
      const auto after = own.statistics();
      auto all = groups[0];
      all += groups[1];
      auto emit = [&](const char* outcome, const LookupProfile& profile, Value checksum) {
        if (!profile.queries) return;  // do not invent a zero-cost absent group
        auto line = [&](const char* stage, double ns, std::uint64_t calls) {
          std::cout << std::setprecision(15) << implementation << ",passed," << repeat << ','
              << outcome << ',' << queries.total << ',' << profile.queries << ',' << stage
              << ',' << ns << ',' << ns / profile.queries << ',';
          if (profile.total_ns > 0) std::cout << 100.0 * ns / profile.total_ns;
          std::cout << ',' << calls << ',' << double(calls) / profile.queries << ','
              << timer.median_ns << ',' << timer.p90_ns << ',' << timer.zero_fraction << ',' << checksum << ',';
          // Counters cover the entire sample, not an invented hit/miss split.
          if (std::string_view(outcome) == "all") {
            std::cout << after.full_key_comparisons - before.full_key_comparisons << ','
                << after.tag_tests - before.tag_tests << ','
                << after.directory_fallbacks - before.directory_fallbacks << ','
                << after.tag_batches - before.tag_batches << ',' << after.hash_leaves << ','
                << after.residual_leaves << ',' << after.predicted_first_leaves;
          } else std::cout << ",,,,,,";
          std::cout << '\n';
        };
        line("profiled_total", profile.total_ns, profile.queries);
        for (std::size_t s = 0; s < lookup_stage_count; ++s)
          line(stage_name(static_cast<LookupStage>(s)), profile.stages[s].ns, profile.stages[s].calls);
      };
      emit("all", all, checksums[0] + checksums[1]);
      emit("hit", groups[1], checksums[1]); emit("miss", groups[0], checksums[0]);
    }
    own.validate();  // expensive structural audit outside the diagnostic queries
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
