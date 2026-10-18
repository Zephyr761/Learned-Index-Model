// Companion to the teammate's static runner. Same v1 interface and SIBROW1
// interchange, but a frozen mixed workload with local maintenance timed.
#include "factories.h"
#include "model_registry.h"
#include "learned_index.h"
#include "key_bytes.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

using namespace string_index_bench::v1;
using Clock = std::chrono::steady_clock;
struct Settings {
  std::string rows, implementation = "all";
  std::size_t operations = 100000, repeats = 3, latency_every = 1024;
  std::uint64_t seed = 42;
  double initial_fraction = 0.8, insert_ratio = 0.05, miss_ratio = 0.1;
};
struct Operation { bool insert; std::string_view key; Value value; bool present; };
struct Response { bool present; Value value; };
struct Workload {
  std::vector<Row> initial;
  std::deque<std::string> misses;
  std::vector<Operation> operations;
  std::size_t inserts = 0, reads = 0, miss_reads = 0;
};
static std::uint64_t read_u64(std::istream& in) {
  unsigned char bytes[8];
  if (!in.read(reinterpret_cast<char*>(bytes), 8)) throw std::runtime_error("truncated SIBROW1 integer");
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i) value |= std::uint64_t(bytes[i]) << (8 * i);
  return value;
}
static std::vector<Row> read_rows(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open rows interchange");
  in.seekg(0, std::ios::end); const auto size = in.tellg(); in.seekg(0);
  char magic[8];
  if (!in.read(magic, 8) || std::string_view(magic, 8) != std::string_view("SIBROW1\0", 8))
    throw std::runtime_error("expected teammate benchmark SIBROW1 rows");
  const auto count = read_u64(in);
  if (size < 16 || count > (std::uint64_t(size) - 16) / 16)
    throw std::runtime_error("invalid row count");
  std::vector<Row> rows; rows.reserve(count);
  for (std::uint64_t i = 0; i < count; ++i) {
    const auto length = read_u64(in);
    const auto left = std::uint64_t(size - in.tellg());
    if (left < 8 || length > left - 8) throw std::runtime_error("invalid key length");
    std::string key(length, '\0');
    if (length && !in.read(key.data(), std::streamsize(length))) throw std::runtime_error("truncated key");
    const auto value = read_u64(in);
    if (!rows.empty() && !blocklearn::less(rows.back().key, key))
      throw std::runtime_error("rows not strictly unsigned-byte sorted/unique");
    rows.push_back({std::move(key), value});
  }
  if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing rows bytes");
  return rows;
}
static Workload workload(const std::vector<Row>& rows, const Settings& s) {
  Workload w;
  if (rows.empty()) return w;
  std::mt19937_64 rng(s.seed);
  std::vector<std::size_t> shuffled(rows.size());
  std::iota(shuffled.begin(), shuffled.end(), 0);
  std::shuffle(shuffled.begin(), shuffled.end(), rng);
  const auto initial = std::max<std::size_t>(1, std::size_t(rows.size() * s.initial_fraction));
  std::vector<bool> is_initial(rows.size(), false);
  std::vector<std::size_t> available(shuffled.begin(), shuffled.begin() + initial);
  for (auto id : available) is_initial[id] = true;
  for (std::size_t i = 0; i < rows.size(); ++i) if (is_initial[i]) w.initial.push_back(rows[i]);
  w.operations.reserve(s.operations);
  auto draw = [&] { return std::generate_canonical<double, 53>(rng); };
  std::size_t next = initial;
  for (std::size_t i = 0; i < s.operations; ++i) {
    if (draw() < s.insert_ratio && next < shuffled.size()) {
      const auto id = shuffled[next++]; available.push_back(id);
      w.operations.push_back({true, rows[id].key, rows[id].value, true}); ++w.inserts;
    } else if (draw() < s.miss_ratio) {
      std::string key;
      do {
        key = "~blocklearn-missing/" + std::to_string(rng());
      } while (std::binary_search(rows.begin(), rows.end(), Row{key, 0},
          [](const Row& a, const Row& b) { return blocklearn::less(a.key, b.key); }));
      w.misses.push_back(std::move(key));
      w.operations.push_back({false, w.misses.back(), 0, false}); ++w.reads; ++w.miss_reads;
    } else {
      const auto id = available[rng() % available.size()];
      w.operations.push_back({false, rows[id].key, rows[id].value, true}); ++w.reads;
    }
  }
  return w;
}
static Response execute(Index& index, const Operation& op) {
  if (op.insert) return {index.insert(op.key, op.value) == InsertResult::inserted, 0};
  auto v = index.find(op.key);
  return {v.has_value(), v.value_or(0)};
}
static void verify(const Response& r, const Operation& op) {
  if (r.present != op.present || (!op.insert && op.present && r.value != op.value))
    throw std::runtime_error("mixed workload response mismatch");
}
static bool accepts(const KeyDomain& d, std::string_view key) {
  if (key.size() < d.min_bytes || (d.max_bytes && key.size() > *d.max_bytes) ||
      (d.fixed_bytes && key.size() != *d.fixed_bytes)) return false;
  for (unsigned char c : key) if ((!d.allows_nul && !c) || (!d.allows_high_bit && c >= 128)) return false;
  return true;
}
static double elapsed(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::nano>(end - start).count();
}
static double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  return values[std::size_t((values.size() - 1) * fraction)];
}
static std::string optional(std::optional<std::uint64_t> value) {
  return value ? std::to_string(*value) : "";
}
static std::string quoted(std::string text) {
  std::string result = "\"";
  for (char c : text) { if (c == '"') result += '"'; result += c == '\n' ? ' ' : c; }
  return result + '"';
}
int main(int argc, char** argv) {
  try {
    std::vector<std::pair<std::string, Factory>> implementations;
    for (const auto& model : blocklearn::bench::model_factories())
      implementations.emplace_back(model.id, model.factory);
    implementations.emplace_back("art", make_art_baseline);
    implementations.emplace_back("hot", make_hot_baseline);
    if (argc == 2 && std::string_view(argv[1]) == "--describe") {
      std::cout << "implementation,status,notes\n";
      for (const auto& implementation : implementations) {
        const auto d = implementation.second({})->descriptor();
        const bool usable = d.capabilities.bulk_load != Support::unsupported && d.capabilities.find != Support::unsupported;
        std::cout << implementation.first << ',' << (usable ? "available" : "unsupported")
          << ',' << quoted(d.adaptation_notes) << '\n';
      }
      return 0;
    }
    Settings s;
    for (int i = 1; i < argc; ++i) {
      std::string name = argv[i];
      if (name == "--help") {
        std::cout << "--rows SIBROW1-file [--implementation all|MODEL-ID|art|hot] (model IDs include -shortdir/-first/-shortdir-first ablations; all keeps original six) "
          "[--operations 100000 --repeats 3 --seed 42 --initial-fraction .8 --insert-ratio .05 --miss-ratio .1 --latency-every 1024]\n";
        return 0;
      }
      if (++i == argc) throw std::invalid_argument("missing argument value");
      const std::string value = argv[i];
      auto integer = [&]() {
        if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
          throw std::invalid_argument("invalid integer");
        return std::stoull(value);
      };
      auto number = [&]() { std::size_t p; auto v = std::stod(value, &p);
        if (p != value.size()) throw std::invalid_argument("invalid number"); return v; };
      if (name == "--rows") s.rows = value;
      else if (name == "--implementation") s.implementation = value;
      else if (name == "--operations") s.operations = integer();
      else if (name == "--repeats") s.repeats = integer();
      else if (name == "--seed") s.seed = integer();
      else if (name == "--latency-every") s.latency_every = integer();
      else if (name == "--initial-fraction") s.initial_fraction = number();
      else if (name == "--insert-ratio") s.insert_ratio = number();
      else if (name == "--miss-ratio") s.miss_ratio = number();
      else throw std::invalid_argument("unknown option: " + name);
    }
    if (s.rows.empty() || !s.operations || !s.repeats || !s.latency_every ||
        !(s.initial_fraction > 0 && s.initial_fraction <= 1) ||
        !(s.insert_ratio >= 0 && s.insert_ratio <= 1) || !(s.miss_ratio >= 0 && s.miss_ratio <= 1))
      throw std::invalid_argument("invalid mixed configuration");
    const auto rows = read_rows(s.rows);
    const auto w = workload(rows, s);
    if (s.implementation != "all" && std::none_of(implementations.begin(), implementations.end(),
        [&](const auto& x) { return x.first == s.implementation; })) throw std::invalid_argument("unknown implementation");
    std::cout << "implementation,status,repeat,initial_keys,total_keys,operations,reads,inserts,miss_reads,build_ns,ns_per_op,read_p50_ns,read_p99_ns,insert_p50_ns,insert_p99_ns,index_owned_bytes,adapter_owned_bytes,retained_input_bytes,checksum,merges,leaf_splits,directory_splits,full_key_comparisons,tag_tests,directory_fallbacks,notes\n";
    bool failed = false;
    for (const auto& implementation : implementations) {
      const auto& name = implementation.first;
      const auto factory = implementation.second;
      if (s.implementation != "all" && s.implementation != name) continue;
      if (s.implementation == "all" && blocklearn::bench::is_model_ablation(name)) continue;
      auto emit_status = [&](const std::string& status, const std::string& reason) {
        std::cout << name << ',' << status << ",," << w.initial.size() << ',' << rows.size()
          << ',' << w.operations.size() << ',' << w.reads << ',' << w.inserts << ',' << w.miss_reads;
        for (int i = 0; i < 16; ++i) std::cout << ',';
        std::cout << ',' << quoted(reason) << '\n';
      };
      try {
        auto check_index = factory({});
        const auto d = check_index->descriptor();
        const auto& c = d.capabilities;
        if (rows.empty()) { emit_status("skipped", "empty dataset"); continue; }
        if (c.bulk_load == Support::unsupported || c.find == Support::unsupported ||
            (w.inserts && c.insert == Support::unsupported)) {
          emit_status("unsupported", d.adaptation_notes + "; inserts unsupported or native baseline unavailable"); continue;
        }
        bool domain = true;
        for (const auto& r : w.initial) domain = domain && accepts(c.keys, r.key);
        for (const auto& op : w.operations) domain = domain && accepts(c.keys, op.key);
        if (!domain) { emit_status("unsupported", "workload exceeds declared key domain; no filtering"); continue; }
        // Complete semantic validation uses a separate instance outside timers.
        if (name.rfind("block-", 0) == 0) check_index = factory({{"instrument", "1"}});
        check_index->bulk_load(w.initial);
        for (const auto& op : w.operations) verify(execute(*check_index, op), op);
        blocklearn::Statistics stats;
        if (auto* own = dynamic_cast<blocklearn::LearnedIndex*>(check_index.get())) {
          stats = own->statistics(); own->validate();
        }
        check_index.reset();
        for (std::size_t repeat = 0; repeat < s.repeats; ++repeat) {
          const auto start_build = Clock::now();
          auto index = factory({}); index->bulk_load(w.initial);
          const auto build_ns = elapsed(start_build, Clock::now());
          std::vector<Response> responses(w.operations.size());
          const auto start = Clock::now();
          for (std::size_t i = 0; i < w.operations.size(); ++i) responses[i] = execute(*index, w.operations[i]);
          const auto duration = elapsed(start, Clock::now());
          Value checksum = 0;
          for (std::size_t i = 0; i < responses.size(); ++i) {
            verify(responses[i], w.operations[i]); checksum += responses[i].value + responses[i].present;
          }
          const auto memory = index->memory_usage(); index.reset();
          auto latency = factory({}); latency->bulk_load(w.initial);
          std::vector<double> read_samples, insert_samples;
          for (std::size_t i = 0; i < w.operations.size(); ++i) {
            const auto& op = w.operations[i];
            if (op.insert || i % s.latency_every == 0) {
              const auto before = Clock::now(); const auto result = execute(*latency, op);
              const auto ns = elapsed(before, Clock::now());
              (op.insert ? insert_samples : read_samples).push_back(ns);
              verify(result, op);
            } else verify(execute(*latency, op), op);
          }
          auto metric = [&](const std::vector<double>& samples, double p) {
            return samples.empty() ? std::string() : std::to_string(percentile(samples, p));
          };
          std::cout << std::setprecision(12) << name << ",passed," << repeat << ','
            << w.initial.size() << ',' << rows.size() << ',' << w.operations.size() << ','
            << w.reads << ',' << w.inserts << ',' << w.miss_reads << ',' << build_ns << ','
            << duration / w.operations.size() << ',' << metric(read_samples, .5) << ','
            << metric(read_samples, .99) << ',' << metric(insert_samples, .5) << ','
            << metric(insert_samples, .99) << ',' << optional(memory.index_owned_bytes) << ','
            << optional(memory.adapter_owned_bytes) << ',' << optional(memory.retained_input_bytes)
            << ',' << checksum;
          const bool own = name.rfind("block-", 0) == 0;
          for (auto value : {stats.merges, stats.leaf_splits, stats.directory_splits,
                             stats.full_key_comparisons, stats.tag_tests, stats.directory_fallbacks})
            std::cout << ',' << (own ? std::to_string(value) : std::string());
          std::cout << ',' << quoted(memory.accounting_notes + "; " + d.adaptation_notes) << '\n';
        }
      } catch (const std::exception& e) { failed = true; emit_status("failed", e.what()); }
    }
    return failed ? 1 : 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
