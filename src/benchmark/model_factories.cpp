#include "factories.h"
#include <set>
#include "learned_index.h"

namespace {
blocklearn::Options parse(const string_index_bench::v1::Parameters& params,
                          blocklearn::Verification verification, bool insert) {
  blocklearn::Options o; o.verification = verification; o.allow_insert = insert;
  std::set<std::string> seen;
  for (const auto& [name, text] : params) {
    if (!seen.insert(name).second || text.empty() ||
        text.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("invalid/duplicate parameter: " + name);
    std::size_t consumed = 0;
    const auto value = std::stoull(text, &consumed);
    if (consumed != text.size() || value > std::size_t(-1))
      throw std::invalid_argument("parameter out of range: " + name);
    if (name == "region_size") o.region_size = value;
    else if (name == "directory_fanout") o.directory_fanout = value;
    else if (name == "delta_capacity") o.delta_capacity = value;
    else if (name == "max_segments") o.max_segments = value;
    else if (name == "max_block_error") o.max_block_error = value;
    else if (name == "tag_features" && value <= 4) o.tag_features = unsigned(value);
    else if (name == "instrument" && value <= 1) o.instrument = value != 0;
    // Named factories fix both accepted optimizations; do not permit an ID
    // to silently describe a different routing/probing configuration.
    else throw std::invalid_argument("unknown/invalid parameter: " + name);
  }
  return o;
}
std::unique_ptr<string_index_bench::v1::Index> make_cost_residual(
    const string_index_bench::v1::Parameters& p) {
  auto options = parse(p, blocklearn::Verification::residual, true);
  options.short_directory = options.predicted_first = options.workload_planner = true;
  return std::make_unique<blocklearn::LearnedIndex>(options);
}
std::unique_ptr<string_index_bench::v1::Index> make_adaptive(
    const string_index_bench::v1::Parameters& p) {
  auto options = parse(p, blocklearn::Verification::adaptive, true);
  options.short_directory = options.predicted_first = options.workload_planner = true;
  return std::make_unique<blocklearn::LearnedIndex>(options);
}
}
std::unique_ptr<string_index_bench::v1::Index> make_block_residual_delta_shortdir_first(
    const string_index_bench::v1::Parameters& p) {
  auto options = parse(p, blocklearn::Verification::residual, true);
  options.short_directory = options.predicted_first = true;
  return std::make_unique<blocklearn::LearnedIndex>(options);
}
std::unique_ptr<string_index_bench::v1::Index> make_block_residual_delta_shortdir_first_cost(
    const string_index_bench::v1::Parameters& p) { return make_cost_residual(p); }
std::unique_ptr<string_index_bench::v1::Index> make_block_adaptive_delta_shortdir_first(
    const string_index_bench::v1::Parameters& p) { return make_adaptive(p); }
