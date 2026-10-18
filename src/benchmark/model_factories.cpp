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
    else if (name == "short_directory" && value <= 1) o.short_directory = value != 0;
    else if (name == "predicted_first" && value <= 1) o.predicted_first = value != 0;
    else throw std::invalid_argument("unknown/invalid parameter: " + name);
  }
  return o;
}
std::unique_ptr<string_index_bench::v1::Index> variant(
    const string_index_bench::v1::Parameters& params,
    blocklearn::Verification verification, bool insert, bool shortdir, bool first) {
  auto options = parse(params, verification, insert);
  for (const auto& parameter : params)
    if (parameter.first == "short_directory" || parameter.first == "predicted_first")
      throw std::invalid_argument("ablation factory fixes optimization switches");
  options.short_directory = shortdir; options.predicted_first = first;
  return std::make_unique<blocklearn::LearnedIndex>(options);
}
}
std::unique_ptr<string_index_bench::v1::Index> make_block_plain(
    const string_index_bench::v1::Parameters& p) {
  return std::make_unique<blocklearn::LearnedIndex>(parse(p, blocklearn::Verification::full_key, false));
}
std::unique_ptr<string_index_bench::v1::Index> make_block_hash(
    const string_index_bench::v1::Parameters& p) {
  return std::make_unique<blocklearn::LearnedIndex>(parse(p, blocklearn::Verification::hash, false));
}
std::unique_ptr<string_index_bench::v1::Index> make_block_residual(
    const string_index_bench::v1::Parameters& p) {
  return std::make_unique<blocklearn::LearnedIndex>(parse(p, blocklearn::Verification::residual, false));
}
std::unique_ptr<string_index_bench::v1::Index> make_block_residual_delta(
    const string_index_bench::v1::Parameters& p) {
  return std::make_unique<blocklearn::LearnedIndex>(parse(p, blocklearn::Verification::residual, true));
}
std::unique_ptr<string_index_bench::v1::Index> make_block_hash_shortdir(
    const string_index_bench::v1::Parameters& p) {
  return variant(p, blocklearn::Verification::hash, false, true, false);
}
std::unique_ptr<string_index_bench::v1::Index> make_block_hash_first(
    const string_index_bench::v1::Parameters& p) {
  return variant(p, blocklearn::Verification::hash, false, false, true);
}
std::unique_ptr<string_index_bench::v1::Index> make_block_hash_shortdir_first(
    const string_index_bench::v1::Parameters& p) {
  return variant(p, blocklearn::Verification::hash, false, true, true);
}
std::unique_ptr<string_index_bench::v1::Index> make_block_residual_delta_shortdir(
    const string_index_bench::v1::Parameters& p) {
  return variant(p, blocklearn::Verification::residual, true, true, false);
}
std::unique_ptr<string_index_bench::v1::Index> make_block_residual_delta_first(
    const string_index_bench::v1::Parameters& p) {
  return variant(p, blocklearn::Verification::residual, true, false, true);
}
std::unique_ptr<string_index_bench::v1::Index> make_block_residual_delta_shortdir_first(
    const string_index_bench::v1::Parameters& p) {
  return variant(p, blocklearn::Verification::residual, true, true, true);
}
