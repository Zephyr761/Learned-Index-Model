#include "query_cost.h"
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <set>
#include <stdexcept>
#include <utility>

namespace blocklearn {
namespace {
using Field = std::pair<const char*, double QueryCostWeights::*>;
constexpr std::array<Field, 12> fields{{
    {"encode_byte", &QueryCostWeights::encode_byte}, {"segment_step", &QueryCostWeights::segment_step},
    {"tag_batch", &QueryCostWeights::tag_batch}, {"scalar_tag", &QueryCostWeights::scalar_tag},
    {"key_verify", &QueryCostWeights::key_verify}, {"verify_byte", &QueryCostWeights::verify_byte},
    {"hash_base", &QueryCostWeights::hash_base}, {"hash_byte", &QueryCostWeights::hash_byte},
    {"residual_base", &QueryCostWeights::residual_base}, {"feature_read", &QueryCostWeights::feature_read},
    {"feature_line", &QueryCostWeights::feature_line}, {"space_byte", &QueryCostWeights::space_byte}}};
std::uint64_t integer(const std::string& text) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("invalid cost-profile integer");
  return std::stoull(text);
}
}
std::string cost_target() {
#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64))
  return "x86_64-avx2";
#elif defined(__aarch64__)
  return "aarch64-neon";
#elif defined(__x86_64__) || defined(_M_X64)
  return "x86_64-scalar";
#else
  return "portable-scalar";
#endif
}
std::string cost_build() {
#ifdef NDEBUG
  return "release";
#else
  return "debug";
#endif
}
void QueryCostWeights::validate() const {
  for (const auto& field : fields)
    if (!std::isfinite(this->*field.second) || this->*field.second <= 0 || this->*field.second > 1e9)
      throw std::invalid_argument(std::string("invalid cost coefficient: ") + field.first);
}
CostProfile read_cost_profile(std::istream& input) {
  std::string schema;
  if (!(input >> schema) || schema != "blocklearn-query-cost/v1")
    throw std::invalid_argument("invalid cost profile schema");
  CostProfile result;
  std::set<std::string> seen;
  std::string name, value;
  while (input >> name) {
    if (!(input >> value) || !seen.insert(name).second)
      throw std::invalid_argument("truncated/duplicate cost profile field");
    if (name == "origin") result.origin = value;
    else if (name == "target") result.target = value;
    else if (name == "build") result.build = value;
    else if (name == "seed") result.seed = integer(value);
    else if (name == "iterations") result.iterations = integer(value);
    else {
      bool known = false;
      for (const auto& field : fields) if (name == field.first) {
        std::size_t end = 0;
        result.weights.*field.second = std::stod(value, &end);
        if (end != value.size()) throw std::invalid_argument("invalid cost profile number");
        known = true; break;
      }
      if (!known) throw std::invalid_argument("unknown cost profile field: " + name);
    }
  }
  if (!input.eof() || seen.size() != fields.size() + 5 ||
      result.origin != "independent-synthetic-v1" || !result.iterations)
    throw std::invalid_argument("incomplete/non-independent cost profile");
  if (result.target != cost_target() || result.build != cost_build())
    throw std::invalid_argument("cost profile SIMD target/build mismatch; recalibrate on this machine with this build");
  result.weights.validate();
  return result;
}
void write_cost_profile(std::ostream& output, const CostProfile& profile) {
  profile.weights.validate();
  if (profile.origin != "independent-synthetic-v1" || !profile.iterations ||
      profile.target != cost_target() || profile.build != cost_build())
    throw std::invalid_argument("invalid calibration provenance");
  output << "blocklearn-query-cost/v1\norigin " << profile.origin << "\nseed " << profile.seed
         << "\niterations " << profile.iterations << "\ntarget " << profile.target
         << "\nbuild " << profile.build << '\n' << std::setprecision(17);
  for (const auto& field : fields) output << field.first << ' ' << profile.weights.*field.second << '\n';
  if (!output) throw std::runtime_error("could not write cost profile");
}
CostProfile load_cost_profile(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::invalid_argument("could not read cost profile: " + path);
  return read_cost_profile(input);
}
CostProfile environment_cost_profile() {
  const char* path = std::getenv("BLOCKLEARN_COST_PROFILE");
  if (!path || !*path)
    throw std::invalid_argument("cost planner requires calibrated BLOCKLEARN_COST_PROFILE; use --planner-costs or the suite's automatic calibration");
  return load_cost_profile(path);
}
double QueryWork::verification_cost(const QueryCostWeights& w) const {
  return scalar_tags * w.scalar_tag + key_verifications * w.key_verify + compared_bytes * w.verify_byte
      + hash_calls * w.hash_base + hash_bytes * w.hash_byte + residual_calls * w.residual_base
      + feature_reads * w.feature_read + feature_lines * w.feature_line;
}
QueryWork& QueryWork::operator+=(const QueryWork& other) {
  tag_batches += other.tag_batches; scalar_tags += other.scalar_tags; tag_tests += other.tag_tests;
  key_verifications += other.key_verifications; compared_bytes += other.compared_bytes;
  hash_calls += other.hash_calls; hash_bytes += other.hash_bytes; residual_calls += other.residual_calls;
  feature_reads += other.feature_reads; feature_lines += other.feature_lines;
  return *this;
}
QueryWork QueryWork::scaled(double scale) const {
  QueryWork result = *this;
  result.tag_batches *= scale; result.scalar_tags *= scale; result.tag_tests *= scale;
  result.key_verifications *= scale; result.compared_bytes *= scale;
  result.hash_calls *= scale; result.hash_bytes *= scale; result.residual_calls *= scale;
  result.feature_reads *= scale; result.feature_lines *= scale;
  return result;
}
}  // namespace blocklearn
