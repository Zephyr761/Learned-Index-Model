#pragma once
#include <cstdint>
#include <iosfwd>
#include <string>

namespace blocklearn {
std::string cost_target();
std::string cost_build();

// Positive ns/work-unit coefficients. All-one values are for deterministic
// unit tests, NOT an implicitly calibrated performance configuration.
struct QueryCostWeights {
  double encode_byte = 1, segment_step = 1, tag_batch = 1, scalar_tag = 1;
  double key_verify = 1, verify_byte = 1, hash_base = 1, hash_byte = 1;
  double residual_base = 1, feature_read = 1, feature_line = 1, space_byte = 1;
  void validate() const;
};
struct CostProfile {
  QueryCostWeights weights;
  std::uint64_t seed = 0, iterations = 0;
  std::string origin;
  std::string target = cost_target(), build = cost_build();
};
CostProfile read_cost_profile(std::istream& input);
void write_cost_profile(std::ostream& output, const CostProfile& profile);
CostProfile load_cost_profile(const std::string& path);
CostProfile environment_cost_profile();

struct QueryWork {
  double tag_batches = 0, scalar_tags = 0, tag_tests = 0;
  double key_verifications = 0, compared_bytes = 0;
  double hash_calls = 0, hash_bytes = 0, residual_calls = 0;
  double feature_reads = 0, feature_lines = 0;
  double verification_cost(const QueryCostWeights&) const;
  QueryWork& operator+=(const QueryWork&);
  QueryWork scaled(double) const;
};

}  // namespace blocklearn
