#pragma once
#include "factories.h"
#include <array>

namespace blocklearn::bench {
struct ModelFactory {
  const char* id;
  string_index_bench::v1::Factory factory;
};
inline const std::array<ModelFactory, 3>& model_factories() {
  static const std::array<ModelFactory, 3> models{{
      {"block-residual-delta-shortdir-first", make_block_residual_delta_shortdir_first},
      {"block-residual-delta-shortdir-first-cost", make_block_residual_delta_shortdir_first_cost},
      {"block-adaptive-delta-shortdir-first", make_block_adaptive_delta_shortdir_first}}};
  return models;
}
}  // namespace blocklearn::bench
