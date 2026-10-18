#pragma once
#include "factories.h"
#include <array>
#include <string_view>

namespace blocklearn::bench {
struct ModelFactory {
  const char* id;
  string_index_bench::v1::Factory factory;
  bool control;
};
inline const std::array<ModelFactory, 10>& model_factories() {
  static const std::array<ModelFactory, 10> models{{
      {"block-plain", make_block_plain, true},
      {"block-hash", make_block_hash, true},
      {"block-residual", make_block_residual, true},
      {"block-residual-delta", make_block_residual_delta, true},
      {"block-hash-shortdir", make_block_hash_shortdir, false},
      {"block-hash-first", make_block_hash_first, false},
      {"block-hash-shortdir-first", make_block_hash_shortdir_first, false},
      {"block-residual-delta-shortdir", make_block_residual_delta_shortdir, false},
      {"block-residual-delta-first", make_block_residual_delta_first, false},
      {"block-residual-delta-shortdir-first", make_block_residual_delta_shortdir_first, false}}};
  return models;
}
inline bool is_model_ablation(std::string_view id) {
  for (const auto& model : model_factories()) if (id == model.id) return !model.control;
  return false;
}
}  // namespace blocklearn::bench
