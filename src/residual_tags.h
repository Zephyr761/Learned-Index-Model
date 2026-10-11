#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>
#include "block_model.h"

namespace blocklearn {

struct TagPlan {
  std::array<std::size_t, 4> offsets{};  // absolute offsets, not ordinal features
  unsigned count = 0;
  std::uint16_t tag(std::string_view key) const;
  double candidate_matches = 0;
  double cost = 0;
};

// Greedy, bounded feature selection. The objective counts tag collisions
// inside the model's candidate windows, rather than global byte entropy.
TagPlan learn_tags(const std::vector<std::string_view>& keys,
                   const BlockModel& model, unsigned max_features = 2);
double evaluate_tags(const std::vector<std::string_view>& keys,
                     const BlockModel& model, const TagPlan& plan);

}  // namespace blocklearn
