#pragma once
#include "block_model.h"
#include "query_cost.h"
#include "residual_tags.h"
#include <string>

namespace blocklearn {
enum class TagKind { hash, residual };
struct TagStrategy {
  TagKind kind = TagKind::residual;
  TagPlan learned;
  bool predicted_first = false;
};
struct PlanningSamples {
  std::vector<std::size_t> hits;
  std::vector<std::string> misses;
};
struct WorkEvaluation {
  QueryWork hits, misses, expected;
  std::size_t hit_samples = 0, miss_samples = 0;
  double cost = 0;
};
struct RegionPlan {
  BlockModel model;
  TagStrategy strategy;
  WorkEvaluation evaluation;
};
struct RegionPlanningOptions {
  std::size_t max_segments = 8, max_block_error = 1;
  unsigned max_features = 2;
  bool allow_hash = true, allow_residual = true;
  bool predicted_first = true, choose_order = true;
};
// Split stored-key planning samples, never benchmark query input. Synthetic
// misses stay inside the region's stored-key bounds and are verified absent.
PlanningSamples planning_samples(const std::vector<std::string_view>&, bool validation);
WorkEvaluation evaluate_query_work(const std::vector<std::string_view>&,
    const BlockModel&, const TagStrategy&, const QueryCostWeights&, const PlanningSamples&);
RegionPlan plan_region(const std::vector<std::string_view>&,
    const QueryCostWeights&, const RegionPlanningOptions&);
RegionPlan plan_model(const std::vector<std::string_view>&, const BlockModel&,
    const QueryCostWeights&, const RegionPlanningOptions&);
}  // namespace blocklearn
