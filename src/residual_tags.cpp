#include "residual_tags.h"

#include <algorithm>
#include <limits>
#include <set>
#include "key_bytes.h"

namespace blocklearn {

std::uint16_t TagPlan::tag(std::string_view key) const {
  // Same key always produces the same tag, including short keys/NUL bytes.
  // This is NOT a perfect hash or an equality test.
  std::uint32_t h = static_cast<std::uint32_t>(key.size()) * 0x9e3779b1U;
  for (unsigned i = 0; i < count; ++i) {
    const auto p = offsets[i];
    const auto symbol = p < key.size()
        ? unsigned(static_cast<unsigned char>(key[p])) + 1 : 0;
    h = (h ^ symbol) * 16777619U;
    h ^= h >> 13;
  }
  return static_cast<std::uint16_t>(h ^ (h >> 16));
}
double evaluate_tags(const std::vector<std::string_view>& keys,
                     const BlockModel& model, const TagPlan& plan) {
  if (keys.empty()) return 0;
  // Sorted (block, tag) counts avoid an N^2 pairwise training pass.
  std::vector<std::uint64_t> entries;
  entries.reserve(keys.size());
  for (std::size_t i = 0; i < keys.size(); ++i)
    entries.push_back(((i / model.block_size()) << 16) | plan.tag(keys[i]));
  std::sort(entries.begin(), entries.end());
  double matches = 0;
  const auto sample = std::min<std::size_t>(keys.size(), 128);
  for (std::size_t j = 0; j < sample; ++j) {
    const auto i = j * keys.size() / sample;
    const auto tag = plan.tag(keys[i]);
    const auto w = model.window(keys[i]);
    for (auto b = w.first; b < w.last; ++b) {
      const auto token = (b << 16) | tag;
      auto range = std::equal_range(entries.begin(), entries.end(), token);
      matches += range.second - range.first;
    }
  }
  return matches / sample;
}
TagPlan learn_tags(const std::vector<std::string_view>& keys,
                   const BlockModel& model, unsigned max_features) {
  TagPlan best;
  if (keys.empty()) return best;
  max_features = std::min(max_features, 4U);
  std::size_t max_length = 0;
  for (auto key : keys) max_length = std::max(max_length, key.size());
  const auto start = model.prefix_size();
  std::set<std::size_t> positions;
  // Bounded candidate set: early suffix bytes and spaced later windows.
  for (std::size_t i = 0; i < 8 && start + i < max_length; ++i)
    positions.insert(start + i);
  for (std::size_t q = 1; q <= 8 && max_length > start; ++q)
    positions.insert(start + (max_length - start - 1) * q / 8);
  best.candidate_matches = evaluate_tags(keys, model, best);
  best.cost = best.candidate_matches;
  for (unsigned step = 0; step < max_features; ++step) {
    TagPlan next = best;
    for (auto position : positions) {
      if (std::find(best.offsets.begin(), best.offsets.begin() + best.count,
                    position) != best.offsets.begin() + best.count) continue;
      auto trial = best;
      trial.offsets[trial.count++] = position;
      trial.candidate_matches = evaluate_tags(keys, model, trial);
      std::set<std::size_t> lines;
      for (unsigned i = 0; i < trial.count; ++i)
        lines.insert(trial.offsets[i] / 64);
      // Relative cache-line estimate only: actual string alignment varies.
      trial.cost = trial.candidate_matches + 0.04 * trial.count
          + 0.12 * (lines.size() > 1 ? lines.size() - 1 : 0);
      if (trial.cost < next.cost) next = trial;
    }
    if (next.count == best.count) break;
    best = next;
  }
  return best;
}

}  // namespace blocklearn
