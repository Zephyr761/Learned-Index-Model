#include "block_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "key_bytes.h"
#include "query_cost.h"

namespace blocklearn {

std::size_t BlockModel::predict(const Segment& s, std::uint64_t x) const {
  // Subtract in integer space first: converting full 63-bit coordinates to
  // double before subtraction would discard distinguishing low bits.
  const double dx = x >= s.x_min ? double(x - s.x_min) : -double(s.x_min - x);
  const double y = std::clamp(s.intercept + s.slope * dx,
                             double(s.y_min), double(s.y_max));
  return static_cast<std::size_t>(std::llround(y));
}
const BlockModel::Segment& BlockModel::segment(std::uint64_t x) const {
  auto it = std::lower_bound(segments_.begin(), segments_.end(), x,
      [](const Segment& s, std::uint64_t v) { return s.x_max < v; });
  return it == segments_.end() ? segments_.back() : *it;
}
BlockModel BlockModel::fit(const std::vector<std::string_view>& keys,
                           std::size_t block_size, unsigned bytes,
                           std::size_t segment_budget, std::size_t max_error) {
  if (!block_size || !segment_budget || !bytes || bytes > 7)
    throw std::invalid_argument("invalid block model options");
  BlockModel m;
  m.block_size_ = block_size;
  m.bytes_ = bytes;
  if (keys.empty()) { m.certified_ = true; return m; }
  m.prefix_ = keys.front().substr(0, common_prefix(keys.front(), keys.back()));
  m.blocks_ = (keys.size() + block_size - 1) / block_size;
  std::vector<std::uint64_t> xs;
  xs.reserve(keys.size());
  for (auto key : keys) xs.push_back(encode(key, m.prefix_.size(), bytes));
  std::size_t first = 0;
  while (first < keys.size()) {
    const auto target = std::max<std::size_t>(1,
        (keys.size() + segment_budget - 1) / segment_budget);
    std::size_t end = std::min(keys.size(), first + target);
    // Equal encoded coordinates cannot be sent to different predictors.
    while (end < keys.size() && xs[end] == xs[end - 1]) ++end;
    Segment s;
    s.x_min = xs[first]; s.x_max = xs[end - 1];
    s.y_min = static_cast<std::uint32_t>(first / block_size);
    s.y_max = static_cast<std::uint32_t>((end - 1) / block_size);
    s.intercept = s.x_min == s.x_max
        ? (double(s.y_min) + s.y_max) / 2 : double(s.y_min);
    s.slope = s.x_min == s.x_max ? 0
        : double(s.y_max - s.y_min) / double(s.x_max - s.x_min);
    for (std::size_t i = first; i < end; ++i) {
      const auto p = m.predict(s, xs[i]), y = i / block_size;
      s.radius = std::max(s.radius,
          static_cast<std::uint32_t>(p > y ? p - y : y - p));
    }
    m.segments_.push_back(s);
    first = end;
  }
  m.certified_ = std::all_of(m.segments_.begin(), m.segments_.end(),
      [max_error](const Segment& s) { return s.radius <= max_error; });
  // Certify the actual lookup path, not just the segment used during fitting.
  double total = 0, batches = 0;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    auto w = m.window(keys[i]);
    if (i / block_size < w.first || i / block_size >= w.last)
      m.certified_ = false;
    total += w.last - w.first;
    const auto records = std::min(keys.size(), w.last * block_size) - w.first * block_size;
    batches += (records + 15) / 16;
  }
  m.average_blocks_ = total / keys.size();
  m.average_tag_batches_ = batches / keys.size();
  return m;
}
bool BlockModel::accepts_prefix(std::string_view key) const {
  return key.size() >= prefix_.size() && key.substr(0, prefix_.size()) == prefix_;
}
BlockWindow BlockModel::window(std::string_view key) const {
  return locate(key).window;
}
BlockPrediction BlockModel::locate(std::string_view key) const {
  if (!blocks_ || segments_.empty()) return {};
  const auto x = encode(key, prefix_.size(), bytes_);
  const auto& s = segment(x);
  const auto p = predict(s, x);
  return {{p > s.radius ? p - s.radius : 0,
           std::min(blocks_, p + s.radius + 1)}, p};
}
std::size_t BlockModel::owned_bytes() const {
  return segments_.capacity() * sizeof(Segment);
}
double BlockModel::legacy_planning_cost(std::size_t count) const {
  // Explicit, portable cost proxy; these coefficients are not measured ns.
  return 0.2 * bytes_ + average_blocks_ +
      double(owned_bytes()) / std::max<std::size_t>(count, 1) * 0.08;
}
double BlockModel::planning_cost(std::size_t count, const QueryCostWeights& weights,
                                  double visited_batches) const {
  const auto steps = std::ceil(std::log2(double(segments_.size()) + 1));
  return bytes_ * weights.encode_byte + steps * weights.segment_step
      + visited_batches * weights.tag_batch
      + double(owned_bytes()) / std::max<std::size_t>(count, 1) * weights.space_byte;
}

}  // namespace blocklearn
