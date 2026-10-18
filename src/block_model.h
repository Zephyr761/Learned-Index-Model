#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace blocklearn {
struct QueryCostWeights;

struct BlockWindow {
  std::size_t first = 0;  // inclusive block index
  std::size_t last = 0;   // exclusive block index
};
struct BlockPrediction {
  BlockWindow window;
  std::size_t predicted = 0;
};

class BlockModel {
 public:
  struct Segment {
    std::uint64_t x_min = 0, x_max = 0;
    double slope = 0, intercept = 0;
    std::uint32_t y_min = 0, y_max = 0, radius = 0;
  };
  static BlockModel fit(const std::vector<std::string_view>& keys,
                        std::size_t block_size, unsigned bytes,
                        std::size_t segments, std::size_t max_error);
  BlockWindow window(std::string_view key) const;
  BlockPrediction locate(std::string_view key) const;
  bool accepts_prefix(std::string_view key) const;
  bool certified() const { return certified_; }
  std::size_t block_size() const { return block_size_; }
  std::size_t prefix_size() const { return prefix_.size(); }
  unsigned encoding_bytes() const { return bytes_; }
  std::size_t segment_count() const { return segments_.size(); }
  std::size_t owned_bytes() const;
  double average_blocks() const { return average_blocks_; }
  double average_tag_batches() const { return average_tag_batches_; }
  // The original controls retain their original objective explicitly.
  double legacy_planning_cost(std::size_t count) const;
  double planning_cost(std::size_t count, const QueryCostWeights&,
                       double visited_batches) const;
  const std::vector<Segment>& segments() const { return segments_; }

 private:
  std::size_t predict(const Segment&, std::uint64_t) const;
  const Segment& segment(std::uint64_t) const;
  std::string_view prefix_;
  std::vector<Segment> segments_;
  std::size_t block_size_ = 32, blocks_ = 0;
  unsigned bytes_ = 2;
  bool certified_ = false;
  double average_blocks_ = 0;
  double average_tag_batches_ = 0;
};

}  // namespace blocklearn
