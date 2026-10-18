#pragma once

#include "key_bytes.h"
#include <optional>
#include <vector>

namespace blocklearn {

// Exact ordered routing over compact, non-injective 63-bit suffix encodings.
// Unequal codes establish the full-key order. Equal codes ALWAYS consult the
// actual fence, including EOS/prefix cases and differences beyond byte seven.
class ShortDirectory {
 public:
  void rebuild(const std::vector<std::string_view>& fences) {
    codes_.clear(); prefix_ = {}; useful_ = false;
    if (fences.empty()) return;
    prefix_ = fences.front().substr(0, common_prefix(fences.front(), fences.back()));
    codes_.reserve(fences.size());
    std::size_t distinct = 0;
    for (auto fence : fences) {
      const auto code = encode(fence, prefix_.size(), 7);
      if (codes_.empty() || code != codes_.back()) ++distinct;
      codes_.push_back(code);
    }
    // Keep the old model/exact route when fewer than a quarter of the codes
    // are distinct. This is a build-time choice, not a query-hotspot cache.
    useful_ = distinct * 4 >= fences.size();
  }
  bool useful() const { return useful_; }
  std::size_t owned_bytes() const { return codes_.capacity() * sizeof(std::uint64_t); }

  template<class FullLess>
  std::optional<std::size_t> predecessor(const std::vector<std::string_view>& fences,
                                        std::string_view key, FullLess full_less) const {
    if (!useful_ || codes_.size() != fences.size()) return std::nullopt;
    if (fences.size() == 1) return 0;
    // Every fence has this exact prefix. A different/shorter query prefix
    // establishes its order against ALL fences, including range extremes.
    const auto prefix_order = compare(key.substr(0, prefix_.size()), prefix_);
    if (prefix_order < 0) return 0;
    if (prefix_order > 0) return fences.size() - 1;
    const auto code = encode(key, prefix_.size(), 7);
    auto query_less = [&](std::size_t i) {
      return code == codes_[i] ? full_less(key, fences[i]) : code < codes_[i];
    };
    std::size_t lo = 0, hi = fences.size();
    while (lo < hi) {
      const auto mid = lo + (hi - lo) / 2;
      if (query_less(mid)) hi = mid;
      else lo = mid + 1;
    }
    const auto p = lo ? lo - 1 : 0;
    // Retain the true predecessor bracket. Numeric comparisons are exact
    // only for unequal codes; ambiguous comparisons use the original fences.
    if ((p == 0 || !query_less(p)) &&
        (p + 1 == fences.size() || query_less(p + 1))) return p;
    return std::nullopt;  // caller retains its complete original routing path
  }

  bool valid_for(const std::vector<std::string_view>& fences) const {
    if (codes_.size() != fences.size()) return false;
    if (fences.empty()) return !useful_ && prefix_.empty();
    const auto prefix = fences.front().substr(0, common_prefix(fences.front(), fences.back()));
    if (prefix != prefix_) return false;
    for (std::size_t i = 0; i < fences.size(); ++i)
      if (codes_[i] != encode(fences[i], prefix_.size(), 7)) return false;
    return true;
  }
 private:
  std::vector<std::uint64_t> codes_;
  std::string_view prefix_;
  bool useful_ = false;
};

}  // namespace blocklearn
