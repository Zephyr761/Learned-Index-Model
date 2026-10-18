#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace blocklearn {

// Exclusive (non-overlapping) query stages. Directory routing includes its
// separator models and bracket checks; leaf_model only means the leaf model.
enum class LookupStage : std::size_t {
  directory_route, leaf_model, tag_generate, tag_filter, key_verify,
  delta_scan, exact_fallback, other, count
};
constexpr std::size_t lookup_stage_count = static_cast<std::size_t>(LookupStage::count);
inline const char* stage_name(LookupStage stage) {
  constexpr const char* names[] = {"directory_route", "leaf_model", "tag_generate",
      "tag_filter", "key_verify", "delta_scan", "exact_fallback", "other"};
  return names[static_cast<std::size_t>(stage)];
}
struct StageTime { double ns = 0; std::uint64_t calls = 0; };
struct LookupProfile {
  std::array<StageTime, lookup_stage_count> stages{};
  std::uint64_t queries = 0;
  double total_ns = 0;
  LookupProfile& operator+=(const LookupProfile& other) {
    queries += other.queries; total_ns += other.total_ns;
    for (std::size_t i = 0; i < stages.size(); ++i) {
      stages[i].ns += other.stages[i].ns;
      stages[i].calls += other.stages[i].calls;
    }
    return *this;
  }
};

// The ordinary lookup instantiates these empty scopes: no clock reads, timing
// state, per-stage branches, or nullable recorder on the production path.
struct NoLookupTrace {
  struct Scope { Scope(NoLookupTrace&, LookupStage) {} };
};

// Diagnostic only. Nested scopes pause the parent so fallback and key checks
// are not double-counted. Raw times INCLUDE clock/recorder perturbation; never
// subtract a synthetic overhead to manufacture "true" per-stage nanoseconds.
class TimedLookupTrace {
 public:
  using Clock = std::chrono::steady_clock;
  explicit TimedLookupTrace(LookupProfile& profile)
      : profile_(profile), start_(Clock::now()), last_(start_) {
    ++profile_.stages[static_cast<std::size_t>(LookupStage::other)].calls;
  }
  ~TimedLookupTrace() {
    const auto end = Clock::now();
    account(end);
    profile_.total_ns += nanoseconds(end - start_);
    ++profile_.queries;
  }
  TimedLookupTrace(const TimedLookupTrace&) = delete;
  TimedLookupTrace& operator=(const TimedLookupTrace&) = delete;
  class Scope {
   public:
    Scope(TimedLookupTrace& trace, LookupStage stage)
        : trace_(trace), previous_(trace.current_) {
      trace_.switch_to(stage);
      ++trace_.profile_.stages[static_cast<std::size_t>(stage)].calls;
    }
    ~Scope() { trace_.switch_to(previous_); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
   private:
    TimedLookupTrace& trace_;
    LookupStage previous_;
  };
 private:
  static double nanoseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::nano>(duration).count();
  }
  void account(Clock::time_point now) {
    profile_.stages[static_cast<std::size_t>(current_)].ns += nanoseconds(now - last_);
    last_ = now;
  }
  void switch_to(LookupStage stage) { account(Clock::now()); current_ = stage; }
  LookupProfile& profile_;
  Clock::time_point start_, last_;
  LookupStage current_ = LookupStage::other;
};

}  // namespace blocklearn
