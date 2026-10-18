// Independent synthetic primitive calibration. Deliberately no dataset/query
// arguments, no benchmark reader, no model-selection feedback from test QPS.
#include "block_model.h"
#include "key_bytes.h"
#include "query_cost.h"
#include "residual_tags.h"
#include "tag_probe.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

using namespace blocklearn;
namespace {
volatile std::uint64_t calibration_sink = 0;
template<class Operation> double measure(std::size_t iterations, Operation operation) {
  std::vector<double> rounds;
  for (unsigned repeat = 0; repeat < 5; ++repeat) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) calibration_sink = operation(i);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    rounds.push_back(std::chrono::duration<double, std::nano>(elapsed).count() / iterations);
  }
  std::sort(rounds.begin(), rounds.end());
  return rounds[rounds.size() / 2];
}
double positive(double value) { return std::max(0.01, value); }
std::uint64_t number(const std::string& text) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("invalid calibration integer");
  return std::stoull(text);
}
}
int main(int argc, char** argv) {
  try {
    std::string output, validate;
    std::size_t iterations = 32768;
    for (int i = 1; i < argc; ++i) {
      const std::string arg(argv[i]);
      if (arg == "--help") {
        std::cout << "--output NEW-profile-file [--iterations 32768] | --validate profile-file; independent synthetic calibration only\n";
        return 0;
      }
      if (++i == argc) throw std::invalid_argument("missing calibration argument");
      if (arg == "--output") output = argv[i];
      else if (arg == "--validate") validate = argv[i];
      else if (arg == "--iterations") iterations = number(argv[i]);
      else throw std::invalid_argument("unknown calibration option: " + arg);
    }
    if (!validate.empty()) {
      if (!output.empty()) throw std::invalid_argument("choose calibration or validation, not both");
      load_cost_profile(validate);
      std::cout << "Valid independent synthetic cost profile: " << validate << '\n'; return 0;
    }
    if (output.empty() || iterations < 1024 || iterations > (1U << 24))
      throw std::invalid_argument("output required; iterations must be 1024..2^24");
    CostProfile profile; profile.origin = "independent-synthetic-v1";
    profile.seed = 0x424c4b43414cULL; profile.iterations = iterations;
    std::mt19937_64 rng(profile.seed);
    constexpr std::size_t count = 1024;
    std::vector<std::string> short_keys, long_keys, short_copies, long_copies;
    for (std::size_t i = 0; i < count; ++i) {
      std::string key(256, '\0');
      for (auto& byte : key) byte = char(rng());
      short_keys.push_back(key.substr(0, 32)); long_keys.push_back(std::move(key));
    }
    short_copies = short_keys; long_copies = long_keys;  // distinct addresses
    const auto overhead = measure(iterations, [](std::size_t i) { return std::uint64_t(i); });
    auto time = [&](auto operation) { return positive(measure(iterations, operation) - overhead); };
    auto& w = profile.weights;
    const auto e2 = time([&](std::size_t i) { return encode(long_keys[i % count], 0, 2); });
    const auto e7 = time([&](std::size_t i) { return encode(long_keys[i % count], 0, 7); });
    w.encode_byte = positive((e7 - e2) / 5);
    std::vector<std::string_view> sorted;
    for (auto& key : short_keys) sorted.push_back(key);
    std::sort(sorted.begin(), sorted.end(), less);
    const auto one = BlockModel::fit(sorted, 16, 4, 1, 64);
    const auto eight = BlockModel::fit(sorted, 16, 4, 8, 64);
    const auto p1 = time([&](std::size_t i) { return one.locate(sorted[i % count]).predicted; });
    const auto p8 = time([&](std::size_t i) { return eight.locate(sorted[i % count]).predicted; });
    w.segment_step = positive((p8 - p1) / 3);
    std::vector<std::uint16_t> tags(count * 16);
    for (auto& value : tags) value = std::uint16_t(rng());
    w.tag_batch = time([&](std::size_t i) { return tag_mask(tags.data() + (i % count) * 16, 16, std::uint16_t(i)); });
    const auto s1 = time([&](std::size_t i) { return tag_mask(tags.data() + (i % count) * 16, 1, std::uint16_t(i)); });
    const auto s15 = time([&](std::size_t i) { return tag_mask(tags.data() + (i % count) * 16, 15, std::uint16_t(i)); });
    w.scalar_tag = positive((s15 - s1) / 14);
    const auto v32 = time([&](std::size_t i) { return std::uint64_t(short_keys[i % count] == short_copies[i % count]); });
    const auto v256 = time([&](std::size_t i) { return std::uint64_t(long_keys[i % count] == long_copies[i % count]); });
    w.verify_byte = positive((v256 - v32) / 224);
    w.key_verify = positive(v32 - 32 * w.verify_byte);
    const auto h32 = time([&](std::size_t i) { return hash_tag(short_keys[i % count], 0); });
    const auto h256 = time([&](std::size_t i) { return hash_tag(long_keys[i % count], 0); });
    w.hash_byte = positive((h256 - h32) / 224);
    w.hash_base = positive(h32 - 32 * w.hash_byte);
    TagPlan zero, near, far;
    near.count = far.count = 4;
    near.offsets = {{0, 1, 2, 3}}; far.offsets = {{0, 64, 128, 224}};
    w.residual_base = time([&](std::size_t i) { return zero.tag(long_keys[i % count]); });
    const auto adjacent = time([&](std::size_t i) { return near.tag(long_keys[i % count]); });
    const auto separated = time([&](std::size_t i) { return far.tag(long_keys[i % count]); });
    w.feature_read = positive((adjacent - w.residual_base) / 4);
    w.feature_line = positive((separated - adjacent) / 3);
    // Capacity tradeoff proxy: charge bytes/key in units of a measured 32-byte
    // tag batch. This is NOT a calibrated prediction of DRAM misses or RSS.
    w.space_byte = w.tag_batch / 32;
    std::ostringstream text; write_cost_profile(text, profile);
    const auto contents = text.str();
    std::FILE* file = std::fopen(output.c_str(), "wx");  // never overwrite
    if (!file) throw std::runtime_error("cannot create new cost profile: " + output);
    const bool written = std::fwrite(contents.data(), 1, contents.size(), file) == contents.size();
    const bool closed = std::fclose(file) == 0;
    if (!written || !closed) throw std::runtime_error("failed to write calibration profile");
    std::cout << "Independent synthetic primitive calibration: " << output
              << "\nCoefficients are planning proxies, not measured end-to-end lookup ns.\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
