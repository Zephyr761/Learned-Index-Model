#include "block_model.h"
#include "learned_index.h"
#include "residual_tags.h"
#include "tag_probe.h"
#include "key_bytes.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

static void check(bool v, const char* s) { if (!v) throw std::runtime_error(s); }
int main() {
  try {
    std::mt19937_64 rng(19);
    for (unsigned trial = 0; trial < 1000; ++trial) {
      std::uint16_t tags[16]; const auto wanted = std::uint16_t(rng());
      std::uint32_t expected = 0;
      for (unsigned i = 0; i < 16; ++i) {
        tags[i] = rng() % 3 ? std::uint16_t(rng()) : wanted;
        expected |= std::uint32_t(tags[i] == wanted) << i;
      }
      for (unsigned count = 0; count <= 16; ++count)
        check(blocklearn::tag_mask(tags, count, wanted) == (expected & ((1U << count) - 1)), "SIMD/scalar tag mask differs");
    }
    std::vector<std::string> stored;
    for (unsigned i = 0; i < 512; ++i) {
      std::string k = "prefix/";
      k.push_back(char('a' + i / 32)); k += std::string(15, 'x');
      k.push_back(char(i % 32)); stored.push_back(k);
    }
    std::vector<std::string_view> keys;
    for (const auto& k : stored) keys.push_back(k);
    for (unsigned bytes : {2U, 4U, 7U})
      for (std::size_t block : {1U, 16U, 32U, 64U})
        for (std::size_t segments : {1U, 2U, 4U, 8U}) {
          const auto model = blocklearn::BlockModel::fit(keys, block, bytes, segments, 1);
          // Windows remain valid even for a rejected fit; rejection only means
          // its cost/error budget is unsuitable for the fast path.
          for (std::size_t i = 0; i < keys.size(); ++i) {
            const auto w = model.window(keys[i]);
            check(w.first <= i / block && i / block < w.last, "bad model certificate");
          }
        }
    const auto model = blocklearn::BlockModel::fit(keys, 32, 2, 1, 1);
    check(model.certified(), "test model did not fit");
    const auto learned = blocklearn::learn_tags(keys, model, 2);
    blocklearn::TagPlan length_only;
    check(learned.count && learned.candidate_matches < blocklearn::evaluate_tags(keys, model, length_only),
          "residual selector did not reduce ambiguity");
    // Force tag collisions: no selected bytes, all equal-length keys share a
    // tag. Exact full-string verification must still reject arbitrary misses.
    std::vector<string_index_bench::v1::Row> rows;
    for (std::size_t i = 0; i < keys.size(); ++i) rows.push_back({std::string(keys[i]), i});
    blocklearn::Options options; options.tag_features = 0; options.region_size = 64;
    options.directory_fanout = 4; options.delta_capacity = 1;
    blocklearn::LearnedIndex index(options); index.bulk_load(rows);
    for (std::size_t i = 0; i < rows.size(); ++i) {
      check(index.find(rows[i].key) == i, "collision lost existing key");
      auto absent = rows[i].key; absent.back() = char(200);
      check(!index.find(absent), "collision became false positive");
    }
    index.insert("", 0); index.insert("zzzz", UINT64_MAX); index.validate();
    check(index.find("") == 0 && index.find("zzzz") == UINT64_MAX, "out-of-domain insertion");
    std::cout << "model certificates, residual feature selection, forced collisions, SIMD masks passed\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
