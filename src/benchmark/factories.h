#pragma once
#include <index_contract.h>

std::unique_ptr<string_index_bench::v1::Index> make_art_baseline(
    const string_index_bench::v1::Parameters&);
std::unique_ptr<string_index_bench::v1::Index> make_hot_baseline(
    const string_index_bench::v1::Parameters&);

std::unique_ptr<string_index_bench::v1::Index> make_block_residual_delta_shortdir_first(
    const string_index_bench::v1::Parameters&);
std::unique_ptr<string_index_bench::v1::Index> make_block_residual_delta_shortdir_first_cost(
    const string_index_bench::v1::Parameters&);
std::unique_ptr<string_index_bench::v1::Index> make_block_adaptive_delta_shortdir_first(
    const string_index_bench::v1::Parameters&);
