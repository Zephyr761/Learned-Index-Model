# 第三项优化：实际工作量规划与区域自适应标签

## 实现范围

这次新增两种完整动态模型，不修改原工厂的规划策略：

| ID | 模型/标签规划目标 | 标签方式 | 扫描顺序 |
| --- | --- | --- | --- |
| `block-residual-delta-shortdir-first` | 原目标，对照 | 残差标签 | 固定预测块优先 |
| `block-residual-delta-shortdir-first-cost` | 校准的实际工作量 | 残差标签 | 固定预测块优先 |
| `block-adaptive-delta-shortdir-first` | 校准的实际工作量 | 每区域哈希或残差 | 每区域从左或预测块优先 |

两种新模型都保留多层短键目录、逐键误差认证、完整 key 验证和增量区维护。区域边界仍按既有 `region_size` 和分裂规则，不实现第二轮的自适应边界。

目标包含编码、模型选段、真实 16 项标签批次、标量尾项、成功返回之前的完整 key 验证及比较字节、哈希/字节特征生成和容量成本。命中批次的标签全部计入，但命中 lane 之后的碰撞不计为完整验证。未命中仍覆盖全部窗口。

每个区域用至多 64 个存储键样本拟合字节位置，另用至多 64 个不重叠样本选择策略；单记录区域除外。至多 16 个局部负样本通过变异产生、检查不存在且位于本区域键范围内。规划固定使用 90/10 命中先验，不读取正式 benchmark 查询或 `--miss-ratio`。候选模型先筛到四个再详细比较，属于有预算规划，不保证全局最优。

在线合并先重新拟合原模型并完整认证；每八次快速合并或成本估计恶化超过 25% 时重新选择当前模型的标签/顺序，重新拟合失败才全量规划。变更配置时 delta 已合并清空，随后重建全部基础标签，不混用旧标签。新键仍存入稳定拥有存储；空键、NUL、高位字节、前缀关系和值为零均保持精确语义。

## 独立校准

统一入口会在加载数据/查询之前，以固定种子合成独立数据，测量原语成本并生成 `planner-costs.txt`。文件在整轮静态、正确性、混合和阶段诊断中复用；`suite.json` 记录 SHA-256，结束时再次检查。

系数文件记录 SIMD 目标与 Debug/Release 类型，类型不匹配会拒绝。Linux 上的校准程序与模型核心使用同一个 SIMD 后端；不要上传 Mac 或 Debug 构建生成的系数来代替 Linux Release 校准。默认每个 suite 在目标机上重新校准；想控制不同数据集之间的校准噪声，可显式复用同一机器、同一编译配置下的既有文件：

```sh
--planner-costs build/此前通过的测试目录/planner-costs.txt
```

校准只接受输出文件、迭代次数或既有成本文件验证，不接受 `rows`、`queries` 或数据集参数。原语系数是规划代理，不是端到端延迟；额外相对 64B 窗口是布局代理，`space_byte` 是 32B 标签批次成本摊到字节的容量权衡，不宣称精确预测 DRAM/cache miss 或 RSS。测量差分使用正下限，仍需真实多数据集验证收益。

## 服务器需要更新哪些文件

服务器目录：`/mnt/nas/gezuhao/25_zhw/Learned-Index-Model`。

**必须等服务器当前测试结束再覆盖源码**，否则正在运行的 suite 会因源码摘要变化被判为失败。不要覆盖服务器数据、基线 clone 或既有测试结果；不需要上传本机 `build/`、CMake 缓存、`__pycache__/`、`.pyc`、`.DS_Store` 或校准系数。

最不容易漏文件的方式：同步整个 `src/`（排除上述缓存）、根 `CMakeLists.txt`，以及下列测试。仅需运行 benchmark 时，必需的源码最小清单是：

```text
src/CMakeLists.txt
src/block_model.h
src/block_model.cpp
src/learned_index.h
src/learned_index.cpp
src/query_cost.h                                  新增
src/query_cost.cpp                                新增
src/region_planner.h                              新增
src/region_planner.cpp                            新增
src/benchmark/factories.h
src/benchmark/model_factories.cpp
src/benchmark/model_registry.h
src/benchmark/profile_runner.cpp
src/benchmark/calibrate_planner.cpp                新增
src/experiments/run_experiments.py
src/experiments/stage_profile.py
src/experiments/block-residual-delta-shortdir-first-cost/benchmark-adapter.json   新增
src/experiments/block-adaptive-delta-shortdir-first/benchmark-adapter.json        新增
```

另外更新原十个模型 manifest 的 `source_identity` 到 v3；它们的工厂/开关行为不变：

```text
src/experiments/block-plain/benchmark-adapter.json
src/experiments/block-hash/benchmark-adapter.json
src/experiments/block-residual/benchmark-adapter.json
src/experiments/block-residual-delta/benchmark-adapter.json
src/experiments/block-hash-shortdir/benchmark-adapter.json
src/experiments/block-hash-first/benchmark-adapter.json
src/experiments/block-hash-shortdir-first/benchmark-adapter.json
src/experiments/block-residual-delta-shortdir/benchmark-adapter.json
src/experiments/block-residual-delta-first/benchmark-adapter.json
src/experiments/block-residual-delta-shortdir-first/benchmark-adapter.json
```

构建和测试一并更新：

```text
CMakeLists.txt
tests/work_planner_test.cpp                        新增
tests/planner_profile_test.py                      新增
tests/ablation_contract_test.py
tests/profile_runner_test.py
```

说明文件（不影响可执行接口）：

```text
README.md
src/DESIGN.md
src/experiments/README.md
docs/PLANNER_V3.md
```

本轮未改 `src/benchmark/art_baseline.cpp`、`hot_baseline.cpp`、`mixed_runner.cpp`、原残差标签编码文件或队友 benchmark。它们已有前两轮代码的服务器副本可保留；不要用旧二进制与新源码混跑。

## Linux 验证

根目录重新构建，不复用 Mac 缓存：

```sh
cd /mnt/nas/gezuhao/25_zhw/Learned-Index-Model
CC=gcc CXX=g++ cmake -S . -B build/linux-planner-v3-debug \
  -DCMAKE_BUILD_TYPE=Debug -DBLOCKLEARN_SANITIZERS=ON -DBLOCKLEARN_REQUIRE_HOT=ON
cmake --build build/linux-planner-v3-debug -j 2
ctest --test-dir build/linux-planner-v3-debug --output-on-failure
```

先通过小数据接入，再用完整数据跑六个版本，区分普通哈希对照、原模型、新目标和区域自适应的收益：

```sh
CC=gcc CXX=g++ python3 -B src/experiments/run_experiments.py \
  --dataset data/dblp_7498293_maxlen128 --reader raw-lines \
  --output build/linux-dblp-planner-v3 --mode static --require-hot \
  --implementation block-hash-shortdir-first \
  --implementation block-residual-delta-shortdir-first \
  --implementation block-residual-delta-shortdir-first-cost \
  --implementation block-adaptive-delta-shortdir-first \
  --implementation art --implementation hot \
  --operations 5000000 --repeats 5 --seed 42 --miss-ratio 0.1 \
  --profile-stages --profile-samples 100000 \
  --memory-budget-mib 8192 --timeout 7200
```

输出目录必须不存在；文件实际位于解压子目录时调整 `--dataset`。统一入口自动以 Release 编译和校准，不使用上面的 Debug 二进制/系数测性能。

读写测试用新输出目录，改为 `--mode mixed --initial-fraction 0.8 --insert-ratio 0.05`，去掉 `--profile-stages`，并去掉不支持插入的 `block-hash-shortdir-first`。五个动态版本共用操作流；为与静态实验复用系数，可添加上述 `--planner-costs`。

真实吞吐/延迟看 `static/report.md` 和 `mixed-summary.csv`。`stage-profile.md/.csv` 另有 `tag_batches`、`hash_leaves`、`residual_leaves`、`predicted_first_leaves`，便于确认每区域实际选了什么以及减少了多少工作；插桩时间不用于性能排名。
