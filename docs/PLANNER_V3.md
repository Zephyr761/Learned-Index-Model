# 第三项优化：实际工作量规划与区域自适应标签

## 当前保留的消融矩阵

目录短编码已接受为共同配置，不再保留关闭目录或仅开一项的独立工厂。当前只有以下三个动态模型以及 ART/HOT，默认一次共五个实现，不再运行独立 block-hash：

| ID | 模型/标签规划目标 | 标签方式 | 扫描顺序 |
| --- | --- | --- | --- |
| `block-residual-delta-shortdir-first` | 原目标，必要的消融参照 | 残差标签 | 固定预测块优先 |
| `block-residual-delta-shortdir-first-cost` | 校准的实际工作量 | 残差标签 | 固定预测块优先 |
| `block-adaptive-delta-shortdir-first` | 校准的实际工作量 | 每区域哈希或残差 | 每区域从左或预测块优先 |

第一项到第二项评价新规划器整体，不仅是一处系数替换；第二项到第三项评价标签种类与扫描顺序的联合自适应，不能单独归因为某一个开关。自适应版本的 `-first` 表示允许区域选择该顺序，不强制所有区域采用。哈希标签生成原语仍保留为区域候选。

## 机制与范围

三种模型都保留多层短键目录、逐键误差认证、完整 key 验证和增量区维护。区域边界仍按既有 `region_size` 和分裂规则，不实现第二轮自适应边界。

新目标包含编码、模型选段、真实 16 项标签批次、标量尾项、成功返回之前的完整 key 验证及比较字节、哈希/字节特征生成和容量成本。命中批次的标签全部计入，但命中 lane 之后的碰撞不计为完整验证。未命中仍覆盖全部窗口。

每个区域用至多 64 个存储键样本拟合字节位置，另用至多 64 个不重叠样本选择策略；单记录区域除外。每组至多 16 个局部负样本通过变异产生、检查不存在且位于本区域键范围内。规划固定使用 90/10 命中先验，不读取正式 benchmark 查询或 `--miss-ratio`。候选模型先筛到四个再详细比较，属于有预算规划，不保证全局最优。

在线合并先重新拟合原模型并完整认证；每八次快速合并或成本估计恶化超过 25% 时重新选择当前模型的标签/顺序，重新拟合失败才全量规划。变更配置时 delta 已合并清空，随后重建全部基础标签，不混用旧标签。新键仍存入稳定拥有存储；空键、NUL、高位字节、前缀关系和值为零均保持精确语义。

## 独立校准

统一入口在加载数据/查询之前，以固定种子合成独立数据，测量原语成本并生成 `planner-costs.txt`。文件在整轮静态、正确性、混合和阶段诊断中复用；`suite.json` 记录 SHA-256，结束时再次检查。

系数文件记录 SIMD 目标与 Debug/Release 类型，类型不匹配会拒绝。Linux 上的校准程序与模型核心使用同一个 SIMD 后端；不要上传 Mac 或 Debug 系数。默认每个 suite 在目标机重新校准；想控制跨数据集校准噪声，可复用同机器、同编译配置的既有独立文件：

```sh
--planner-costs build/此前通过的测试目录/planner-costs.txt
```

校准不接受数据集或查询参数。原语系数是规划代理，不是端到端延迟；额外相对 64B 窗口是布局代理，`space_byte` 是 32B 标签批次成本摊到字节的容量权衡，不精确预测 DRAM/cache miss 或 RSS。差分测量有正下限，仍需多数据集实测。CPU 型号不自动校验，跨机器不应复用。

## 这次清理需要向服务器更新的文件

服务器目录：`/mnt/nas/gezuhao/25_zhw/Learned-Index-Model`。

等待当前服务器测试结束，再同步源文件。Git 同步可应用删除；手工上传时同步以下修改，并移除后面的九个旧 manifest。只覆盖文件会留下过期入口。

```text
CMakeLists.txt
src/benchmark/factories.h
src/benchmark/model_factories.cpp
src/benchmark/model_registry.h
src/benchmark/mixed_runner.cpp
src/experiments/run_experiments.py
src/experiments/stage_profile.py
tests/blocklearn_test.cpp
tests/query_optimizations_test.cpp
tests/experiment_contract_test.py
tests/ablation_contract_test.py
tests/profile_runner_test.py
```

删除的是以下九个 manifest 文件，不是数据或历史结果目录：

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
```

说明文件：`README.md`、`src/DESIGN.md`、`src/experiments/README.md`、本文件、`docs/LINUX_HOT.md`。

若服务器还没有第三项规划器，应同步完整 `src/`、根 CMake 和 `tests/`，包括 `query_cost.*`、`region_planner.*`、`benchmark/calibrate_planner.cpp` 及三个当前模型 manifest，并应用上述删除。不要上传 `build/`、数据、基线 clone、CMake 缓存、`__pycache__/`、`.pyc`、`.DS_Store` 或 Mac 系数。已删除文件可从 Git 历史恢复；历史实验产物没有删除。

## Linux 正确性

```sh
cd /mnt/nas/gezuhao/25_zhw/Learned-Index-Model
CC=gcc CXX=g++ cmake -S . -B build/linux-planner-debug \
  -DCMAKE_BUILD_TYPE=Debug -DBLOCKLEARN_SANITIZERS=ON -DBLOCKLEARN_REQUIRE_HOT=ON
cmake --build build/linux-planner-debug -j 2
ctest --test-dir build/linux-planner-debug --output-on-failure
```

## 五版本正式消融

```sh
CC=gcc CXX=g++ python3 -B src/experiments/run_experiments.py \
  --dataset data/dblp_7498293_maxlen128 --reader raw-lines \
  --output build/linux-dblp-planner-ablation-v1 --mode static \
  --ablation planner --require-hot \
  --operations 5000000 --repeats 5 --seed 42 --miss-ratio 0.1 \
  --profile-stages --profile-samples 100000 \
  --memory-budget-mib 8192 --timeout 7200
```

输出目录必须不存在；数据位于解压子目录时调整路径。统一入口以 Release 编译和校准，不使用 Debug 二进制/系数测性能。

读写另开输出目录，改为 `--mode mixed --initial-fraction 0.8 --insert-ratio 0.05`，去掉 `--profile-stages --profile-samples 100000`。五个实现全部保留，可显式复用静态实验的 Release 系数。实际插入比例受可插入 key 总数限制。

真实吞吐/延迟看 `static/report.md` 和 `mixed-summary.csv`。`stage-profile.md/.csv` 另有 `tag_batches`、`hash_leaves`、`residual_leaves`、`predicted_first_leaves`，用于观察区域配置和实际工作；插桩时间不用于性能排名。完整评估口径见 [实验说明](../src/experiments/README.md)。
