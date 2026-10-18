# 第三项优化消融与 ART/HOT 对比

目录短编码已固定开启；预测块优先在两个固定残差版本中开启，在自适应版本中允许区域自行选择。当前只保留三个模型工厂和两个真实基线：

| 简称 | 实现 ID | 规划目标 | 标签 / 访问顺序 |
| --- | --- | --- | --- |
| P0 | `block-residual-delta-shortdir-first` | 原规划器，保留作第三项对照 | 残差标签 / 固定预测块优先 |
| P1 | `block-residual-delta-shortdir-first-cost` | 独立校准的实际查询工作量 | 残差标签 / 固定预测块优先 |
| P2 | `block-adaptive-delta-shortdir-first` | 同 P1 | 每区域选择哈希或学习字节标签，以及左到右或预测块优先 |
| ART | `art` | 真实 armon/libart | 原生接口 |
| HOT | `hot` | 真实 speedskater/HOT | 兼容平台的原生接口 |

三种模型均支持在线插入、局部 delta 合并重训和多层分裂。独立的 block-hash 家族、早期四个模型入口、仅目录/仅预测块开关变体及对应 manifest 已删除，不再参与默认或显式评测。哈希标签原语仍是 P2 的区域候选，不能因为移除独立哈希基线就一并删除。

P0→P1 评价整个新规划器方案，包括实际批次与提前返回目标、特征选择目标、独立校准和有预算候选选择；不是仅替换一个系数的纯单因素实验。P1→P2 评价区域标签种类与访问顺序的联合自适应，不能单独归因于哈希/残差选择。暂不增加额外版本。区域边界与分裂规则未自适应。

## 一条命令跑五个实现

默认不写 `--implementation` 就跑上表五个；`--ablation planner` 是同一固定矩阵的显式写法，不能与 `--implementation` 混用。旧的 `--ablation directory / probe / both` 已移除，传入会报错，不会偷偷替换实现。

Linux 正式静态测试：

```sh
cd /mnt/nas/gezuhao/25_zhw/Learned-Index-Model
CC=gcc CXX=g++ python3 -B src/experiments/run_experiments.py \
  --dataset data/dblp_7498293_maxlen128 --reader raw-lines \
  --output build/linux-dblp-planner-ablation-v1 \
  --mode static --ablation planner --require-hot \
  --operations 5000000 --repeats 5 --seed 42 --miss-ratio 0.1 \
  --latency-samples 10000 --profile-stages --profile-samples 100000 \
  --memory-budget-mib 8192 --timeout 7200
```

将数据路径和输出目录替换为 URL 或其他数据集，可以复用整套配置。每个输出目录必须不存在；`--operations` 是每轮查询数，不是训练样本数，程序不自动抽样或截断数据。预算是数据/负载准备的逻辑预算，不是 OS 内存硬限制，需为索引、训练和 Python 进程留余量。每轮依次运行，避免多组性能测量互相干扰。

只有 ART 可用时，去掉 `--ablation planner --require-hot`，显式选择三个模型和 ART：

```sh
--implementation block-residual-delta-shortdir-first \
--implementation block-residual-delta-shortdir-first-cost \
--implementation block-adaptive-delta-shortdir-first \
--implementation art
```

不支持 HOT 时默认开发模式会诚实记录 unsupported；这不是 HOT 吞吐为零，也不是完整的 ART/HOT 比较。正式 Linux 比较保留 `--require-hot`，缺少原生基线或任一正式结果失败会明确报错。

## 独立校准与冻结

P1/P2 在读取数据或正式查询之前，在目标机上用固定种子的独立合成数据测量原语成本，生成 `planner-costs.txt`。静态、混合、正确性和阶段诊断进程复用同一份文件，`suite.json` 保存并复核其 SHA-256。正式查询与 `--miss-ratio` 不参与调参，区域规划使用固定 90/10 命中先验。

可以显式复用同机器、同编译配置的文件：

```sh
--planner-costs build/此前通过的测试目录/planner-costs.txt
```

不要上传 Mac 或 Debug 的系数来代替 Linux Release 校准。系数文件检查 SIMD 后端与构建类型，但并不自动识别具体 CPU 型号。成本是解释性的工作量代理，不是完整缓存/DRAM 或端到端延迟模拟。机制和限制见 [规划器说明](../../docs/PLANNER_V3.md)。

## 构建与正确性测试

```sh
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DBLOCKLEARN_SANITIZERS=ON
cmake --build build/debug -j 2
ctest --test-dir build/debug --output-on-failure
```

性能入口会另行用 Release 编译，不使用 Debug/sanitizer 二进制测性能。`--cmake /absolute/path/to/cmake` 可指定不在 PATH 中的 CMake；`--benchmark /path/to/learned-index` 指定队友仓库。

ART/HOT 默认源码路径为 `related_works/libart`、`related_works/hot`，可通过 CMake 的 `ART_SOURCE_DIR` / `HOT_SOURCE_DIR` 改变；不要运行上游自己的顶层构建。HOT 要求 GCC、x86-64 AVX2/BMI/BMI2/LZCNT/POPCNT；ARM Mac 不提供原生 HOT 性能结果。Linux 先按 [HOT 验收](../../docs/LINUX_HOT.md) 检查。

本机或服务器先做接入小测：

```sh
python3 -B src/experiments/run_experiments.py \
  --dataset benchmark/learned-index/examples/keys.txt --reader raw-lines \
  --output build/planner-example-suite --mode both --ablation planner \
  --operations 2000 --repeats 1
```

直接运行 `blocklearn_mixed --implementation all` 也会执行五个实现。P1/P2 的直接 C++ 运行需先用同构建的 `blocklearn_calibrate --output NEWFILE` 校准，并设置 `BLOCKLEARN_COST_PROFILE`；Python 统一入口自动完成。

## 混合读写

静态命令换新输出目录，改为 `--mode mixed --initial-fraction 0.8 --insert-ratio 0.05`，去掉 `--profile-stages --profile-samples 100000`，其余五个实现全部保留。也可以 `--mode both` 在同一 suite 中执行两类测试。

初始键为随机选取的 80%，随后插入保留的真实 key，不只测单调追加。实际插入数受保留 key 总数限制，耗尽后转为读；以 CSV 的 reads/inserts 为准。`--miss-ratio` 是读查询的未命中概率。

同步合并、重训和分裂计入插入成本。吞吐与延迟使用不同的新实例；完整正确性另测，正式吞吐包含响应结果写入但不包含答案校验。插入全部延迟采样，读按 `--latency-every` 周期采样。不能只报告读收益而忽略新规划器的构建/维护代价。

## 阶段诊断与结果解释

- `static/report.md`：真正的 QPS、端到端延迟、构建时间和内存分项，与 ART/HOT 比较用此报告。
- `mixed-summary.csv`：混合负载吞吐、读/插入 P50/P99，以及合并、分裂等维护计数。
- `stage-profile.md/.csv`：独立插桩诊断，不能用其时间替代正式吞吐。
- `suite.json`：源码摘要、参数、数据/查询身份、平台、基线 commit、校准摘要及可用性。

阶段诊断只在静态正式评测后执行，复用同一 `rows.bin / queries.bin` 并校验 SHA-256。按 `floor(i*N/K)` 固定位置抽样，所有模型使用相同位置，每条答案校验；每个实现独立进程、构建一次后重复诊断。

| 阶段 | 范围 |
| --- | --- |
| directory_route | 短编码或目录模型、候选定位和边界验证，不含完整回退 |
| leaf_model | 叶公共前缀、编码、选段、预测与窗口 |
| tag_generate | 普通哈希或学习字节标签生成 |
| tag_filter | 基础区 SIMD/标量标签筛选 |
| key_verify | 叶内完整 key 验证 |
| delta_scan | 增量遍历与标签检查，完整验证单列 |
| exact_fallback | 目录或未认证叶区域的精确有序搜索 |
| other | 剩余控制流、记录/结果访问及部分记录器开销 |

阶段时间互斥，累计等于带插桩总时间；包含计时器与计数器开销，不机械扣除空读钟时间。ART/HOT 不插桩，不伪造内部阶段。`--mode mixed --profile-stages` 会拒绝，`both` 只诊断静态部分。

重点观察真实 `tag_batches`、完整 key 比较数、标签生成耗时，以及 `hash_leaves / residual_leaves / predicted_first_leaves` 的区域选择结果。配置数和整份样本计数只报告在 all 行，不伪造 hit/miss 分项。

## 多数据集与负载

对 dblp / imdb / reddit / url / wiki 使用相同五版本矩阵，分别测试：

1. 静态全部命中：`--miss-ratio 0`。
2. 静态 90/10：`--miss-ratio 0.1`。
3. 静态 50/50：`--miss-ratio 0.5`。
4. 混合读写：`--mode mixed --insert-ratio 0.05 --initial-fraction 0.8`。

不根据正式查询结果反向校准或挑成本系数。没有使用热门前缀缓存制造收益。记录 CPU/编译器/VM 配置，自行控制绑核、宿主机负载与换页；当前没有自动绑核或显式预热。内存是容量估计而非独立峰值 RSS。功能测试通过不证明超过 ART/HOT，也不证明论文新颖性。

## 更新服务器

等待当前测试完成再同步源文件，否则源码摘要检查会判失败。使用 Git 同步可同时应用旧 manifest 删除；手工上传只覆盖文件还不够，必须移除 [服务器清单](../../docs/PLANNER_V3.md) 中的九个旧 manifest。不要删除历史结果、数据集或基线 clone。
