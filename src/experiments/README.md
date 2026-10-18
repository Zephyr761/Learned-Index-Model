# 四个消融与 ART/HOT 对比

| ID | 验证方式 | 在线插入 |
| --- | --- | --- |
| `block-plain`（E1） | 模型块窗口内完整 key 二分 | unsupported |
| `block-hash`（E2） | 相同模型，普通完整后缀哈希标签过滤，再精确比较 | unsupported |
| `block-residual`（E3） | 相同模型，残差感知字节标签过滤，再精确比较 | unsupported |
| `block-residual-delta`（E4） | E3 + 区域局部 delta / 同步合并重训 / 多层分裂 | 支持 |
| `art` | 真实 armon/libart | 支持 |
| `hot` | 真实 speedskater/HOT single-threaded | 兼容平台支持 |

四个模型共享同一结构规划，避免把区域大小或训练预算差异误认为标签收益。E3/E4 静态路径相同；E4 的意义主要在混合读写实验。

第三项优化另有 `block-residual-delta-shortdir-first-cost`（新目标、固定残差标签）和 `block-adaptive-delta-shortdir-first`（新目标、区域自选标签/顺序）。旧 ID 继续作为原规划器对照；不要认为更新源码就自动选中了新版本。新入口在读取数据/查询之前自动做独立合成原语校准、冻结系数，也可用 `--planner-costs FILE` 复用同目标、同构建类型的独立配置。详细机制、Linux 六版本命令和服务器更新文件清单见 [规划器 v3](../../docs/PLANNER_V3.md)。

## 目录短编码 / 预测块优先：三组开关消融

`block-hash` 和 `block-residual-delta` 各支持三个固定配置后缀：

| `--ablation` | 自动加入的 ID 后缀 | 目录短编码 | 预测块优先 |
| --- | --- | --- | --- |
| `directory` | `-shortdir` | 开 | 关 |
| `probe` | `-first` | 关 | 开 |
| `both` | `-shortdir-first` | 开 | 开 |

原 ID 是两项都关闭的同版本对照。`--ablation` 自动在所选的每个上述模型后插入对应优化版本，ART/HOT 不改变。因此下面每组实际跑五个实现：普通哈希对照及其优化版、完整动态模型对照及其优化版、原生 ART。同一组共用冻结的训练数据与查询；三组同种子和参数，输出 SHA-256 可核对。

在项目根目录，先设置共享参数（zsh/bash）：

```sh
ABLATION_ARGS=(
  --cmake /private/tmp/blocklearn-build-tools.HyImkg/venv/bin/cmake
  --dataset data/dblp_7498293_maxlen128 --reader raw-lines
  --mode static
  --implementation block-hash --implementation block-residual-delta --implementation art
  --operations 5000000 --repeats 5 --seed 42 --miss-ratio 0.1
  --latency-samples 10000 --profile-stages --profile-samples 100000
  --memory-budget-mib 8192 --timeout 7200
)
```

三条命令依次运行：

```sh
python3 -B src/experiments/run_experiments.py "${ABLATION_ARGS[@]}" --ablation directory --output build/dblp-ablation-directory-v1
python3 -B src/experiments/run_experiments.py "${ABLATION_ARGS[@]}" --ablation probe --output build/dblp-ablation-probe-v1
python3 -B src/experiments/run_experiments.py "${ABLATION_ARGS[@]}" --ablation both --output build/dblp-ablation-both-v1
```

CMake 路径是当前验证所用的临时安装，可能被系统清理；有正式安装后去掉该参数或替换路径。每个输出目录必须不存在，再跑时改后缀。每轮使用完整 DBLP，不是抽 500 万条训练记录；预算不是 OS 内存上限，需按可用 RAM 留余量。先跑完一条再跑下一条，避免并行测量相互干扰。换 URL 等数据集时同时改数据路径和输出目录。

看各组 `static/report.md` 的 QPS、延迟和内存估计，比较**本组**新旧版本与 ART，不只对照历史最好值。辅助看 `stage-profile.md/.csv`：目录优化应减少完整 key 比较/目录回退；预测块优先应减少命中时扫描的标签批次，但未命中仍检查完整认证窗口。这些是机制假设，不保证最终吞吐提高。短编码区分度差时使用原路径；编码相同仍精确比较，不截断 key。

两项开关的总效应及各单项效应可由三组结果比较；`both` 不应预设等于两个单项收益之和。所有版本保留相同的模型/标签学习代价函数，本次没有加入新的代价模型或学习块访问次序。

动态写入检查可另开目录，选择 `--mode mixed --implementation block-residual-delta --implementation art --ablation both --initial-fraction 0.8 --insert-ratio 0.05`，不加 `--profile-stages`。普通哈希家族依旧不支持插入，不能作为混合写入性能结果。

## 构建和正确性

在项目根目录运行：

```sh
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DBLOCKLEARN_SANITIZERS=ON
cmake --build build/debug -j 2
ctest --test-dir build/debug --output-on-failure
```

正式性能比较不要使用 sanitizers 或 Debug。x86 Linux 推荐 GCC，并强制 HOT 可用：

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DBLOCKLEARN_REQUIRE_HOT=ON
cmake --build build/release -j 2
```

ART/HOT 默认源码路径分别是 `related_works/libart`、`related_works/hot`，可用 CMake 的 `ART_SOURCE_DIR` / `HOT_SOURCE_DIR` 指定其他 clone。不要运行它们自己的顶层 CMake，避免引入额外依赖和覆盖本项目编译选项。

HOT 需要 GCC 和 x86-64 AVX2/BMI/BMI2/LZCNT/POPCNT；当前 ARM Mac 只能验证不可用分支。上游源码含 GCC 扩展，Apple Clang 不能直接编译。CPU、编译器或源码不满足时，所有 HOT 操作诚实声明 unsupported，不使用替代结构。不支持并不等于 HOT 性能为 0。

`--require-hot` 用于正式比较，缺少原生 HOT 时立即失败。默认功能开发允许缺少 HOT，报告保留 unavailable 原因。

## 一键静态和混合测试

```sh
python3 -B src/experiments/run_experiments.py \
  --dataset benchmark/learned-index/examples/keys.txt --reader raw-lines \
  --output build/example-suite --mode both --operations 2000 --repeats 1
```

这只是接入测试，不是性能结论。`--output` 必须不存在，不覆盖旧结果。

项目中无扩展名的 `data/dblp_7498293_maxlen128` 和 `data/url_19715903_maxlen128` 已确认是文本，应显式选择 `raw-lines`，不要因为名字像旧 TLI 就按二进制读取。其余文件也应先检查格式。

```sh
CC=gcc CXX=g++ python3 -B src/experiments/run_experiments.py \
  --dataset data/url_19715903_maxlen128 --reader raw-lines \
  --output build/url-readwrite --mode both --require-hot \
  --operations 1000000 --repeats 5 --miss-ratio 0.1 \
  --initial-fraction 0.8 --insert-ratio 0.05 --memory-budget-mib 8192
```

内存预算是示例，按机器调整；程序不自动抽样、截断长 key 或删除 NUL。百万操作也不是所有规模的充分测量，最终要扩大操作数、重复次数并控制机器噪声。

`--cmake /absolute/path/to/cmake` 可指定临时或非 PATH 安装。`--benchmark /path/to/learned-index` 可接队友的另一份 clone。可重复 `--implementation` 选择子集。

## 查询分阶段计时（独立诊断，不污染正式吞吐）

给静态测试添加 `--profile-stages --profile-samples 10000`。阶段诊断只在正式静态评测之后、独立进程和索引实例内运行；正式 QPS 仍使用未插入阶段计时的原 runner。重复次数沿用 `--repeats`，诊断进程只构建一次模型，随后重复同一抽样查询。

它直接读取静态评测生成的 `static/interchange/rows.bin` / `queries.bin` 并核对 SHA-256，不再重新生成负载。若查询总数为 N、抽样数为 K（不超过 N），选择位置 `floor(i*N/K)`，全部模型复用相同位置。查询只保留抽样部分，避免诊断额外加载全部查询字符串。每条抽样答案都检查，校验与训练不在查询阶段计时内。

输出：

- `static/report.md`：真正的吞吐和端到端延迟，与 ART 比较用这个。
- `stage-profile.md`：各阶段原始 ns/查询、占比、计时作用域次数/查询。
- `stage-profile.csv`：每轮 `all` / `hit` / `miss` 的全部阶段，包括完整 key 比较数、标签检查数与目录回退数。计数器只统计整份样本，不伪造 hit/miss 分项；没有命中或未命中的组不输出零成本记录。
- `block-*-stages.csv`：单个实现的原始诊断结果。

阶段定义：

| 阶段 | 范围 |
| --- | --- |
| `directory_route` | 多层目录短编码或目录模型、候选搜索和真实 fence 边界检查；不含精确回退 |
| `leaf_model` | 基础叶区域的前缀检查、编码、分段预测和候选窗口 |
| `tag_generate` | 生成查询的普通哈希 / 学习字节标签 |
| `tag_filter` | 基础标签数组的 SIMD / 标量筛选；不含完整 key 比较 |
| `key_verify` | 叶内完整 key 相等验证；E1 还包括模型窗口内的二分 |
| `delta_scan` | 局部增量区遍历及标签检查；完整 key 验证单列 |
| `exact_fallback` | 目录预测不可靠 / 未认证叶区域的精确有序搜索 |
| `other` | 剩余控制流、记录 ID / 结果访问及部分计时器记录开销 |

时间是互斥的，嵌套阶段暂停父阶段；各阶段之和等于**带插桩的**查询总时间。`scope_calls` 是计时作用域执行数，不是完整 key 比较数；E1 窗口二分和最终相等验证可能有嵌套同类作用域。`clock_pair_median_ns` 是两次空读时钟的间隔参考，不是全部记录器开销，不机械扣除；中位数为 0 可能是时钟粒度所致，不代表计时免费，另提供 P90 和零间隔比例。计时器/计数器和执行布局都会扰动结果，尤其是小数据几十纳秒级查询：不能用阶段总时间替代 QPS，也不能把原始阶段占比当成无插桩时的真实占比。需结合执行次数、数据规模和正式查询结果判断。

当前只细分静态查询，`--mode mixed --profile-stages` 会明确拒绝；`both` 会细分其中的静态部分。没有插入时 `delta_scan` 不执行是正常的。ART/HOT 源码不插桩，其内部阶段不提供，而不是写成 0。

### 大规模 ART 对比命令

先跑完整 DBLP（文件名标注约 750 万条；报告以实际排序去重后的数量为准）：

```sh
python3 -B src/experiments/run_experiments.py \
  --dataset data/dblp_7498293_maxlen128 --reader raw-lines \
  --output build/dblp-art-profile-v1 --mode static \
  --implementation block-plain --implementation block-hash \
  --implementation block-residual --implementation block-residual-delta \
  --implementation art \
  --operations 5000000 --repeats 5 --miss-ratio 0.1 \
  --latency-samples 10000 \
  --profile-stages --profile-samples 100000 \
  --memory-budget-mib 8192 --timeout 7200
```

然后跑完整 URL（文件名标注约 1972 万条），保持其余配置一致，将上述两个参数替换为：

```sh
--dataset data/url_19715903_maxlen128
--output build/url-art-profile-v1
```

当前电脑如果仍使用临时 CMake，额外加 `--cmake /private/tmp/blocklearn-build-tools.HyImkg/venv/bin/cmake`；临时目录可能被系统清理，长期使用应安装 CMake 并加入 PATH。

`--operations 5000000` 是每轮查询 500 万次，不是只加载 500 万条数据；上述命令都使用完整文件。`--memory-budget-mib 8192` 是数据/负载准备的逻辑预算，不是进程总内存或 OS 硬上限：Python 准备与 C++ runner 共存，还需训练、查询和原生 ART 内存。按可用 RAM 调整并留余量；发生换页时不要用结果做查询性能排名。输出目录必须不存在；预算不足会明确报错，不自动截断数据或偷换小样本。首次建议先跑 DBLP，再决定是否跑更大的 URL。命中率对比另开输出目录，分别设置 `--miss-ratio 0` / `0.5`。

## 固定的评估矩阵

对 dblp / imdb / reddit / url / wiki 分别准备：

1. 均匀 100% 命中：`--mode static --miss-ratio 0`。
2. 均匀 90% 命中 / 10% 未命中：`--mode static --miss-ratio 0.1`。
3. 均匀 50% 命中 / 50% 未命中：`--mode static --miss-ratio 0.5`。
4. 95% 读 / 5% 插入：`--mode mixed --insert-ratio 0.05 --initial-fraction 0.8`；另可测试 80/20。

这里“四个消融版本”和“四类负载”是两个维度。没有使用 Zipf 或热门前缀来制造收益。最初形成的索引使用随机 80% key，插入随机保留的真实 key，避免只测单调追加。

## 计时与产物

- 静态：直接调用队友未修改的 `scripts/benchmark.py`，所有实现复用规范数据和查询。结果在 `static/report.md`、JSON、CSV。
- 混合：复用其 `prepare_dataset` / `SIBROW1`，所有实现重用固定种子的相同操作流。完整正确性、吞吐、延迟使用不同新实例。
- 混合吞吐包含调用、响应结果写入、插入分配及同步合并/训练；不含数据读取、排序、负载生成和答案校验。
- 单独的延迟轮对所有插入测时，读每 `--latency-every` 个操作抽样，包含计时器开销；每轮分别报告 P50/P99，不混合跨轮百分位。
- 合并/分裂、查找路径中的完整 key 比较（包括目录边界比较）、标签检查和目录 fallback 在独立 instrumented 正确性轮统计，不给正式吞吐路径增加计数。比较次数不含训练/维护排序及扫描。
- 混合结果在 `mixed-summary.csv`；每个实现独立进程，保留原始 CSV。静态 E1/E2/E3 因不支持 insert，不参与混合性能排名。
- `suite.json` 记录源文件摘要、数据身份、架构、基线 commit、配置、可用性和限制。查看根目录 `report.md` 时先确认基线可用。
- 内存分项只是容量/逻辑分配估计，未提供每算法独立峰值 RSS，不能拿它当真实总内存排名。

队友当前静态 runner 会把 `bulk_load/find` 都 unsupported 的 descriptor 视作 invalid_input，而不是可跳过实现。因此本入口先运行真实工厂的可用性探测，把不可用基线记录在 `availability.csv` / `suite.json` / 根报告，再仅将可用实现交给静态 runner。未修改队友仓库，也没有谎报 HOT 支持。兼容 x86 GCC 环境下的 HOT manifest 可直接用于其 runner。

性能测试应自行绑核、避免 VM 过量分配和宿主机负载，并记录 CPU/缓存/编译器配置。当前入口没有自动预热或绑核。功能测试通过不意味着超过 ART/HOT，也不意味着论文新颖性成立。
