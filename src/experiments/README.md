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
