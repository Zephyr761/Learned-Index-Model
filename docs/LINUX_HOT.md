# x86 Linux：HOT 接口验收与正式测试

## 1. 运行条件

- x86-64 Linux，使用 GCC / g++ 编译 C++17。
- CPU 必须向当前进程暴露 AVX2、BMI、BMI2、LZCNT、POPCNT。项目配置时会执行 CPU 特性检查，不满足即失败。
- 在 x86 宿主机上的 Linux 虚拟机可以先验收；记录宿主/客体配置，避免资源超分、换页和并行负载。ARM Linux 虚拟机不能替代 x86 原生环境。
- 不使用 Mac 上的 x86 仿真做原生性能结论。入口检查系统和进程架构，但不能识别所有转译/仿真环境，机器来源仍需人工确认。

Ubuntu/Debian 示例依赖（由用户在目标机器执行）：

```sh
sudo apt-get update
sudo apt-get install -y build-essential cmake python3 git
uname -m
g++ --version
lscpu
```

所有后续命令都在 `Learned-Index-Model` 根目录运行。不要复制 Mac 的 `build/`、CMake 缓存或临时 CMake 路径到 Linux；在目标机器重新编译。

## 2. 准备源码与数据

迁移必须带上当前模型、`src/benchmark`、`src/experiments`、`tests` 和 `tools` 的未提交更改。只 clone 原仓库旧 commit 不会带上本轮实现。数据集按原始字节复制，文件名不能代替内容校验。

本项目当前将以下目录记录为 Git gitlink，但没有根 `.gitmodules`；不能假设新 clone 会自动取得其中源码。先确认这些文件存在：

```text
related_works/hot/libs/hot/single-threaded/include/hot/singlethreaded/HOTSingleThreaded.hpp
related_works/libart/src/art.c
benchmark/learned-index/scripts/benchmark.py
```

仅在对应目录不存在/尚未取得源码时，按下面方式准备；已有 clone 不要覆盖，也不要在有改动时强行切换：

```sh
mkdir -p related_works benchmark

git clone --no-checkout https://github.com/speedskater/hot.git related_works/hot
git -C related_works/hot checkout --detach 96bf6fb7103b27e50e16a6026db8974c090ee84a

git clone --no-checkout https://github.com/armon/libart.git related_works/libart
git -C related_works/libart checkout --detach 301046804af165269e37da6725f5a4aec9ecc881

git clone https://github.com/zsh20070718/learned-index.git benchmark/learned-index
```

这两个基线版本与适配器 descriptor 中的版本一致。验收入口拒绝版本不一致或存在已跟踪源码改动的基线；不自动下载、覆盖或修改仓库。单线程 HOT 通过本项目直接包含头文件，不需要构建其顶层 benchmark、并发库或子模块依赖。

## 3. 一条命令验收

```sh
python3 -B tools/prepare_linux_hot.py --build build/linux-hot-ready-v1
```

`--build` 必须是不存在的新目录。可用 `--cc gcc-12 --cxx g++-12` 选择已安装的 GCC，`--cmake /absolute/path/to/cmake` 选择 CMake，`--benchmark /path/to/learned-index` 选择另一份队友仓库。编译器和所有比较均在同一环境运行。不要和正式性能测试并行执行。

入口依次执行：

1. 核对 Linux/x86 架构、工具、基线版本和源码完整性。
2. 用 Release、GCC 和 `BLOCKLEARN_REQUIRE_HOT=ON` 编译，缺少原生 HOT 时失败，不接受 unavailable 替身。
3. 运行 HOT 专项契约测试，以及 ART/HOT 共用契约测试。
4. 调用真实工厂的 `--describe`，确认模型、ART、HOT 均可用。
5. 生成 4099 条可复现测试键，含 253/254 字节与前缀键；通过原静态 benchmark 和混合 runner 跑同一最终模型、ART、HOT。
6. 检查三个静态结果全部 passed、三个混合结果全部 passed 且 checksum 相同，才输出验收通过。

验收测试包括：空树和单记录转移、空 key、零值与 uint64 最大值、unsigned-byte 顺序、高位字节、非 NUL 结尾的查询 `string_view`、254 字节边界、随机命中/未命中、新键所有权、重复插入不覆盖、有序 scan、闭区间求和及模 2^64 溢出。内嵌 NUL、255 字节键与非法 bulk 数据必须拒绝；update/erase 声明为 unsupported。

输出在指定目录：

- `readiness.json`：最终通过/失败状态和工具、源码身份。
- `configure.log`、`build.log`、`contracts.log`、`availability.log`：原生检查记录。
- `smoke/static/report.md`、`smoke/mixed-summary.csv`：小规模静态/读写接入结果。
- `suite-smoke.log`：统一 benchmark 执行日志，超时保留已有输出。

这些结果只证明环境与接口接入通过，不用于声称性能超过 ART/HOT。Mac 可以验证拒绝分支和脚本单元测试，不能代替原生 HOT 验收；未在 x86 Linux 实际执行前，不应写“HOT 原生测试已通过”。

## 4. 大规模静态比较

验收通过后，测试最终完整配置与两个基线：

```sh
CC=gcc CXX=g++ python3 -B src/experiments/run_experiments.py \
  --dataset data/dblp_7498293_maxlen128 --reader raw-lines \
  --output build/linux-dblp-hot-static-v1 --mode static --require-hot \
  --implementation block-residual-delta-shortdir-first \
  --implementation art --implementation hot \
  --operations 5000000 --repeats 5 --seed 42 --miss-ratio 0.1 \
  --memory-budget-mib 8192 --timeout 7200
```

`-shortdir-first` 是两项同时开启的配置，并非预先认定的最快配置；也可按 Mac 消融结果选择仅 `-shortdir`、仅 `-first` 或原模型。每次输出目录改名，不能覆盖旧报告。HOT 不会自动加入只选择 ART 的旧命令，必须明确选择 `--implementation hot --require-hot`。

若要在 Linux 重做三组消融，公共参数选择 `block-residual-delta`、`art`、`hot`，增加 `--require-hot`，继续使用 `--ablation directory / probe / both`；每组是完整模型对照、当前优化版、ART、HOT，不再额外加入哈希家族。

看 `static/report.md` 的正式吞吐与延迟。需要模型阶段诊断时另加 `--profile-stages --profile-samples 100000`；HOT/ART 源码不插桩，不提供伪造的内部阶段时间。

## 5. 读写比较

```sh
CC=gcc CXX=g++ python3 -B src/experiments/run_experiments.py \
  --dataset data/dblp_7498293_maxlen128 --reader raw-lines \
  --output build/linux-dblp-hot-mixed-v1 --mode mixed --require-hot \
  --implementation block-residual-delta-shortdir-first \
  --implementation art --implementation hot \
  --operations 5000000 --repeats 5 --seed 42 --miss-ratio 0.1 \
  --initial-fraction 0.8 --insert-ratio 0.05 \
  --memory-budget-mib 8192 --timeout 7200
```

`mixed-summary.csv` 提供读写吞吐和独立延迟轮的插入 P50/P99，模型同步合并/重训计入插入成本。不启用纯静态哈希家族。内存预算不是进程 RSS 硬上限，按机器 RAM 留余量；换页、仿真或负载竞争下的结果不作为原生性能排名。

## 6. HOT 接口语义与限制

实现位于 `src/benchmark/hot_baseline.cpp`，工厂 `make_hot_baseline`，manifest 位于 `src/experiments/hot`，注册 ID 为 `hot`，实际算法身份为 `hot-single-threaded`。

- bulk-load：按有序唯一输入逐键调用原生 insert，输入 key/value 由调用者保持存活且不可修改。
- find：把有长度的查询复制为有界 C 字符串，追加一个 EOS；完整固定长度补零仍由原生 HOT 完成，不在适配层重复全量补零。
- insert：新增字符串和记录引用存储在稳定 deque，重复键不覆盖原 value。
- scan：原生 lower_bound + 有序迭代，返回拥有字符串的行。
- range_sum：原生 lower_bound + 迭代，闭区间求和；查询上下界都先验证键域。
- 键域：无内嵌 NUL、至多 254 字节，允许空键、高位字节和前缀对；不截断、不编码改写、不按基线限制偷偷过滤数据。
- 内存：原生存活树节点统计、适配层与保留输入分项；不包含原生空闲池/分配器和临时峰值，不是 RSS。

当前项目的 128 字节上限文本数据仍应以报告实际键域为准；数据/负载含 NUL 或超长键时，`--require-hot` 应拒绝“不完整比较”，不能把 HOT unsupported 当作零吞吐。
