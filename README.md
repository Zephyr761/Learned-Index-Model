# Learned-Index  面向内存字符串键的学习型索引

## 项目目标

构建一个运行于内存中的字符串键值索引。索引需要支持精确点查、范围扫描和后续的动态更新，并在真实与可控合成字符串数据上，与 ART、HOT、B+ 树及普通学习型索引进行对比。

本项目的核心问题是：对一个字符串 key，如何高效预测它在当前子树有序 key 集合中的累计分布函数（CDF）或 rank，同时保持查询结果完全正确。

## 当前负责内容：字符串拟合模型

模型不直接决定查询结果，而是为索引提供导航预测：

```text
当前子树的 key 集合 + lookup string
                ↓
       预测局部 CDF / rank / child slot
                ↓
   真实边界检查、标签过滤与局部精确验证
```

模型允许存在预测误差；索引的 fallback 必须保证不会漏查或误查。

## 第一阶段范围

第一版只实现并评估以下能力：

- 静态 bulk-load；
- 单线程精确点查；
- 短范围扫描；
- 字符串 key 的字典序；
- 模型误差、查询吞吐、内存和构建时间测量。

当前 BlockLearn 已支持单线程局部增量合并、重训练和 SIMD 标签筛选；并发、删除和持久化不在本轮实现范围。

## 当前可运行模型

原来的三个 adapter 已移除。新的 BlockLearn 实现在 `src/learned_index.*`：多层范围模型目录、区域内块位置预测、残差感知字节标签、精确字符串验证，以及局部 delta 的同步合并重训和分裂。不依赖热门 key 缓存。

四个消融版本和真实 ART/HOT 的 v1 工厂、manifest、静态/混合评测入口均在 `src`。构建和运行见 [实验说明](src/experiments/README.md)，完整机制与正确性论证见 [设计复核](src/DESIGN.md)。尚未证明在真实数据上超过 ART/HOT。

迁移到 x86 Linux 后，先运行 `python3 -B tools/prepare_linux_hot.py` 强制验收原生 HOT，再做正式比较。环境准备、源码版本和静态/读写命令见 [Linux HOT 接口验收](docs/LINUX_HOT.md)。

新增实际批次/提前返回成本规划和区域自适应标签版本，旧模型保留为对照；机制、校准约束、服务器更新清单与六版本测试命令见 [规划器 v3](docs/PLANNER_V3.md)。

## 模型方案

当前方案直接训练局部 `key 编码 -> 块编号`，并根据预测候选集合学习用于验证的字节位置；不使用全局 HPT 来假定某个局部后缀的真实条件 CDF。

```text
局部区域中的有序 key
  ↓
去掉共同前缀后的保序编码
  ↓
分段线性模型预测候选块
  ↓
残差感知标签过滤 -> 完整 key 精确验证
```

困难区域拆分，模型预算不满足时精确搜索兜底。ART/HOT 是独立评测基线，不是新模型内部偷偷调用的查询实现。

## 对比基线

- ART：自适应节点布局的字符串 Trie；
- HOT：按区分 bit 降低树高的字符串 Trie；
- B+ 树：比较式有序索引；
- 普通学习型索引：字符串数值化后的线性模型或 RMI；
- LITS：HPT、模型节点、紧凑叶节点与混合结构选择的字符串学习型索引。

## 评估

### 数据集

- 真实数据：URL、email、词典、用户 ID 等；
- 合成数据：可控制 key 长度、公共前缀长度、前缀偏斜度和 key 数量。

### Workload

- 100% point lookup；
- 95% lookup + 5% insert；
- 短范围 scan；
- 均匀访问与 Zipf 热点访问。

### 指标

- 点查吞吐、平均延迟与 p99 延迟；
- range scan 吞吐；
- rank/CDF 预测误差和纠错窗口大小；
- 索引内存占用；
- bulk-load / 模型构建时间；
- 更新或重建成本（后续阶段）。

## 设计约定

- key 按字节字典序比较，并明确处理空字节、变长 key 与重复 key；
- 模型接口应返回预测位置、候选范围或 child slot，而非直接返回值；
- 所有模型预测必须经过精确校验；
- 先提供可移植的标量实现，再在 x86_64 Linux 环境上补充 AVX2、BMI2/PEXT 等优化；
- Apple Silicon Mac 可用于功能开发，但不应用于 HOT 的 AVX2/BMI2 最终性能结论。

## 待确认事项

- benchmark、索引主体、并发模块分别由谁负责；
- 模型与索引节点之间的最终接口；
- 目标 x86_64 Linux 测试机器与可用 CPU 指令集；
- 首批真实数据集、数据规模与性能目标；
- 动态更新采用局部重训、批量重建还是 delta index。

## 参考论文

- Kraska et al. *The Case for Learned Index Structures*. SIGMOD 2018.
- Leis et al. *The Adaptive Radix Tree*. ICDE 2013.
- Binna et al. *HOT: A Height Optimized Trie Index*. SIGMOD 2018.
- Ferragina and Vinciguerra. *The PGM-index*. PVLDB 2020.
- Yang and Chen. *LITS: An Optimized Learned Index for Strings*. PVLDB 2024.
