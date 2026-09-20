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
   HOT、ART 或局部搜索进行精确纠错
```

模型允许存在预测误差；索引的 fallback 必须保证不会漏查或误查。

## 第一阶段范围

第一版只实现并评估以下能力：

- 静态 bulk-load；
- 单线程精确点查；
- 短范围扫描；
- 字符串 key 的字典序；
- 模型误差、查询吞吐、内存和构建时间测量。

第一阶段暂不包含并发、在线重训练、删除和平台特定 SIMD 汇编优化。

## 模型方案

初始方案参考 LITS：用前缀条件概率近似字符串 CDF，再在节点内使用局部模型预测位置。

```text
prefix
  ↓
下一个字符的条件概率 P(c | prefix)
  ↓
字符串 CDF 近似
  ↓
子树内 rank 或 child slot 预测
```

对于难以拟合的局部字符串集合，允许回退到 HOT 或 ART 子树。

## 对比基线

- ART：自适应节点布局的字符串 Trie；
- HOT：按区分 bit 降低树高的字符串 Trie；
- B+ 树：比较式有序索引；
- 普通学习型索引：字符串数值化后的线性模型或 RMI；
- LITS：HPT + 局部模型 + HOT 子树的字符串学习型索引。

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
