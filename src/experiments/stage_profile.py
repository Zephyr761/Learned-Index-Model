"""Diagnostic report only; never add instrumented nanoseconds to QPS results."""
import csv
import hashlib
import io

STAGES = {
    "directory_route": "目录路由（短编码或目录模型/边界验证，不含精确回退）",
    "leaf_model": "叶模型（公共前缀检查、编码、预测和窗口计算）",
    "tag_generate": "查询标签生成",
    "tag_filter": "基础区 SIMD/标量标签筛选",
    "key_verify": "叶内完整 key 验证",
    "delta_scan": "增量区遍历（不含完整 key 验证）",
    "exact_fallback": "目录 / 未认证叶区域的精确二分回退",
    "other": "其余控制流、ID/结果访问及部分记录器开销",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_static(args, output, build, env, metadata, static_report, run):
    rows = output / "static/interchange/rows.bin"
    queries = output / "static/interchange/queries.bin"
    frozen = {"rows_sha256": file_sha256(rows), "queries_sha256": file_sha256(queries)}
    if (frozen["rows_sha256"] != static_report["dataset"]["rows_sha256"] or
            frozen["queries_sha256"] != static_report["workload"]["queries_sha256"]):
        raise RuntimeError("profiling interchange differs from the static benchmark")
    results = []
    skipped = []
    passed = {r["implementation_id"] for r in static_report["results"] if r["status"] == "passed"}
    for name in metadata["implementations"]:
        if not name.startswith("block-"):
            skipped.append({"implementation": name, "reason": "baseline internals are not instrumented; use static/report.md for its real QPS"})
            continue
        if name not in passed:
            skipped.append({"implementation": name, "reason": "static correctness did not pass"})
            continue
        command = [build / "src/blocklearn_profile", "--rows", rows, "--queries", queries,
                   "--implementation", name, "--samples", args.profile_samples, "--repeats", args.repeats]
        text = run(command, env, args.timeout, output / (name + "-stages.csv"))
        results.extend(csv.DictReader(io.StringIO(text)))
    with (output / "stage-profile.csv").open("x", newline="", encoding="utf-8") as stream:
        if results:
            writer = csv.DictWriter(stream, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    metadata["stage_profile"] = {
        **frozen, "samples_requested": args.profile_samples, "repeats": args.repeats,
        "sampling": "floor(i * source_operations / sample_count), identical frozen positions for all models",
        "scope": "static point lookup only; one fresh bulk-loaded process per implementation, immutable model reused across diagnostic repeats",
        "timing": "exclusive raw steady_clock stages; includes instrumentation and counter overhead; no correction/subtraction; separate from throughput",
        "skipped": skipped,
    }
    lines = ["# 查询分阶段诊断（不是正式吞吐量）", "",
             "复用静态评测的同一 rows.bin / queries.bin，按固定等距位置抽样；命中 / 未命中分别保留在 CSV。",
             "每个实现独立进程，构建一次，重复诊断相同查询；不做显式预热。答案逐条验证，读取、训练、校验不计入查询阶段。", "",
             "**以下是带计时器与计数器开销的原始时间。不能将其相加后替代正式 ns/op、QPS 或 ART 延迟，也不能认为阶段占比就是无插桩程序的真实占比。**",
             "时间独占、嵌套不重复计时；不扣除时钟开销。clock_pair_median_ns 是空的两次读钟间隔参考，不是完整记录器开销。", "",
             "时钟粒度可能使空读钟中位数为 0，这不代表读时钟没有开销。CSV 同时提供 P90 间隔和零间隔比例。", "",
             "每阶段结果按全部重复的累计时间 / 累计抽样查询数汇总；all/hit/miss 的每轮原始值见 stage-profile.csv。", "",
             "| 实现 | 阶段 | 原始 ns/查询 | 原始总时间占比 | 计时作用域次数/查询 |", "| --- | --- | ---: | ---: | ---: |"]
    clock_notes = []
    region_notes = []
    for name in metadata["implementations"]:
        group = [r for r in results if r["implementation"] == name and r["outcome"] == "all"]
        totals = [r for r in group if r["stage"] == "profiled_total"]
        count = sum(int(r["sampled_queries"]) for r in totals)
        total_ns = sum(float(r["total_ns"]) for r in totals)
        if not count:
            continue
        lines.append("| {} | 带插桩总时间 | {:.2f} | {} | 1 |".format(name, total_ns / count, "100%" if total_ns else "不可用"))
        for stage, title in STAGES.items():
            entries = [r for r in group if r["stage"] == stage]
            ns = sum(float(r["total_ns"]) for r in entries)
            calls = sum(int(r["scope_calls"]) for r in entries)
            share = "{:.1f}%".format(100 * ns / total_ns) if total_ns else "不可用"
            lines.append("| {} | {} | {:.2f} | {} | {:.3f} |".format(name, title, ns / count, share, calls / count))
        clock = totals[0]
        if clock.get("hash_leaves") is not None:
            region_notes.append("{}：已认证区域 hash={}，residual={}，预测块优先={}；实际标签批次数/查询={:.3f}。".format(
                name, clock["hash_leaves"], clock["residual_leaves"], clock["predicted_first_leaves"],
                sum(int(r["tag_batches"]) for r in totals) / count))
        clock_notes.append("{}：每轮抽样 {} 条；空读钟间隔中位数 {:.2f} ns，P90 {:.2f} ns，零间隔 {:.1f}%。".format(
            name, clock["sampled_queries"], float(clock["clock_pair_median_ns"]),
            float(clock["clock_pair_p90_ns"]), 100 * float(clock["clock_pair_zero_fraction"])))
    lines += ["", *region_notes, "", *clock_notes, "", "未插入时 delta_scan 不执行，这是预期结果；混合读写中的合并/重训并未在此细分。",
              "ART/HOT 未改动源码，不伪造其内部阶段；与基线比较请查看 static/report.md。", ""]
    for item in skipped:
        lines.append("- {}: {}".format(item["implementation"], item["reason"]))
    (output / "stage-profile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
