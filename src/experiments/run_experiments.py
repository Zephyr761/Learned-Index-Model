#!/usr/bin/env python3
"""Four matched ablations + real ART/HOT, using the teammate's v1 contract.

Static: delegate unchanged to the teammate's runner.
Mixed: reuse its canonical dataset reader/interchange, then run our companion.
No downloads, source rewrites, automatic sampling or fabricated baseline scores.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from stage_profile import profile_static

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = Path(__file__).resolve().parent
DEFAULT_IDS = ("block-plain", "block-hash", "block-residual", "block-residual-delta", "art", "hot")
ABLATION_SUFFIXES = {"directory": "-shortdir", "probe": "-first", "both": "-shortdir-first"}
ABLATION_FAMILIES = ("block-hash", "block-residual-delta")
ABLATION_IDS = tuple(name + suffix for name in ABLATION_FAMILIES for suffix in ABLATION_SUFFIXES.values())
PLANNER_IDS = ("block-residual-delta-shortdir-first-cost", "block-adaptive-delta-shortdir-first")
IDS = DEFAULT_IDS + ABLATION_IDS + PLANNER_IDS


def expand_ablations(implementations, ablation):
    result = []
    for name in implementations:
        result.append(name)
        if ablation and name in ABLATION_FAMILIES:
            result.append(name + ABLATION_SUFFIXES[ablation])
    return result


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def fraction(text):
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def run(command, env, timeout, log=None):
    result = subprocess.run([str(x) for x in command], env=env, timeout=timeout,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log:
        log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("command failed ({}): {}\n{}".format(result.returncode, command[0], result.stdout[-4000:]))
    return result.stdout


def source_identity():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*")):
        if path.is_file() and path.suffix in (".h", ".cpp", ".py", ".json", ".txt"):
            if "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def git_revision(directory):
    try:
        return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reader", choices=("auto", "raw-lines", "jsonl", "legacy-tli"), default="auto")
    parser.add_argument("--key")
    parser.add_argument("--output", type=Path, required=True, help="must not already exist")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "benchmark/learned-index")
    parser.add_argument("--mode", choices=("static", "mixed", "both"), default="static")
    parser.add_argument("--implementation", action="append", choices=IDS)
    parser.add_argument("--ablation", choices=tuple(ABLATION_SUFFIXES),
                        help="pair selected block-hash / block-residual-delta controls with directory-only, probe-only, or combined variants")
    parser.add_argument("--operations", type=positive, default=1000000)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--miss-ratio", type=fraction, default=.1)
    parser.add_argument("--insert-ratio", type=fraction, default=.05)
    parser.add_argument("--initial-fraction", type=fraction, default=.8)
    parser.add_argument("--latency-samples", type=positive, default=10000)
    parser.add_argument("--latency-every", type=positive, default=1024)
    parser.add_argument("--profile-stages", action="store_true", help="separate diagnostic query pass, never used for QPS; requires static/both mode")
    parser.add_argument("--profile-samples", type=positive, default=10000)
    parser.add_argument("--memory-budget-mib", type=positive, default=1024)
    parser.add_argument("--timeout", type=positive, default=1800)
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--cmake", type=Path, help="optional absolute path to cmake")
    parser.add_argument("--planner-costs", type=Path, help="reuse an independent synthetic cost profile; otherwise new planner variants calibrate before dataset/query loading")
    parser.add_argument("--require-hot", action="store_true", help="fail instead of claiming a complete ART/HOT comparison without native HOT")
    args = parser.parse_args()
    if args.seed < 0 or args.seed > (1 << 64) - 1 or args.initial_fraction == 0:
        parser.error("seed must fit uint64; initial-fraction must be >0")
    if args.profile_stages and args.mode == "mixed":
        parser.error("--profile-stages currently profiles static queries; use --mode static or both")
    benchmark = args.benchmark.resolve()
    if not (benchmark / "scripts/benchmark.py").is_file():
        parser.error("--benchmark is not the teammate's repository")
    implementations = args.implementation or list(DEFAULT_IDS)
    if len(set(implementations)) != len(implementations):
        parser.error("duplicate implementation")
    if args.ablation:
        if any(name in ABLATION_IDS + PLANNER_IDS for name in implementations):
            parser.error("--ablation selects variants automatically; select base implementation IDs")
        if not any(name in ABLATION_FAMILIES for name in implementations):
            parser.error("--ablation requires selecting block-hash or block-residual-delta")
        implementations = expand_ablations(implementations, args.ablation)
    if args.require_hot and "hot" not in implementations:
        parser.error("--require-hot requires selecting hot")
    needs_planner = any(name in PLANNER_IDS for name in implementations)
    if args.planner_costs and (not needs_planner or not args.planner_costs.is_file()):
        parser.error("--planner-costs requires a readable file and a cost/adaptive implementation")
    env = os.environ.copy()
    if args.cmake:
        cmake = args.cmake.resolve()
        if not cmake.is_file():
            parser.error("cmake path does not exist")
        env["PATH"] = str(cmake.parent) + os.pathsep + env.get("PATH", "")
    else:
        cmake = shutil.which("cmake")
    if not cmake:
        parser.error("CMake is required; provide --cmake /absolute/path/to/cmake")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema": "blocklearn-experiment-suite/v1", "mode": args.mode,
        "source_sha256": source_identity(), "platform": platform.platform(),
        "architecture": platform.machine(), "python": sys.version,
        "art_revision": git_revision(ROOT / "related_works/libart"),
        "hot_revision": git_revision(ROOT / "related_works/hot"),
        "implementations": implementations, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "optimization_variants": {name: {"short_directory": "-shortdir" in name,
            "predicted_first": "-first" in name,
            "planner": "calibrated-work" if name in PLANNER_IDS else "legacy",
            "adaptive_tags_and_order": name.startswith("block-adaptive-")} for name in implementations if name.startswith("block-")},
        "limitations": ["No automatic CPU pinning or prewarm; control the test environment.",
                        "Reported owned memory is an estimate, not peak RSS.",
                        "E3 and E4 have identical static lookup paths before inserts.",
                        "Planning is shared by E1/E2/E3/E4 for controlled ablations.",
                        "No speedup or novelty is inferred from functional smoke tests."],
    }
    (output / "suite.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        build = output / "suite-build"
        configure = [cmake, "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF"]
        if args.require_hot:
            configure.append("-DBLOCKLEARN_REQUIRE_HOT=ON")
        run(configure, env, args.timeout, output / "configure.log")
        targets = ["blocklearn_mixed"] + (["blocklearn_profile"] if args.profile_stages else [])
        if needs_planner:
            targets.append("blocklearn_calibrate")
        run([cmake, "--build", build, "--target", *targets, "-j", args.jobs],
            env, args.timeout, output / "build.log")
        if needs_planner:
            profile = output / "planner-costs.txt"
            calibrator = build / "src/blocklearn_calibrate"
            if args.planner_costs:
                shutil.copyfile(args.planner_costs.resolve(), profile)
                run([calibrator, "--validate", profile], env, args.timeout, output / "calibration.log")
            else:
                run([calibrator, "--output", profile], env, args.timeout, output / "calibration.log")
            env["BLOCKLEARN_COST_PROFILE"] = str(profile)
            metadata["planner_cost_profile"] = {"path": str(profile), "sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
                "origin": "independent-synthetic-v1", "policy": "frozen before dataset/query loading; never tuned to formal workload; fixed 90/10 planning prior"}
        description = run([build / "src/blocklearn_mixed", "--describe"], env, args.timeout,
                          output / "availability.csv")
        availability = {row["implementation"]: row for row in csv.DictReader(io.StringIO(description))}
        metadata["availability"] = availability
        static_implementations = [name for name in implementations if availability[name]["status"] == "available"]
        metadata["static_unsupported"] = [availability[name] for name in implementations if name not in static_implementations]
        if args.mode in ("static", "both"):
            workload = output / "static-workload.json"
            workload.write_text(json.dumps({"schema": "static-uniform-point/v1", "seed": args.seed,
                "operations": args.operations, "miss_ratio": str(args.miss_ratio),
                "repeats": args.repeats, "latency_samples": args.latency_samples}), encoding="utf-8")
            command = [sys.executable, benchmark / "scripts/benchmark.py", "--dataset", args.dataset.resolve(),
                       "--reader", args.reader, "--output", output / "static", "--workload", workload,
                       "--memory-budget-mib", args.memory_budget_mib, "--jobs", args.jobs,
                       "--build-timeout", args.timeout, "--run-timeout", args.timeout]
            if args.key:
                command.extend(["--key", args.key])
            for implementation in static_implementations:
                command.extend(["--implementation", MANIFESTS / implementation])
            if static_implementations:
                run(command, env, args.timeout * (len(implementations) + 2), output / "static.log")
                static_report = json.loads((output / "static/report.json").read_text(encoding="utf-8"))
                metadata["static_statuses"] = [{"id": r["implementation_id"], "status": r["status"]}
                                               for r in static_report["results"]]
                if args.require_hot and any(r["status"] != "passed" for r in static_report["results"]):
                    raise RuntimeError("required baseline comparison did not pass on the full workload")
                if args.profile_stages:
                    profile_static(args, output, build, env, metadata, static_report, run)
        if args.mode in ("mixed", "both"):
            sys.path.insert(0, str(benchmark / "scripts"))
            from benchmark_lib.dataset import prepare_dataset
            from benchmark_lib.interchange import write_rows
            prepared = prepare_dataset(args.dataset.resolve(), args.reader, args.key, args.memory_budget_mib * 1024 * 1024)
            metadata["mixed_dataset"] = prepared.metadata()
            rows = output / "mixed-rows.bin"
            metadata["mixed_rows_sha256"] = write_rows(rows, prepared.keys)
            del prepared
            results = []
            for implementation in implementations:
                command = [build / "src/blocklearn_mixed", "--rows", rows, "--implementation", implementation,
                           "--operations", args.operations, "--repeats", args.repeats, "--seed", args.seed,
                           "--miss-ratio", args.miss_ratio, "--insert-ratio", args.insert_ratio,
                           "--initial-fraction", args.initial_fraction, "--latency-every", args.latency_every]
                text = run(command, env, args.timeout, output / (implementation + "-mixed.csv"))
                results.extend(csv.DictReader(io.StringIO(text)))
            metadata["mixed_results"] = results
            if args.require_hot and any(r["status"] != "passed" for r in results if r["implementation"] in ("art", "hot")):
                raise RuntimeError("required mixed baseline comparison did not pass")
            metadata["mixed_timing"] = "Fresh correctness, throughput, and latency instances; inserts including synchronous merge/retraining timed. All inserts latency-sampled; reads sampled periodically. Response materialization included in throughput. No prewarm."
            with (output / "mixed-summary.csv").open("x", newline="", encoding="utf-8") as stream:
                if results:
                    writer = csv.DictWriter(stream, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
        if source_identity() != metadata["source_sha256"]:
            raise RuntimeError("model/adapter/experiment source changed during the run; results are not a frozen comparison")
        if needs_planner and hashlib.sha256((output / "planner-costs.txt").read_bytes()).hexdigest() != metadata["planner_cost_profile"]["sha256"]:
            raise RuntimeError("planner cost profile changed during the run")
        partial = bool(metadata["static_unsupported"]) or any(r["status"] == "unsupported" for r in metadata.get("static_statuses", []))
        partial = partial or any(r["status"] == "unsupported" for r in metadata.get("mixed_results", []))
        metadata["status"] = "completed_with_unsupported" if partial else "completed"
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        raise
    finally:
        (output / "suite.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        lines = ["# BlockLearn experiment suite", "", "Status: " + metadata.get("status", "unknown"), "",
                 "Availability (unsupported entries have no invented performance score):", ""]
        for name in implementations:
            item = metadata.get("availability", {}).get(name, {"status": "unknown", "notes": "preflight not completed"})
            lines.append("- {}: {} — {}".format(name, item["status"], item["notes"]))
        lines += ["", "Static results: static/report.md (only available implementations).",
                  "Mixed results: mixed-summary.csv. Full provenance: suite.json.", "",
                  "E3/E4 static paths are identical; this is not proof of speedup or novelty."]
        if "stage_profile" in metadata:
            lines += ["", "Separate diagnostic stages: stage-profile.md / stage-profile.csv (instrumented times, NOT throughput)."]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Results:", output)


if __name__ == "__main__":
    main()
