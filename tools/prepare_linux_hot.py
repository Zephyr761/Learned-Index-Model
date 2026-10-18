#!/usr/bin/env python3
"""Require real x86 Linux HOT, test its adapter, then smoke-test the shared suite.

Does not download dependencies, change src, or accept an unavailable HOT stub.
Do not run alongside a performance measurement on the same host.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "hot": "96bf6fb7103b27e50e16a6026db8974c090ee84a",
    "libart": "301046804af165269e37da6725f5a4aec9ecc881",
}


def check_host(system, machine):
    if system != "Linux" or machine.lower() not in ("x86_64", "amd64"):
        raise RuntimeError("requires native x86-64 Linux; ARM Linux/macOS are not accepted. "
                           "Do not use x86 emulation for the final performance comparison.")


def check_sources(root, benchmark):
    for name, expected in PINS.items():
        path = root / "related_works" / name
        header = (path / "libs/hot/single-threaded/include/hot/singlethreaded/HOTSingleThreaded.hpp"
                  if name == "hot" else path / "src/art.c")
        if not header.is_file():
            raise RuntimeError("missing {} source: {}; see docs/LINUX_HOT.md".format(name, path))
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise RuntimeError("{} revision is {}, expected {} (adapter identity must be truthful)".format(name, actual, expected))
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True)
        if dirty.strip():
            raise RuntimeError("{} has tracked source changes; native baseline must remain unmodified".format(name))
    if not (benchmark / "scripts/benchmark.py").is_file():
        raise RuntimeError("missing teammate benchmark clone: {}".format(benchmark))


def executable(name):
    found = shutil.which(str(name))
    if not found:
        raise RuntimeError("required tool unavailable: {}".format(name))
    return str(Path(found).resolve())


def require_hot(description):
    reader = csv.DictReader(io.StringIO(description))
    if reader.fieldnames != ["implementation", "status", "notes"]:
        raise RuntimeError("invalid availability CSV header")
    rows = {}
    for row in reader:
        if None in row or any(value is None for value in row.values()) or row["implementation"] in rows:
            raise RuntimeError("invalid/duplicate availability CSV row")
        rows[row["implementation"]] = row
    for name in ("hot", "art", "block-residual-delta-shortdir-first"):
        if name not in rows or rows[name]["status"] != "available":
            raise RuntimeError("{} is unavailable: refusing a substitute/incomplete smoke test".format(name))


def require_suite(suite):
    expected = {"block-residual-delta-shortdir-first", "art", "hot"}
    if suite.get("status") != "completed":
        raise RuntimeError("smoke suite is not completely passed")
    static = suite.get("static_statuses", [])
    if len(static) != len(expected) or {r.get("id") for r in static} != expected or \
            any(r.get("status") != "passed" for r in static):
        raise RuntimeError("native static workload did not pass for all three implementations")
    mixed = suite.get("mixed_results", [])
    if len(mixed) != len(expected) or {r.get("implementation") for r in mixed} != expected or \
            any(r.get("status") != "passed" for r in mixed) or \
            any(not str(r.get("checksum", "")).isdigit() for r in mixed) or \
            len({r["checksum"] for r in mixed}) != 1:
        raise RuntimeError("native mixed workload did not pass with identical checksums")


def write_fixture(path):
    keys = ["https://example.org/shared-prefix/{:08x}".format(i) for i in range(4096)]
    keys += ["p" * 253, "p" * 254, "p" * 253 + "q"]
    path.write_text("\n".join(keys) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/linux-hot-ready-v1", help="new, non-existing directory")
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "benchmark/learned-index")
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("timeout must be positive")
    build = args.build.resolve()
    metadata = None
    try:
        check_host(platform.system(), platform.machine())
        check_sources(ROOT, args.benchmark.resolve())
        cmake, cc, cxx = executable(args.cmake), executable(args.cc), executable(args.cxx)
        ctest = executable(str(Path(cmake).with_name("ctest")))
        build.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        env.update(CC=cc, CXX=cxx)
        env["PATH"] = str(Path(cmake).parent) + os.pathsep + env.get("PATH", "")
        metadata = {"status": "running", "purpose": "native correctness/integration acceptance, not a performance conclusion",
                    "host": platform.platform(), "architecture": platform.machine(), "revisions": PINS,
                    "cmake": cmake, "cc": cc, "cxx": cxx}

        def run(label, command):
            print(label + " ...", flush=True)
            try:
                result = subprocess.run([str(x) for x in command], env=env, timeout=args.timeout,
                                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            except subprocess.TimeoutExpired as error:
                partial = error.stdout or ""
                if isinstance(partial, bytes):
                    partial = partial.decode("utf-8", errors="replace")
                (build / (label + ".log")).write_text(partial, encoding="utf-8")
                raise RuntimeError("{} timed out; partial log: {}".format(label, build / (label + ".log"))) from error
            (build / (label + ".log")).write_text(result.stdout, encoding="utf-8")
            if result.returncode:
                raise RuntimeError("{} failed: {}\n{}".format(label, build / (label + ".log"), result.stdout[-5000:]))
            return result.stdout

        native = build / "native"
        run("configure", [cmake, "-S", ROOT / "tests/linux_hot", "-B", native,
            "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=" + cc, "-DCMAKE_CXX_COMPILER=" + cxx,
            "-DBLOCKLEARN_SANITIZERS=OFF", "-DBUILD_TESTING=ON"])
        run("build", [cmake, "--build", native, "--target", "hot_native_contract_test",
            "baseline_contract_test", "blocklearn_mixed", "-j", args.jobs])
        run("contracts", [ctest, "--test-dir", native, "--output-on-failure"])
        require_hot(run("availability", [native / "adapters/blocklearn_mixed", "--describe"]))
        fixture = build / "keys.txt"
        write_fixture(fixture)
        run("suite-smoke", [sys.executable, "-B", ROOT / "src/experiments/run_experiments.py",
            "--cmake", cmake, "--benchmark", args.benchmark.resolve(), "--dataset", fixture,
            "--reader", "raw-lines", "--output", build / "smoke", "--mode", "both", "--require-hot",
            "--implementation", "block-residual-delta-shortdir-first", "--implementation", "art",
            "--implementation", "hot", "--operations", "10000", "--repeats", "1", "--seed", "42",
            "--initial-fraction", "0.75", "--insert-ratio", "0.05", "--miss-ratio", "0.1",
            "--jobs", args.jobs, "--timeout", args.timeout])
        suite = json.loads((build / "smoke/suite.json").read_text(encoding="utf-8"))
        require_suite(suite)
        metadata["status"] = "passed"
        print("Native HOT adapter and static/mixed integration passed. Results: " + str(build))
        return 0
    except (RuntimeError, OSError, subprocess.SubprocessError, ValueError) as error:
        if metadata is not None:
            metadata.update(status="failed", error=str(error))
        print("HOT acceptance failed: " + str(error), file=sys.stderr)
        return 2
    finally:
        if metadata is not None:
            (build / "readiness.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
