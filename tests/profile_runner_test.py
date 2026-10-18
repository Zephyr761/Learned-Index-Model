"""Frozen workload sampling, exclusive accounting, and CLI failure checks."""
import csv
import io
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/experiments"))
from stage_profile import file_sha256, profile_static
from run_experiments import MODEL_IDS, CONTROL_ID


def main():
    binary = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="blocklearn-profile-") as folder:
        folder = Path(folder)
        costs = folder / "costs.txt"
        subprocess.run([sys.argv[2], "--output", str(costs), "--iterations", "1024"], check=True, stdout=subprocess.PIPE)
        environment = os.environ.copy()
        environment["BLOCKLEARN_COST_PROFILE"] = str(costs)
        rows = folder / "rows.bin"
        queries = folder / "queries.bin"
        keys = [b"", b"a", b"a\0b", b"https://example/001", b"https://example/002", b"\xff"]
        values = [0, 1, (1 << 64) - 1, 3, 4, 5]
        rows.write_bytes(b"SIBROW1\0" + struct.pack("<Q", len(keys)) + b"".join(
            struct.pack("<Q", len(k)) + k + struct.pack("<Q", v) for k, v in zip(keys, values)))
        workload = [(keys[i % len(keys)], True, values[i % len(keys)]) if i % 3 else
                    (b"missing-" + str(i).encode(), False, 0) for i in range(31)]

        def write_queries(items):
            queries.write_bytes(b"SIBQRY1\0" + struct.pack("<Q", len(items)) + b"".join(
                struct.pack("<BQQ", int(p), v, len(k)) + k for k, p, v in items))

        def invoke(name, samples=13, ok=True):
            result = subprocess.run([binary, "--rows", str(rows), "--queries", str(queries),
                "--implementation", name, "--samples", str(samples), "--repeats", "2"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
            assert (result.returncode == 0) == ok, result.stdout + result.stderr
            return list(csv.DictReader(io.StringIO(result.stdout))) if ok else []

        write_queries(workload)
        positions = [i * len(workload) // 13 for i in range(13)]
        checksum = sum(workload[i][2] + workload[i][1] for i in positions) % (1 << 64)
        for name in MODEL_IDS:
            result = invoke(name)
            assert all(None not in row and all(v is not None for v in row.values()) for row in result)
            for repeat in ("0", "1"):
                all_rows = [r for r in result if r["repeat"] == repeat and r["outcome"] == "all"]
                total = next(r for r in all_rows if r["stage"] == "profiled_total")
                assert int(total["sampled_queries"]) == 13
                assert int(total["source_operations"]) == len(workload)
                assert int(total["checksum"]) == checksum
                stages = [r for r in all_rows if r["stage"] != "profiled_total"]
                assert len(stages) == 8
                assert abs(sum(float(r["total_ns"]) for r in stages) - float(total["total_ns"])) < .01
                assert all(float(r["total_ns"]) >= 0 for r in stages)
                assert float(total["clock_pair_p90_ns"]) >= float(total["clock_pair_median_ns"]) >= 0
                assert 0 <= float(total["clock_pair_zero_fraction"]) <= 1
                assert next(r for r in stages if r["stage"] == "delta_scan")["scope_calls"] == "0"
                assert all(r["full_key_comparisons"] == "" for r in result if r["outcome"] != "all")
                assert all(int(r["checksum"]) == 0 for r in result if r["outcome"] == "miss")
        result = invoke(CONTROL_ID, 1000)
        assert all(int(r["sampled_queries"]) == len(workload) for r in result if r["outcome"] == "all")
        write_queries([(b"a", True, 1)] * 3)
        assert {r["outcome"] for r in invoke(CONTROL_ID)} == {"all", "hit"}
        write_queries([(b"missing", False, 0)] * 3)
        assert {r["outcome"] for r in invoke(CONTROL_ID)} == {"all", "miss"}
        # Exercise the actual Python integration/report writer with real C++
        # output, without rebuilding a suite inside every unit test.
        write_queries(workload)
        interchange = folder / "static/interchange"
        interchange.mkdir(parents=True)
        shutil.copyfile(rows, interchange / "rows.bin")
        shutil.copyfile(queries, interchange / "queries.bin")
        metadata = {"implementations": list(MODEL_IDS) + ["art"]}
        report = {"dataset": {"rows_sha256": file_sha256(rows)},
                  "workload": {"queries_sha256": file_sha256(queries)},
                  "results": [{"implementation_id": name, "status": "passed"} for name in metadata["implementations"]]}

        def run(command, env, timeout, log):
            result = subprocess.run([str(x) for x in command], env=env, timeout=timeout,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            log.write_text(result.stdout, encoding="utf-8")
            return result.stdout

        args = SimpleNamespace(timeout=30, profile_samples=13, repeats=2)
        profile_static(args, folder, Path(binary).resolve().parents[1], environment, metadata, report, run)
        combined = list(csv.DictReader(io.StringIO((folder / "stage-profile.csv").read_text())))
        assert {r["implementation"] for r in combined} == set(MODEL_IDS)
        assert metadata["stage_profile"]["queries_sha256"] == file_sha256(queries)
        assert metadata["stage_profile"]["skipped"][0]["implementation"] == "art"
        assert "不是正式吞吐量" in (folder / "stage-profile.md").read_text()
        report["workload"]["queries_sha256"] = "incorrect"
        try:
            profile_static(args, folder, Path(binary).resolve().parents[1], os.environ.copy(), {}, report, run)
            raise AssertionError("mismatched frozen workload accepted")
        except RuntimeError:
            pass
        rejected = folder / "must-not-be-created"
        mode = subprocess.run([sys.executable, "-B", str(ROOT / "src/experiments/run_experiments.py"),
            "--dataset", str(rows), "--output", str(rejected), "--mode", "mixed", "--profile-stages"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert mode.returncode == 2 and not rejected.exists()
        invoke("art", ok=False)  # no fabricated baseline stages
        invoke(CONTROL_ID, 0, ok=False)
        invoke(CONTROL_ID, -1, ok=False)
        write_queries([(b"a", True, 99)])
        invoke(CONTROL_ID, ok=False)  # oracle/answer mismatch
        write_queries(workload)
        valid = queries.read_bytes()
        for bad in (valid[:-1], valid + b"extra", b"INVALID!" + valid[8:],
                    valid[:16] + b"\x02" + valid[17:], valid[:25] + struct.pack("<Q", (1 << 64) - 1) + valid[33:]):
            queries.write_bytes(bad)
            invoke(CONTROL_ID, ok=False)
    print("profile frozen sample/checksum, hit/miss splits, time conservation and malformed input checks passed")


if __name__ == "__main__":
    main()
