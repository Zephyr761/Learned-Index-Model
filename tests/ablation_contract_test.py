"""Five-way planner matrix, retired IDs, manifests and static/mixed dispatch."""
import csv
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/experiments"))
from run_experiments import MODEL_IDS, DEFAULT_IDS, IDS, select_implementations


def main():
    binary = sys.argv[1]
    assert select_implementations(None, "planner") == list(DEFAULT_IDS)
    assert select_implementations(None, None) == list(DEFAULT_IDS)
    assert select_implementations([MODEL_IDS[0], "art"], None) == [MODEL_IDS[0], "art"]
    for name in IDS:
        manifest = json.loads((ROOT / "src/experiments" / name / "benchmark-adapter.json").read_text())
        assert manifest["id"] == name
        symbol = "make_" + name.replace("-", "_") + ("_baseline" if name in ("art", "hot") else "")
        assert manifest["factory_symbol"] == symbol
        assert (ROOT / "src/experiments" / name / manifest["source_dir"]).resolve() == ROOT / "src"
    manifest_ids = {json.loads(p.read_text())["id"] for p in (ROOT / "src/experiments").glob("*/benchmark-adapter.json")}
    assert manifest_ids == set(IDS), manifest_ids
    described = subprocess.check_output([binary, "--describe"], text=True)
    availability = {r["implementation"]: r for r in csv.DictReader(io.StringIO(described))}
    assert len(availability) == len(IDS)
    assert set(availability) == set(IDS)
    assert all(availability[name]["status"] == "available" for name in MODEL_IDS)
    with tempfile.TemporaryDirectory(prefix="blocklearn-ablation-") as folder:
        folder = Path(folder)
        profile = folder / "costs.txt"
        subprocess.run([sys.argv[2], "--output", str(profile), "--iterations", "1024"],
                       check=True, stdout=subprocess.PIPE)
        env = os.environ.copy()
        env["BLOCKLEARN_COST_PROFILE"] = str(profile)
        rows = folder / "rows.bin"
        with rows.open("wb") as stream:
            stream.write(b"SIBROW1\0" + struct.pack("<Q", 2400))
            for i in range(2400):
                key = ("https://example.org/common/{:08x}".format(i)).encode()
                stream.write(struct.pack("<Q", len(key)) + key + struct.pack("<Q", i))
        checksums = set()
        mixed_checksums = set()
        for name in MODEL_IDS + ("art",):
            command = [binary, "--rows", str(rows), "--implementation", name,
                       "--operations", "1800", "--repeats", "1", "--latency-every", "32"]
            result = list(csv.DictReader(io.StringIO(subprocess.check_output(
                command + ["--initial-fraction", "1", "--insert-ratio", "0"], text=True, env=env))))
            assert len(result) == 1 and result[0]["implementation"] == name
            assert result[0]["status"] == "passed", result
            checksums.add(result[0]["checksum"])
            result = list(csv.DictReader(io.StringIO(subprocess.check_output(
                command + ["--initial-fraction", ".5", "--insert-ratio", ".8"], text=True, env=env))))
            assert result[0]["status"] == "passed" and int(result[0]["inserts"]) > 100
            mixed_checksums.add(result[0]["checksum"])
            if name != "art":
                assert int(result[0]["merges"]) > 0
        assert len(checksums) == len(mixed_checksums) == 1
        # Removed entry points and ambiguous matrix selection fail before output.
        retired = ("block-plain", "block-hash", "block-residual", "block-residual-delta",
                   "block-hash-shortdir", "block-hash-first", "block-hash-shortdir-first",
                   "block-residual-delta-shortdir", "block-residual-delta-first")
        for name in retired:
            result = subprocess.run([binary, "--rows", str(rows), "--implementation", name],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            assert result.returncode == 2
        invalid = [["--implementation", name] for name in retired]
        invalid += [["--ablation", kind] for kind in ("directory", "probe", "both")]
        invalid += [["--ablation", "planner", "--implementation", "art"],
                    ["--implementation", MODEL_IDS[0], "--implementation", MODEL_IDS[0]]]
        for extra in invalid:
            rejected = folder / "must-not-exist"
            result = subprocess.run([sys.executable, "-B", str(ROOT / "src/experiments/run_experiments.py"),
                "--dataset", str(rows), "--output", str(rejected), *extra],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert result.returncode == 2 and not rejected.exists()
    print("five-way planner matrix, five manifests, retired entry rejection and static/mixed checksums passed")


if __name__ == "__main__":
    main()
