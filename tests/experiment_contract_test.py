"""Integration assertions, not a performance benchmark."""
import csv
import io
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/experiments"))
from run_experiments import MODEL_IDS, DEFAULT_IDS


def invoke(binary, rows, insert_ratio, initial_fraction, operations, env):
    result = subprocess.run([binary, "--rows", str(rows), "--operations", str(operations),
        "--repeats", "1", "--initial-fraction", str(initial_fraction),
        "--insert-ratio", str(insert_ratio), "--latency-every", "32"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    reader = csv.DictReader(io.StringIO(result.stdout))
    records = list(reader)
    assert len(records) == len(DEFAULT_IDS) == 5, result.stdout
    assert {r["implementation"] for r in records} == set(DEFAULT_IDS)
    assert all(None not in r and all(v is not None for v in r.values()) for r in records), result.stdout
    return {r["implementation"]: r for r in records}


def main():
    binary = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="blocklearn-integration-") as folder:
        profile = Path(folder) / "costs.txt"
        subprocess.run([sys.argv[2], "--output", str(profile), "--iterations", "1024"],
                       check=True, stdout=subprocess.PIPE)
        env = os.environ.copy()
        env["BLOCKLEARN_COST_PROFILE"] = str(profile)
        rows = Path(folder) / "rows.bin"
        with rows.open("wb") as stream:
            stream.write(b"SIBROW1\0")
            stream.write(struct.pack("<Q", 2400))
            for i in range(2400):
                key = ("https://example.org/common/{:08x}".format(i)).encode()
                stream.write(struct.pack("<Q", len(key)) + key + struct.pack("<Q", i))
        mixed = invoke(binary, rows, .8, .5, 1800, env)
        ours = mixed[MODEL_IDS[0]]
        for name in MODEL_IDS:
            assert mixed[name]["status"] == "passed" and int(mixed[name]["inserts"]) > 100
            assert int(mixed[name]["merges"]) > 0
            assert mixed[name]["checksum"] == ours["checksum"]
        for name in ("art", "hot"):
            assert mixed[name]["status"] in ("passed", "unsupported")
            if mixed[name]["status"] == "passed":
                assert mixed[name]["checksum"] == ours["checksum"]
                assert mixed[name]["full_key_comparisons"] == "", "unknown baseline counters must stay unavailable"
        static = invoke(binary, rows, 0, 1, 1000, env)
        checksums = set()
        for name in DEFAULT_IDS:
            assert static[name]["status"] in ("passed", "unsupported")
            if static[name]["status"] == "passed":
                checksums.add(static[name]["checksum"])
                assert static[name]["insert_p99_ns"] == ""
        assert len(checksums) == 1
        bad = subprocess.run([binary, "--rows", str(rows), "--implementation", "bogus"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert bad.returncode == 2
    print("static/mixed interface, real baselines, maintenance timing, CSV semantics passed")


if __name__ == "__main__":
    main()
