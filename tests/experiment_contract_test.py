"""Integration assertions, not a performance benchmark."""
import csv
import io
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def invoke(binary, rows, insert_ratio, initial_fraction, operations):
    result = subprocess.run([binary, "--rows", str(rows), "--operations", str(operations),
        "--repeats", "1", "--initial-fraction", str(initial_fraction),
        "--insert-ratio", str(insert_ratio), "--latency-every", "32"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    reader = csv.DictReader(io.StringIO(result.stdout))
    records = list(reader)
    assert len(records) == 6, result.stdout
    assert all(None not in r and all(v is not None for v in r.values()) for r in records), result.stdout
    return {r["implementation"]: r for r in records}


def main():
    binary = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="blocklearn-integration-") as folder:
        rows = Path(folder) / "rows.bin"
        with rows.open("wb") as stream:
            stream.write(b"SIBROW1\0")
            stream.write(struct.pack("<Q", 2400))
            for i in range(2400):
                key = ("https://example.org/common/{:08x}".format(i)).encode()
                stream.write(struct.pack("<Q", len(key)) + key + struct.pack("<Q", i))
        mixed = invoke(binary, rows, .8, .5, 1800)
        for name in ("block-plain", "block-hash", "block-residual"):
            assert mixed[name]["status"] == "unsupported"
            assert mixed[name]["ns_per_op"] == ""
        ours = mixed["block-residual-delta"]
        assert ours["status"] == "passed" and int(ours["inserts"]) > 100
        assert int(ours["merges"]) > 0
        for name in ("art", "hot"):
            assert mixed[name]["status"] in ("passed", "unsupported")
            if mixed[name]["status"] == "passed":
                assert mixed[name]["checksum"] == ours["checksum"]
                assert mixed[name]["full_key_comparisons"] == "", "unknown baseline counters must stay unavailable"
        static = invoke(binary, rows, 0, 1, 1000)
        checksums = set()
        for name in ("block-plain", "block-hash", "block-residual", "block-residual-delta", "art", "hot"):
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
