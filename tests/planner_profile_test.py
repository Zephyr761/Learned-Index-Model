"""Calibration provenance/strict parsing and real cost/adaptive runner checks."""
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
from run_experiments import PLANNER_IDS


def main():
    calibrator, mixed = sys.argv[1:3]
    with tempfile.TemporaryDirectory(prefix="blocklearn-planner-") as directory:
        directory = Path(directory)
        profile = directory / "costs.txt"
        subprocess.run([calibrator, "--output", str(profile), "--iterations", "1024"], check=True, stdout=subprocess.PIPE)
        original = profile.read_bytes()
        assert b"origin independent-synthetic-v1" in original
        subprocess.run([calibrator, "--validate", str(profile)], check=True, stdout=subprocess.PIPE)
        overwrite = subprocess.run([calibrator, "--output", str(profile), "--iterations", "1024"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert overwrite.returncode == 2 and profile.read_bytes() == original
        for bad in (original + b"hash_byte 2\n", original.replace(b"origin independent-synthetic-v1", b"origin benchmark-queries"),
                    original + b"unknown 4\n", original.replace(b"hash_byte ", b"hash_byte nan ignored "),
                    original.replace(b"target ", b"target incompatible-"),
                    original.replace(b"build ", b"build incompatible-")):
            path = directory / "invalid.txt"
            path.write_bytes(bad)
            failed = subprocess.run([calibrator, "--validate", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert failed.returncode == 2
        forbidden = subprocess.run([calibrator, "--rows", "benchmark-queries.bin"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert forbidden.returncode == 2
        rows = directory / "rows.bin"
        with rows.open("wb") as stream:
            stream.write(b"SIBROW1\0" + struct.pack("<Q", 384))
            for i in range(384):
                key = "https://example/shared/{:08x}".format(i).encode()
                stream.write(struct.pack("<Q", len(key)) + key + struct.pack("<Q", i))
        env = os.environ.copy()
        env["BLOCKLEARN_COST_PROFILE"] = str(profile)
        for initial, inserts in (("1", "0"), (".5", ".8")):
            checksums = set()
            for name in ("block-residual-delta-shortdir-first",) + PLANNER_IDS:
                command = [mixed, "--rows", str(rows), "--implementation", name, "--operations", "1000", "--repeats", "1",
                           "--initial-fraction", initial, "--insert-ratio", inserts, "--latency-every", "32"]
                result = subprocess.run(command, check=True, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                records = list(csv.DictReader(io.StringIO(result.stdout)))
                assert len(records) == 1 and records[0]["status"] == "passed", result.stdout
                checksums.add(records[0]["checksum"])
            assert len(checksums) == 1
        env.pop("BLOCKLEARN_COST_PROFILE")
        missing = subprocess.run([mixed, "--rows", str(rows), "--implementation", PLANNER_IDS[0],
                                  "--operations", "1", "--repeats", "1", "--insert-ratio", "0"],
                                 env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert missing.returncode != 0 and "cost profile" in missing.stdout.lower().replace("_", " "), missing.stdout
    print("independent calibration, strict/frozen configuration, both new factories and matching static/mixed checksums passed")


if __name__ == "__main__":
    main()
