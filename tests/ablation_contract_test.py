"""Named switches, manifests, explicit runner dispatch and CLI pairing."""
import csv
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/experiments"))
from run_experiments import ABLATION_FAMILIES, ABLATION_IDS, ABLATION_SUFFIXES, expand_ablations


def main():
    binary = sys.argv[1]
    selected = ["block-hash", "block-residual-delta", "art"]
    for kind, suffix in ABLATION_SUFFIXES.items():
        assert expand_ablations(selected, kind) == [
            "block-hash", "block-hash" + suffix,
            "block-residual-delta", "block-residual-delta" + suffix, "art"]
    assert expand_ablations(selected, None) == selected
    for name in ABLATION_IDS:
        manifest = json.loads((ROOT / "src/experiments" / name / "benchmark-adapter.json").read_text())
        assert manifest["id"] == name
        assert manifest["factory_symbol"] == "make_" + name.replace("-", "_")
        assert (ROOT / "src/experiments" / name / manifest["source_dir"]).resolve() == ROOT / "src"
    described = subprocess.check_output([binary, "--describe"], text=True)
    availability = {r["implementation"]: r for r in csv.DictReader(io.StringIO(described))}
    assert len(availability) == 12
    assert all(availability[name]["status"] == "available" for name in ABLATION_IDS)
    with tempfile.TemporaryDirectory(prefix="blocklearn-ablation-") as folder:
        folder = Path(folder)
        rows = folder / "rows.bin"
        with rows.open("wb") as stream:
            stream.write(b"SIBROW1\0" + struct.pack("<Q", 2400))
            for i in range(2400):
                key = ("https://example.org/common/{:08x}".format(i)).encode()
                stream.write(struct.pack("<Q", len(key)) + key + struct.pack("<Q", i))
        checksums = set()
        mixed_checksums = set()
        for name in ABLATION_FAMILIES + ABLATION_IDS + ("art",):
            command = [binary, "--rows", str(rows), "--implementation", name,
                       "--operations", "1800", "--repeats", "1", "--latency-every", "32"]
            result = list(csv.DictReader(io.StringIO(subprocess.check_output(
                command + ["--initial-fraction", "1", "--insert-ratio", "0"], text=True))))
            assert len(result) == 1 and result[0]["implementation"] == name
            assert result[0]["status"] == "passed", result
            checksums.add(result[0]["checksum"])
            result = list(csv.DictReader(io.StringIO(subprocess.check_output(
                command + ["--initial-fraction", ".5", "--insert-ratio", ".8"], text=True))))
            if name.startswith("block-hash"):
                assert result[0]["status"] == "unsupported" and result[0]["ns_per_op"] == ""
            else:
                assert result[0]["status"] == "passed" and int(result[0]["inserts"]) > 100
                mixed_checksums.add(result[0]["checksum"])
                if name != "art":
                    assert int(result[0]["merges"]) > 0
        assert len(checksums) == len(mixed_checksums) == 1
        # Invalid pairings must fail before creating a results directory.
        for extra in (["--implementation", "art"], ["--implementation", "block-hash-first"],
                      ["--implementation", "block-hash", "--implementation", "block-hash"]):
            rejected = folder / "must-not-exist"
            result = subprocess.run([sys.executable, "-B", str(ROOT / "src/experiments/run_experiments.py"),
                "--dataset", str(rows), "--output", str(rejected), "--ablation", "both", *extra],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert result.returncode == 2 and not rejected.exists()
    print("three ablation pairs, six manifests, static/mixed checksums and invalid CLI checks passed")


if __name__ == "__main__":
    main()
