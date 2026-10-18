"""Lightweight guard/report tests; no emulated or fabricated HOT benchmark."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_linux_hot", ROOT / "tools/prepare_linux_hot.py")
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


class LinuxHotEntryTest(unittest.TestCase):
    def test_host_gate(self):
        for system, machine in (("Linux", "x86_64"), ("Linux", "AMD64")):
            ENTRY.check_host(system, machine)
        for system, machine in (("Darwin", "arm64"), ("Darwin", "x86_64"),
                                ("Linux", "aarch64"), ("Linux", "arm64"), ("Windows", "AMD64")):
            with self.assertRaises(RuntimeError):
                ENTRY.check_host(system, machine)

    def test_no_directory_on_rejected_host(self):
        with tempfile.TemporaryDirectory(prefix="hot-entry-") as directory:
            build = Path(directory) / "must-not-exist"
            with mock.patch.object(ENTRY.platform, "system", return_value="Darwin"), \
                    mock.patch.object(ENTRY.platform, "machine", return_value="arm64"), \
                    mock.patch.object(ENTRY.sys, "argv", ["prepare_linux_hot.py", "--build", str(build)]), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(ENTRY.main(), 2)
            self.assertFalse(build.exists())

    def test_available_means_all_real_targets(self):
        valid = "implementation,status,notes\n" + "\n".join(
            name + ",available,real" for name in ("hot", "art", "block-residual-delta-shortdir-first"))
        ENTRY.require_hot(valid)
        for bad in (valid.replace("hot,available", "hot,unsupported"),
                    valid.replace("art,available,real\n", ""),
                    valid + "\nhot,available,duplicate", valid + "\nextra,available", "wrong,header\n"):
            with self.assertRaises(RuntimeError):
                ENTRY.require_hot(bad)

    def test_suite_verdicts_and_checksums(self):
        names = ("hot", "art", "block-residual-delta-shortdir-first")
        valid = {"status": "completed",
                 "static_statuses": [{"id": name, "status": "passed"} for name in names],
                 "mixed_results": [{"implementation": name, "status": "passed", "checksum": "12345"} for name in names]}
        ENTRY.require_suite(valid)
        cases = []
        bad = copy.deepcopy(valid); bad["status"] = "completed_with_unsupported"; cases.append(bad)
        bad = copy.deepcopy(valid); bad["static_statuses"][0]["status"] = "unsupported"; cases.append(bad)
        bad = copy.deepcopy(valid); bad["static_statuses"].pop(); cases.append(bad)
        bad = copy.deepcopy(valid); bad["mixed_results"][0]["checksum"] = "999"; cases.append(bad)
        bad = copy.deepcopy(valid); bad["mixed_results"][0]["checksum"] = ""; cases.append(bad)
        bad = copy.deepcopy(valid); bad["mixed_results"][0]["status"] = "skipped"; cases.append(bad)
        bad = copy.deepcopy(valid); bad["mixed_results"].pop(); cases.append(bad)
        bad = copy.deepcopy(valid); bad["mixed_results"].append(bad["mixed_results"][0]); cases.append(bad)
        for bad in cases:
            with self.assertRaises(RuntimeError):
                ENTRY.require_suite(bad)

    def test_fixture_domain_and_boundary(self):
        with tempfile.TemporaryDirectory(prefix="hot-fixture-") as directory:
            path = Path(directory) / "keys.txt"
            ENTRY.write_fixture(path)
            keys = path.read_bytes().splitlines()
            self.assertEqual(len(keys), len(set(keys)))
            self.assertGreater(len(keys), 4096)
            self.assertEqual(max(map(len, keys)), 254)
            self.assertTrue(all(b"\0" not in key for key in keys))
            self.assertIn(b"p" * 253, keys)
            self.assertIn(b"p" * 254, keys)

    def test_source_pins(self):
        with tempfile.TemporaryDirectory(prefix="hot-sources-") as directory:
            root = Path(directory)
            benchmark = root / "benchmark/learned-index"
            files = [root / "related_works/hot/libs/hot/single-threaded/include/hot/singlethreaded/HOTSingleThreaded.hpp",
                     root / "related_works/libart/src/art.c", benchmark / "scripts/benchmark.py"]
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            def git(command, **kwargs):
                name = Path(command[2]).name
                return ENTRY.PINS[name] + "\n" if command[3] == "rev-parse" else ""

            with mock.patch.object(ENTRY.subprocess, "check_output", side_effect=git):
                ENTRY.check_sources(root, benchmark)
            with mock.patch.object(ENTRY.subprocess, "check_output", return_value="wrong-revision\n"):
                with self.assertRaisesRegex(RuntimeError, "revision"):
                    ENTRY.check_sources(root, benchmark)
            with mock.patch.object(ENTRY.subprocess, "check_output", side_effect=lambda cmd, **kw:
                    git(cmd, **kw) if cmd[3] == "rev-parse" else " M tracked.hpp\n"):
                with self.assertRaisesRegex(RuntimeError, "tracked source changes"):
                    ENTRY.check_sources(root, benchmark)


if __name__ == "__main__":
    unittest.main()
