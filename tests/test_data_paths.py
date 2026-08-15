import json
import os
import tempfile
import tarfile
import unittest
from pathlib import Path

from evaluation.data_process import configured_raw_root, copy_and_untar_raw_data_to_temp
from evaluation.real_emitter import parse_args


class BenchmarkDataPathTest(unittest.TestCase):
    def test_archive_root_comes_from_selected_config(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path = Path(folder) / "config.json"
            config_path.write_text(json.dumps({"paths": {"raw_archive_root": "/tmp/test-archive"}}))
            previous = os.environ.get("URBAN_SYSTEM_CONFIG")
            try:
                os.environ["URBAN_SYSTEM_CONFIG"] = str(config_path)
                self.assertEqual(configured_raw_root(), "/tmp/test-archive")
            finally:
                if previous is None:
                    os.environ.pop("URBAN_SYSTEM_CONFIG", None)
                else:
                    os.environ["URBAN_SYSTEM_CONFIG"] = previous

    def test_cli_roots_override_config(self):
        args = parse_args(["--raw-root", "/tmp/raw", "--temp-root", "/tmp/extracted"])
        self.assertEqual(args.raw_root, "/tmp/raw")
        self.assertEqual(args.temp_root, "/tmp/extracted")

    def test_documented_source_date_tar_layout_extracts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_dir = root / "raw" / "weather_data"
            source_dir.mkdir(parents=True)
            payload = root / "payload.csv"
            payload.write_text("timestamp,value\n2026-08-15T00:00:00Z,1\n")
            with tarfile.open(source_dir / "20260815.tar", "w") as archive:
                archive.add(payload, arcname="payload.csv")
            copy_and_untar_raw_data_to_temp(
                date_strings=["20260815"],
                data_sources=["weather_data"],
                raw_root=root / "raw",
                temp_root=root / "extracted",
                strict=True,
            )
            self.assertTrue((root / "extracted" / "weather_data" / "20260815" / "payload.csv").exists())


if __name__ == "__main__":
    unittest.main()
