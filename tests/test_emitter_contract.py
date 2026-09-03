import tempfile
import unittest
import json
from pathlib import Path

from evaluation.synthetic_observations import parser, replay_settings, to_common_observation
from evaluation.real_emitter import make_report


class SyntheticEmitterContractTest(unittest.TestCase):
    def test_replay_uses_configured_dataset_and_receiver(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "runs"
            config_path = Path(folder) / "config.json"
            config_path.write_text(json.dumps({
                "replay": {
                    "dataset_root": str(root),
                    "receiver": {"enabled": True, "host": "receiver.test", "port": 9001},
                }
            }), encoding="utf-8")
            args = replay_settings(parser().parse_args(["--config", str(config_path)]))

        self.assertEqual(args.root, root)
        self.assertEqual(args.receiver_host, "receiver.test")
        self.assertEqual(args.receiver_port, 9001)

    def test_emitter_produces_versioned_report(self):
        observation = {
            "observation_id": "obs-1",
            "incident_id": "incident-1",
            "source": "weather_data",
            "modality": "time_series",
            "time": "2026-08-15T12:00:00+00:00",
            "sensor_location": {"latitude": 34.0, "longitude": -118.2},
            "row": {"sensor_id": "weather-1", "temperature": 75},
        }
        with tempfile.TemporaryDirectory() as folder:
            common = to_common_observation(observation, Path(folder)).to_dict()
        self.assertEqual(common["schema_version"], "urban-observation.v1")
        self.assertTrue(common["id"].startswith("synthetic:"))
        self.assertNotIn("incident-1", json.dumps(common))

    def test_real_emitter_produces_the_same_versioned_contract(self):
        report = make_report(
            report_id="real-1",
            report_date="2026-08-15T12:00:00+00:00",
            sensor_id="weather-1",
            sensor_name="Weather 1",
            sensor_type="weather_data",
            latitude=34.0,
            longitude=-118.2,
            data={"temperature": 75},
            metadata={"real_data": True},
        )
        self.assertEqual(report["schema_version"], "1.0")


if __name__ == "__main__":
    unittest.main()
