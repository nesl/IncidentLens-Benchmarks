import tempfile
import unittest
from pathlib import Path

from evaluation.synthetic_emitter import observation_to_report
from evaluation.real_emitter import make_report


class SyntheticEmitterContractTest(unittest.TestCase):
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
            report = observation_to_report(observation, Path(folder))
        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(report["report_id"], "obs-1")

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
