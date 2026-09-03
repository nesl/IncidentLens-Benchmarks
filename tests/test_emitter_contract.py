import unittest

from evaluation.real_emitter import make_report


class RealEmitterContractTest(unittest.TestCase):
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
