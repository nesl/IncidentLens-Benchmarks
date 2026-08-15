import json
import unittest
from pathlib import Path


class RealLabelArtifactTest(unittest.TestCase):
    def test_corrected_labels_have_required_evaluation_fields(self):
        path = Path(__file__).resolve().parents[1] / "evaluation" / "ground_truth" / "real" / "low_level_gt_corrected.json"
        labels = json.loads(path.read_text())
        self.assertTrue(labels)
        required = {
            "incident_id", "final_name", "incident_type", "location",
            "start_datetime_pacific", "end_datetime_pacific",
            "earliest_article_datetime_pacific",
        }
        for label in labels.values():
            self.assertTrue(required <= set(label))


if __name__ == "__main__":
    unittest.main()
