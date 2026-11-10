import tempfile, unittest
from sems_attention.profiles.four_stage import ProfileExtractor


class RepairTest(unittest.TestCase):
    def test_one_targeted_repair_and_cache(self):
        class Client:
            def __init__(self):
                self.stages = []

            def complete_json(self, module, system, payload, **kw):
                stage = kw.get("stage")
                self.stages.append(stage)
                if stage == "Profile Review":
                    return {}, None
                if stage == "Profile Repair":
                    return {
                        "user_id": "u",
                        "narrative": "Supported narrative.",
                        "preferences": [
                            {"aspect": "x", "description": "y", "evidence_review_ids": ["r1"]}
                        ],
                        "uncertainties": [],
                    }, None
                return {
                    "user_id": "u",
                    "narrative": "Supported narrative.",
                    "preferences": [
                        {"aspect": "x", "description": "y", "evidence_review_ids": ["bad"]}
                    ],
                    "uncertainties": [],
                }, None

            def record_cache_hit(self, *args):
                pass

        c = Client()
        h = [{"user_id": "u", "review_id": "r1", "timestamp_ms": 1, "text": "test"}]
        with tempfile.TemporaryDirectory() as d:
            extractor = ProfileExtractor(c, d, "test", repair_invalid=True)
            v = extractor.extract("u", h)
            self.assertEqual(v["preferences"][0]["evidence_review_ids"], ["r1"])
            self.assertEqual(c.stages.count("Profile Repair"), 1)
            self.assertEqual(extractor.extract("u", h), v)
            self.assertEqual(c.stages.count("Profile Repair"), 1)


if __name__ == "__main__":
    unittest.main()
