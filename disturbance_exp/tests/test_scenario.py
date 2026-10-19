import unittest

from disturbance_exp.scenario import ScenarioError, validate_scenario


class ScenarioTests(unittest.TestCase):
    def test_defaults_and_loopback_service(self):
        cfg = validate_scenario({})
        self.assertEqual(cfg["service"]["base_url"], "http://127.0.0.1:8080")
        self.assertEqual(cfg["limits"]["max_inflight"], 2)

    def test_rejects_public_target(self):
        with self.assertRaises(ScenarioError):
            validate_scenario({"service": {"base_url": "http://example.com:8080"}})

    def test_rejects_unsafe_gpu_memory(self):
        with self.assertRaises(ScenarioError):
            validate_scenario({"experiment": {"duration_sec": 10}, "events": [
                {"at_sec": 1, "type": "gpu_compute", "memory_limit_mib": 512}
            ]})


if __name__ == "__main__":
    unittest.main()
