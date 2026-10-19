import unittest

from disturbance_exp.scenario import validate_scenario
from disturbance_exp.workload import generate_trace


class WorkloadTests(unittest.TestCase):
    def test_fixed_seed_repeats_trace(self):
        cfg = validate_scenario({
            "experiment": {"seed": 7, "duration_sec": 10},
            "limits": {"max_requests": 4},
            "workload": {"arrival": "poisson", "base_qps": 1, "input_tokens": 8, "output_tokens": 2},
        })
        left, right = generate_trace(cfg), generate_trace(cfg)
        self.assertEqual(left, right)
        self.assertTrue(all(item.input_tokens + item.output_tokens <= 8192 for item in left))

    def test_qps_change_is_deterministic(self):
        cfg = validate_scenario({
            "experiment": {"seed": 12, "duration_sec": 12},
            "limits": {"max_requests": 6},
            "workload": {"arrival": "fixed", "base_qps": 0.5, "input_tokens": 8, "output_tokens": 2},
            "events": [{"at_sec": 4, "type": "workload", "action": "set_qps", "value": 2.0}],
        })
        trace = generate_trace(cfg)
        self.assertEqual(trace, generate_trace(cfg))
        self.assertGreaterEqual(len(trace), 3)
        self.assertTrue(all(a.scheduled_arrival_sec <= b.scheduled_arrival_sec for a, b in zip(trace, trace[1:])))


if __name__ == "__main__":
    unittest.main()
