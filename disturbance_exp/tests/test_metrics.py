import unittest

from disturbance_exp.metrics import parse_benchmark_payload, percentile


class MetricsTests(unittest.TestCase):
    def test_benchmark_token_timeline(self):
        parsed = parse_benchmark_payload({
            "num_output_tokens_cf": 3,
            "per_token_latency": [[1.0, 10.0], [1.1, 4.0], [1.2, 5.0]],
        })
        self.assertEqual(parsed["server_ttft_ms"], 10.0)
        self.assertEqual(parsed["server_tpot_ms"], 4.5)
        self.assertTrue(parsed["timeline_matches_output"])

    def test_mismatched_or_single_token_timeline_has_no_tpot(self):
        mismatch = parse_benchmark_payload({"num_output_tokens_cf": 3, "per_token_latency": [[1, 10], [2, 4]]})
        single = parse_benchmark_payload({"num_output_tokens_cf": 1, "per_token_latency": [[1, 10]]})
        self.assertIsNone(mismatch["server_tpot_ms"])
        self.assertIsNone(single["server_tpot_ms"])

    def test_percentiles_interpolate(self):
        self.assertEqual(percentile([1, 2, 3, 4], .5), 2.5)


if __name__ == "__main__":
    unittest.main()
