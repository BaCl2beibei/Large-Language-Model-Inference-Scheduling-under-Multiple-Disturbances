import unittest

from rl_scheduler.full_completion import cvar95, percentile_type7, summarize_full_cohort
from rl_scheduler.real_metrics import Checkpoint, Event, RequestLedger


class FullCompletionContractTest(unittest.TestCase):
    def test_v2_keeps_audited_service_failure_for_the_reward(self):
        from rl_scheduler.cohort_reward import MigPPOV2Reward
        ledger = RequestLedger('run', 'ep', start_ns=0, arrival_end_ns=1,
            score_deadline_ns=None, expected_sources=['client'],
            expected_request_ids=['r'], full_completion=True)
        ledger.ingest([
            Event('client', 1, 'planned', 0, 0, 'run', 'ep', 'r', data={'scheduled_ns': 0}),
            Event('client', 2, 'sent', 1, 1, 'run', 'ep', 'r'),
            Event('client', 3, 'accepted', 2, 2, 'run', 'ep', 'r', server_request_id='sid'),
            Event('client', 4, 'server_terminal', 3, 3, 'run', 'ep', 'r', server_request_id='sid',
                  data={'outcome': 'service_error'}),
            Event('client', 5, 'client_terminal', 4, 4, 'run', 'ep', 'r', server_request_id='sid',
                  data={'outcome': 'service_error', 'http_status': 500}),
        ])
        ledger.acknowledge(Checkpoint('client', 5, 10, 10, 0, True))
        trace = [dict(request_id='r', input_tokens=128, output_tokens=2, initial_instance_index=0)]
        self.assertEqual(summarize_full_cohort(ledger, trace, through_ns=10)['status'], 'measurement_incomplete')
        summary = summarize_full_cohort(ledger, trace, through_ns=10, allow_request_failures=True)
        self.assertEqual(summary['status'], 'passed', summary['issues'])
        summary['route_audit_passed'] = True
        snapshot = dict(received_ns=0, instances=[dict(instance_id=str(i), waiting=0,
                        used_blocks=0, total_blocks=1000) for i in range(4)])
        result = MigPPOV2Reward(hotspot_threshold=.7).reconstruct(summary, [0], [dict(snapshot=snapshot)])
        self.assertEqual(result['windows'][0]['failure_count'], 1)
        self.assertGreater(result['windows'][0]['D_integral'], 0.)
        self.assertAlmostEqual(result['return'], -1/64)

    def test_type7_and_fractional_empirical_cvar(self):
        self.assertEqual(percentile_type7([0, 10], .5), 5)
        self.assertEqual(cvar95(list(range(1, 11))), 10)
        values = list(range(1, 129))
        expected = (sum(values[-6:]) + .4 * values[-7]) / 6.4
        self.assertAlmostEqual(cvar95(values), expected)

    def test_no_completion_deadline_accepts_late_terminal_and_rejects_fixed_score(self):
        ledger = RequestLedger('run', 'ep', start_ns=10, arrival_end_ns=20,
            score_deadline_ns=None, expected_sources=['client'],
            expected_request_ids=['r1'], full_completion=True)
        events = [
            Event('client', 1, 'planned', 11, 11, 'run', 'ep', 'r1', data={'scheduled_ns': 15}),
            Event('client', 2, 'sent', 30, 30, 'run', 'ep', 'r1'),
            Event('client', 3, 'client_first', 31, 31, 'run', 'ep', 'r1'),
            Event('client', 4, 'client_output', 31, 31, 'run', 'ep', 'r1', data={'cumulative_tokens': 2}),
            Event('client', 5, 'client_terminal', 40, 40, 'run', 'ep', 'r1',
                  data={'outcome': 'success', 'http_status': 200}),
        ]
        ledger.ingest(events)
        ledger.acknowledge(Checkpoint('client', 5, 50, 50, 0, True))
        self.assertEqual(ledger.audit(through_ns=50, require_cleanup=True)['status'], 'passed')
        self.assertEqual(ledger._rows(occurred_by_ns=50)[0][0]['score_outcome'], 'success')
        with self.assertRaises(RuntimeError):
            ledger.freeze_score()

    def test_full_cohort_summary_requires_matching_server_and_client_lengths(self):
        ledger = RequestLedger('run', 'ep', start_ns=10, arrival_end_ns=20,
            score_deadline_ns=None, expected_sources=['client', 'server'],
            expected_request_ids=['r1'], full_completion=True)
        client_events = [
            Event('client', 1, 'planned', 11, 11, 'run', 'ep', 'r1', data={'scheduled_ns': 15}),
            Event('client', 2, 'sent', 30, 30, 'run', 'ep', 'r1'),
            Event('client', 3, 'client_first', 31, 31, 'run', 'ep', 'r1', server_request_id='sid'),
            Event('client', 4, 'client_output', 32, 32, 'run', 'ep', 'r1', server_request_id='sid',
                  data={'cumulative_tokens': 2}),
            Event('client', 5, 'client_terminal', 40, 40, 'run', 'ep', 'r1', server_request_id='sid',
                  data={'outcome': 'success', 'http_status': 200}),
        ]
        server_events = [
            Event('server', 1, 'accepted', 31, 31, 'run', 'ep', 'r1', server_request_id='sid'),
            Event('server', 2, 'server_terminal', 39, 39, 'run', 'ep', 'r1', server_request_id='sid',
                  data={'outcome': 'success', 'input_tokens': 128, 'output_tokens': 2}),
        ]
        ledger.ingest(client_events + server_events)
        ledger.acknowledge(Checkpoint('client', 5, 50, 50, 0, True))
        ledger.acknowledge(Checkpoint('server', 2, 50, 50, 0, True))
        trace = [{'request_id': 'r1', 'input_tokens': 128, 'output_tokens': 2,
                  'initial_instance_index': 0}]
        summary = summarize_full_cohort(ledger, trace, through_ns=50)
        self.assertEqual(summary['status'], 'passed')
        self.assertEqual(summary['primary_metrics']['client_e2e_ms']['mean'], 0.00001)
        self.assertEqual(summary['requests'][0]['actual_output_tokens'], 2)

        trace[0]['output_tokens'] = 3
        invalid = summarize_full_cohort(ledger, trace, through_ns=50)
        self.assertEqual(invalid['status'], 'measurement_incomplete')
        self.assertIsNone(invalid['primary_metrics'])


if __name__ == '__main__':
    unittest.main()
