"""Entry point for the invocation-scoped 07 opportunity service and guard."""
from __future__ import annotations

import argparse
from pathlib import Path

from .real_opportunity_budget_override import OpportunityBudgetLedger


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=('server', 'watchdog'), required=True)
    for name in ('root', 'run-dir', 'config', 'lease', 'server-name', 'guard-name'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)

    # Reuse the audited real service's ownership, foreign-Ray refusal, startup,
    # process tracking and cleanup. Only its budget view is replaced by this
    # invocation's explicit monotonic lease.
    from . import real_service
    lease = Path(args.lease)
    real_service.BudgetLedger = lambda run_dir: OpportunityBudgetLedger(run_dir, lease)
    real_service.service_names = lambda _run_id: (args.server_name, args.guard_name)
    return (real_service.run_server(args) if args.role == 'server'
            else real_service.run_watchdog(args))


if __name__ == '__main__':
    raise SystemExit(main())
