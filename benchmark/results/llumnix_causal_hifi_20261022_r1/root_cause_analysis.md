# Root cause analysis of the Round 2 prediction gap

## Supported by recorded evidence

1. **The fitted service labels are not the system quantity being predicted.**
   P1 and D3 fit executor iteration wall times. The collection does not separate
   worker GPU execution, scheduler overhead, output forwarding, and client
   delivery, nor does it connect every iteration to the actual mixed batch that
   produced a request token. The Round 2 grouped measurement-only cross
   validation has P1 R² 0.026 and D3 R² 0.051. Seed 404 MAE/RMSE is 42.17/89.37
   ms for prefill and 30.12/71.79 ms for decode. A low local MAE therefore cannot
   establish a correct request lifecycle or aggregate throughput.

2. **A single decode cadence factor is not portable across runs.** The fitted
   value is 5.8276 over 15 calibration runs, while run-level P05/P95 are 4.66 and
   13.19. The seed 404 fixed-trace comparison completed 888 requests by 90 s at
   scale 1.0 and 414 at 5.8276, with 474 requests unfinished under the latter.
   The scale is compensating for missing execution/output semantics and can
   materially change queue and completion outcomes.

3. **Output budget and realized generation were conflated in the workload
   abstraction.** The Round 2 output model was trained on 10,347 successful
   calibration responses; requests censored at drain have no length labels. On
   the held-out QPS 9 measurement, successful requests averaged 132.1 output
   tokens against a 609.8-token mean budget. At QPS 21 they averaged 55.7 against
   a 603.8-token mean budget. Length is strongly selected by EOS, budget, load,
   and censoring. The Round 3 `min_tokens=max_tokens` probe also missed its
   target for 147 of 775 HTTP 200 responses (19.0%), so fixed-length labels are
   not reliable for this request path.

4. **Queue and KV labels are sampled states, not complete state transitions.**
   Round 2 reports 50 ms carry-forward resource grids, but historical server logs
   do not contain every admission, preemption, per-request block allocation and
   release, or exact initial routing decision. At QPS 12, closed-loop V3 reports
   mean queue 1.2 requests/instance versus 3.4 observed while mean TTFT is
   934 ms versus 12,898 ms. At QPS 18 the queue direction reverses: 18.7 versus
   1.5. One static queue correction cannot explain both states.

5. **Client completion, API token timestamps, and server token timelines have
   different endpoints.** For complete historical server timelines, mean
   client-minus-server duration ranges from 6.2 s to 23.4 s by QPS. Round 2
   correctly keeps those fields separate, but the old service fit has no event
   chain through output forwarding and API receipt. All 3,970 historical HTTP
   499 responses were drain censored, not server failures.

6. **Migration total time does not identify request pause or recovery.** The
   Round 2 M2 model has seed 404 MAE/RMSE 335/567 ms over only 31 events, and the
   pilot added six successful ON migrations without any directly measured pause
   label. Historical `cost_ms` is end-to-end duration. No validated request
   blocking or first-token recovery cost can be inferred from it.

7. **Some evidence is heterogeneous.** The original 20-run collection includes
   17 runs from before and three after the migration-slot guard change. The
   Round 2 seed 404 comparison was already used during diagnosis, and the seed
   505 length probe failed its intended control. Neither is an independent new
   validation set for a replacement model.

## New event-chain evidence from independent seed 1213

The seed 1213 model remained frozen at the seed 606/909 fit. The diagnostic
overlay only rejoined observations; it did not feed the held-out data back into
model selection. At QPS 18, the first engine output reached the API callback
with a median delay of **6.55 s (P95 35.20 s) in OFF**, versus **19 ms (P95
3.16 s) in ON**. The calibration runs at the same QPS also vary sharply: seed
606 has a 1.02 s median (P95 18.75 s), while seed 909 has a 20 ms median (P95
6.34 s). An offered-QPS-only cadence table cannot explain this run-level spread.

The measurement window also hides most of the slow first-token requests. In
seed 1213 OFF/QPS18, only **98 of 1,095 sent requests** produced an API-visible
first token inside the 60 s window; **997** are right-censored at that window.
ON/QPS18 has 547/1,095 in-window first tokens and 548 right-censored. The
reported conditional TTFT mean/P95 therefore describes only the requests that
made it through in time. It does not summarize the full offered cohort.

The event join locates the delay after `engine_request_output` and before
`request_output_at_api`; it does not yet split the output-forwarding thread,
ZMQ receive queue, API dequeue, and API callback scheduling. The simulator
currently records a generated token and its API-visible token at the same
simulated step (`_record_token` in `rl_scheduler/env.py`). It has no separate
output-delivery queue or censored first-token process. This is a confirmed
missing causal stage. Per-stage queue telemetry is still needed before
assigning the delay to one specific component.

## Remaining unresolved causes

The new traces still do not separate GPU stream execution from CPU/Ray wait in
each iteration. Queue and KV state is sampled at scheduler boundaries and on a
50 ms grid, but not at every queue transition. Output length remains
right-censored for requests aborted at the fixed drain horizon. The two
calibration migration samples are too few to identify recovery cost, and one
has a roughly 95 s request-token gap. These remain unresolved rather than
being covered by a fitted constant.

## Rebuild decisions

- Fit complete scheduled-batch wall time against batch type, prefill/decode
  counts, processed tokens, context lengths, queue and KV state, with grouped
  run validation. Report CUDA event time separately as an explanatory signal.
- Keep engine token production separate from API-visible delivery. Add a
  measured output-forwarding queue with arrival, dequeue, callback, and queue
  depth timestamps; preserve first-token right censoring at the window.
- Fit output length from actual output and censor events, conditioned on request
  budget/profile, while preserving EOS, max-budget, abort, and unfinished
  outcomes.
- Calibrate migration total, source pause, destination commit, and first-token
  recovery as separate response variables.
- Select the replacement on whole-run grouped cross validation. Seed 1213 is
  diagnostic and remains excluded from fitting. No formal PPO run starts unless
  a newly collected independent validation passes the system gates.
