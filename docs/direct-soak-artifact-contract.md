# Direct two-minute soak artifact contract

`scripts/run-digitalocean-direct-soak.py` consumes a completed direct-AIMD
artifact directory and writes a separate, secret-free soak evidence directory.
The preferred input is one valid low-load baseline plus three separated,
healthy confirmation epochs at the same rate, with complete request receipts.
For the one hash-pinned reconciled campaign supported by this release, the
runner may instead choose the lowest fully receipted healthy AIMD epoch as an
explicitly exploratory input. That input is never labelled AIMD-confirmed; only
a completed two-minute soak can confirm its tested operating point. The plan
retains the evidence level, target rate, and realized source schedule rates.

The default target is the frozen 12-endpoint inventory crossed with these four
workload shapes:

| `shape` | Input | Output | Deterministic task mix |
| --- | --- | --- | --- |
| `short_short` | short | short | exact text |
| `input32k_short` | the AIMD manifest's fixed long-input anchor | short | retrieval needle |
| `short_long` | short | the AIMD manifest's controlled long-output anchor | exact-length generation |
| `mixed` | heterogeneous | heterogeneous | exact text, 4K retrieval, controlled 512-word generation, JSON, and tool call |

## Fixed execution design

- Four serial low-load requests are made first. Each is the low-load member of
  one deterministic quality pair. `mixed` adds a fifth pair for its tool-call
  family, assigned to block four, so all five mixed task families are paired.
- The soak is one continuous 120-second open-loop arrival schedule. It is split
  before execution into four 30-second arrival cohorts by global arrival
  offset. It is never restarted or rounded independently per block. The first
  required arrivals in each cohort are exact-payload near-load pair members.
- An independent semaphore enforces the concurrency ceiling. Scheduled
  arrivals wait behind it instead of disappearing, preventing coordinated
  omission.
- Recovery is a separate 30-second open-loop phase at half the soak rate. Its
  requested rate, realized finite-window schedule rate, and predeclared
  transport, latency, and deterministic-quality pass or failure reasons are
  reported separately. Recovery requires a 100% deterministic quality-pass
  rate and no quality drop greater than five percentage points from low load.
- Only one endpoint/workload cell runs at a time.
- A cross-process advisory lease covers runtime-state reload through final
  summary write. A second process fails immediately; it cannot send from a
  stale ledger. Provider-send and hard campaign deadlines are both mandatory,
  and arrival waits plus in-flight calls are bounded by the hard deadline.

## Files and row schemas

| File | Identity field | Schema | Purpose |
| --- | --- | --- | --- |
| `manifest.json` | `campaign_id` | `do_direct_soak_campaign_v1` | source hashes, design, sanitization, and claim scope |
| `plan.json` | `plan_sha256` | `do_direct_soak_plan_v1` | exact candidates, workload contracts, counts, and cost ceiling |
| `reservations.jsonl` | `request_id` | `do_direct_reservation_v1` | durable reservation before each provider send |
| `requests.jsonl` | `request_id` | `do_direct_soak_request_v1` | one sanitized terminal request receipt |
| `phases.jsonl` | `phase_id` | `do_direct_soak_phase_v1` | low-load, 120-second soak, and recovery summaries |
| `analysis-blocks.jsonl` | `analysis_block_id` | `do_direct_soak_analysis_block_v1` | four preregistered cohort summaries per complete cell |
| `quality-pairs.jsonl` | `quality_pair_id` | `do_direct_soak_quality_pair_v1` | exact-payload low/near comparisons |
| `cells.jsonl` | `cell_id` | `do_direct_soak_cell_v1` | cell completeness, block intervals, acceptance, and recovery |
| `execution-windows.jsonl` | `execution_window_id` | `do_direct_soak_execution_window_v1` | immutable send cutoff, hard deadline, and request-timeout receipts for each execution attempt |
| `summary.json` | `campaign_id` | `do_direct_soak_summary_v1` | campaign execution and scientific completeness |

Every request row carries these report-routing fields:

- `provider`, `endpoint_id`, `model_id`, `shape`, `phase`, `task_family`
- `workload_tags.benchmark_lane = direct_two_minute_soak`
- `workload_tags.load_phase` in `paired_low_load`, `two_minute_soak`, or
  `post_soak_recovery`
- `workload_tags.analysis_block_index` in `0..3` for soak arrivals
- `quality_pair_id` and `quality_pair_role` (`low_load` or `near_load`)
- `workload_tags.candidate_rate_rps`, `streaming`, and
  `task_recipe_version`

Rows retain payload/response hashes, byte counts, token usage, timings, scores,
status categories, cost accounting, and sanitized rate-limit signals. They do
not retain credentials, prompts, model text, reasoning text, tool arguments,
response bodies, or raw headers.

## Statistical and interpretation contract

Each block reports Wilson 95% intervals for success and quality-pass rates and
distribution-free DKW 95% intervals for p95 TTFT and latency. A complete cell
also reports exploratory Student-t intervals over the four predeclared blocks
for mean RPM and effective input/output TPM; the artifact explicitly notes that
four contiguous blocks do not model serial correlation.

Block acceptance uses transport/error thresholds, queue growth, TTFT and
latency relative to low load, and the exact-payload quality pair for that block.
Both pair members must independently pass their deterministic scorer. A 0-to-0
pair therefore fails, and any 1-to-0 regression is separately labelled.
Heterogeneous whole-block quality is never compared with a low-load aggregate.

`execution_complete` means the runner reached a terminal execution outcome.
`scientifically_complete` requires the full soak, all four analysis blocks,
four exact-payload pairs (five for `mixed`), and recovery. At campaign level it
requires this for all 48 endpoint/shape cells. A low-load transport gate
failure is execution-complete evidence but is not scientifically complete.

A passing cell is evidence only for the exact endpoint, workload recipe, rate,
and observed two-minute interval. It is not evidence for longer durations,
other times of day, or other offered loads.

## Cost and replay contract

The offline plan discloses the full all-failure reservation ceiling, which may
exceed the campaign cap. Launch is gated by current settled-plus-reserved
exposure plus the largest possible in-flight batch. Immediately before every
send, the durable ledger reserves a padded tokenizer-independent byte-bound
cost. Successful calls settle to provider token usage only when both prompt and
completion counters are present and valid; failed, partial, or ambiguous calls
retain their full reservation. Later requests are censored before send if they
would cross the cumulative cap.

The source AIMD summary exposure is independently reconstructed from its prior
cost, reservation journal, and terminal request accounting. `--prior-cost-usd`
is an explicit total cumulative exposure, not an additive increment, and is
rejected when lower than that reconstructed source exposure.

Campaign, cell, phase, request, block, and pair IDs are deterministic functions
of the hashed source evidence and exact science plan. A reservation without a
terminal row is classified unknown and is never replayed. A partially written
open-loop phase is closed as incomplete rather than resumed as a false
continuous soak.

On every resume, supported schema versions plus campaign, plan, model, cell,
phase, request, payload, task-family, pair, block, reservation, and execution-
window identities are reconciled before any terminal row is trusted. The plan
also binds hashes of the task recipe and deterministic scorer, the frozen
endpoint inventory, documentation dates, and exact selected model specs.
