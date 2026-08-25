# Results guide

The frozen DigitalOcean-hosted publication matrix contains 11 endpoints × 16
engineering dimensions = 176 cells. All 176 are resolved: 169 completed and 7
evidence-backed unsupported. A resolved experiment can document a failure; it
does not imply that every feature or load point passed.

| Question | File | Interpretation |
|---|---|---|
| Is the study matrix complete? | `coverage-matrix.csv` | The exact 176-cell hosted-only publication matrix. |
| What happened in every planned subtest? | `coverage-ledger.csv` | 5,382 granular cells; completed, unsupported, operational failure, inconclusive, superseded, and skipped remain distinct. |
| Which two-minute loads passed? | `soak-cell-summary.csv` | Use `soak_acceptance_pass`; a pass applies only to that endpoint, shape, rate, and observed interval. |
| Where did AIMD confirm healthy load? | `capacity-summary.csv` | Campaign-specific lower bounds and confirmation counts; never pool different runs into one fictional ceiling. |
| What latency/TPM was observed? | `endpoint-workload-metrics.csv` | Matched endpoint/workload cells with 95% intervals and metric-eligibility labels. |
| What capability behavior was observed? | `capability-evidence.csv` | Transport acceptance, functional correctness, and malformed-input validation are separate. |
| What context/output limits were observed? | `observed-limits.csv` | Tested bounds and censoring; 429/5xx/timeout rows do not establish a hard boundary. |
| What did the campaign cost? | `cost-summary.json` | Conservative cumulative exposure and stage ledger; not a provider invoice. |
| How were extreme timing ratios handled? | `metric-audit.csv` | Every qualified, invalid, and censored timing observation and its reason. |
| Did publication-safety scanning pass? | `public-safety-scan.json` | Zero findings across the generated public bundle. |

`normalized-requests.csv` contains 42,864 sanitized request/attempt rows and
`normalized-epochs.csv` contains 1,435 load epochs. They contain no prompt text,
response text, response bodies, raw headers, or credentials. The empty
`orphan-requests.csv` confirms that every included physical attempt reconciles
to frozen evidence.

The full canonical analysis object is `analysis.json.gz`; its uncompressed
SHA-256 is recorded in `analysis-manifest.json`. The very large duplicate
request JSONL representation is intentionally omitted from Git because the CSV
is the published row-level format. The hash-sealed private analysis archive
retains every generated representation.

Charts are under `charts/`. Capacity comparisons use separate matched points;
they do not connect unrelated endpoint/workload observations into looping
pseudo-curves. The AIMD controller figures show chronological controller state
for one campaign and workload at a time.
