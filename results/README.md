# Results guide

Start with these files:

| Question | File | Interpretation |
|---|---|---|
| Is the evidence complete? | `coverage-matrix.csv` | 12 endpoints × 16 dimensions; completed, unsupported, and inconclusive are separate. |
| Which two-minute loads passed? | `soak-cell-summary.csv` | Use `soak_acceptance_pass`; a pass applies only to that endpoint, shape, rate, and interval. |
| Where did AIMD find healthy load? | `capacity-summary.csv` | Campaign-specific observations and confirmation counts; do not pool different runs. |
| What latency/TPM was observed? | `endpoint-workload-metrics.csv` | Matched endpoint/workload cells with intervals and eligibility labels. |
| What capability behavior was observed? | `capability-evidence.csv` | Transport support, functional correctness, and malformed-input validation are separate. |
| What context/output limits were observed? | `observed-limits.csv` | Tested bounds and censoring; 429/5xx/timeout rows do not establish a hard boundary. |
| What did the campaign cost? | `cost-summary.json` | Conservative cumulative exposure and stage ledger; not a provider invoice. |
| How were extreme timing ratios handled? | `metric-audit.csv` | Legacy SSE timing and corrected post-TTFT proxy classifications. |

`normalized-requests.csv` is the sanitized row-level table for 41,595 request
identities. `normalized-epochs.csv` is the epoch-level table. Neither contains
prompt text, response text, raw bodies, raw headers, or credentials.

`analysis.json.gz` is the canonical analysis object compressed for Git hosting.
See [REPRODUCE.md](../REPRODUCE.md) for decompression and PDF regeneration.

Charts are under `charts/`. The AIMD controller figures connect only
chronological epochs; the offered-load comparison figures show separate matched
points and do not draw looping pseudo-curves.
