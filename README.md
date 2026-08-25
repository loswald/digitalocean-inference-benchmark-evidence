# DigitalOcean Inference Endpoint Benchmark

An independent, request-level benchmark of 11 DigitalOcean-hosted Serverless
Inference endpoints, measured on 23–24 August 2026. The study covers low-load
latency, open-loop AIMD capacity probes, two-minute offered-load soaks, short
and long inputs, short and long outputs, mixed workloads, recovery, tools,
structured output, vision, parameter behavior, context acceptance/retrieval,
quality, token accounting, and estimated cost.

This repository is not affiliated with or endorsed by DigitalOcean.

## Bottom line

**This is useful evidence, but it is not a complete certification of the
portfolio.** The hosted-only publication matrix resolves 95 of 176 planned
endpoint-by-dimension cells (54.0%): 88 completed, 7 evidence-backed
unsupported, and 81 inconclusive. Every unresolved cell remains visible.

**Current inference access: restored after prepayment.** On 24 August, the
authorized inference credential began returning HTTP 403 as the Serverless
Inference balance was depleted. After the owner replenished the balance on
25 August, the same credential again returned HTTP 200 from `/v1/models`.
The main account-control API still returns 403, but it is a separate surface
and is not the inference-readiness gate. Run two cheap serial marker controls
before production onboarding or a new load wave.

Only three exact endpoint/workload/rate combinations passed the benchmark's
strict two-minute soak rule:

| Endpoint | Workload | Offered load that passed | What this means |
|---|---|---:|---|
| `deepseek-v4-flash-0731` | short input / short output | 1.0 RPS (60 RPM) | This one tested point passed; it is not a ceiling or SLA. |
| `gemma-4-31B-it` | short input / short output | 1.0 RPS (60 RPM) | This one tested point passed; it is not a ceiling or SLA. |
| `qwen3.8-max` | short input / short output | 1.0 RPS (60 RPM) | This one tested point passed; it is not a ceiling or SLA. |

No other endpoint/workload cell passed the complete two-minute acceptance and
recovery rule at its tested candidate rate. That does **not** prove the
endpoint cannot work at a lower rate; it means this campaign did not establish
a passing operating point.

### Action for engineers

- Keep a positive Serverless Inference prepaid balance and pass two serial
  streamed marker controls. A failed 401/403 control stops the lane; it is not
  a capability result.
- Treat the three 60-RPM results as narrow passing observations, not production
  recommendations. Start below the tested point, enforce your own latency and
  quality SLOs, and ramp with an adaptive controller.
- Put every endpoint behind bounded retries with jitter, multiplicative
  backoff after 429/5xx congestion, a concurrency ceiling, and a circuit
  breaker. Never retry validation 4xx responses.
- For the other endpoint/workload combinations, use shadow traffic or a canary
  until a workload-matched soak passes. Do not infer long-context, long-output,
  tool, vision, or mixed-load capacity from a short/short result.
- Recalibrate after model, account, region, or service changes. DigitalOcean's
  public quota is account-scoped, while the observations also show
  route-specific throttling behavior.
- Do not use the per-request post-TTFT proxy as direct decoder speed. Aggregate
  output goodput over a complete epoch or soak block is the engineering metric.

## What was actually run

- **11 exact DigitalOcean-hosted model IDs** and four load shapes per endpoint.
- A twelfth historical endpoint, `arcee-trinity-large-thinking`, was selected
  in error. DigitalOcean documents it outside the hosted-model table, so its
  3,210 rows and approximately $9.4861 of token-attributed usage are retained
  only in the incident appendix and excluded from production comparisons and
  every future spend-bearing default.
- **10,685 requests in the fresh endpoint-isolated AIMD campaign:** 10,332
  HTTP 200 and 353 HTTP 429; no 402, 5xx, or timeout in that campaign.
- **532 capacity-valid AIMD epochs.** Twenty-two of 48 fresh endpoint/shape
  cells met the three-separated-confirmation rule; 12 remained right-censored.
  Different campaigns sometimes found different healthy rates, so they are
  reported separately and never pooled into a fictional universal ceiling.
- **48 two-minute endpoint/shape soaks:** all executed, 45 were scientifically
  complete and 3 were transport-gated. Follow-up closure waves attempted the
  unresolved cells, but only the three short/short cells above passed the full
  acceptance-plus-recovery rule.
- **1,260 capability cells**, with 1,248 provider attempts.
- **180 fixed context probes plus adaptive refinements** across all endpoints.
- **304 completion/closure probes**, of which 61 were conclusive under the
  strict analysis contract.
- **52 additional matched-closure physical attempts** during the access
  incident: 32 HTTP 403, 4 HTTP 500, and 16 HTTP 503. Twelve semantic cells
  reached a terminal inconclusive state; none became a support/rejection claim.
- **$237.4876 conservative cumulative exposure** under a $400 campaign cap.
  This is an experiment ledger, not a DigitalOcean invoice or proof of credit
  application.

## Start here

- [Engineering encyclopedia PDF](report/DigitalOcean-Inference-Engineering-Encyclopedia-August-2026.pdf)
- [Methods and definitions](METHODS.md)
- [Reproduction guide](REPRODUCE.md)
- [Results guide](results/README.md)
- [Endpoint summary](results/endpoint-summary.csv)
- [Coverage matrix](results/coverage-ledger.jsonl)
- [Capacity-cell table](results/capacity-summary.csv)
- [Soak-cell table](results/soak-cell-summary.csv)
- [Capability table](results/capability-evidence.csv)
- [Context-envelope table](results/observed-limits.csv)
- [Time and cost ledger](results/cost-summary.json)

The PDF is deliberately labelled as an incomplete-evidence report. A report
that issued many requests is not automatically a complete benchmark.

## Interpretation rules

- HTTP 200 means the endpoint returned a response; it does not mean the answer
  was correct.
- A wrong valid answer is a model-plus-serving outcome. Without a
  version-matched external control, it is not automatically a DigitalOcean
  infrastructure failure.
- Prompt tokens divided by TTFT is an end-to-end prefill proxy, not direct
  server-side prefill speed.
- Server-sent events can batch tokens. The old first-to-last event ratio that
  produced extreme token/s numbers is rejected. The corrected proxy uses
  server-reported completion tokens over request end minus TTFT, censors
  timing-unstable sub-100-ms denominators, and is still labelled a proxy.
- No p99 is presented as reliable without roughly 1,000 relevant independent
  observations in the exact comparison cell.
- Missing, rejected, censored, unsupported, and zero are distinct states.

## Repository map

```text
do_benchmark/   standalone request, timing, scoring, and accounting code
scripts/        direct experiment, analysis, plotting, manifest, and PDF tools
tests/          offline contract, accounting, and analysis tests
report/         evidence PDF
results/        canonical tables, ledgers, clean figures, and sanitized evidence
audit/          provenance and evidence-boundary notes
```

## Offline verification

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/build-digitalocean-benchmark-manifest.py .
```

Offline analysis and plotting make no inference calls. Live runners do. Before
any billable rerun, set a new explicit time/cost envelope and keep the key only
in the process environment. Never place a key in a command, issue, report, raw
journal, or commit.

## Documentation freeze

The benchmark used DigitalOcean's official [model
catalog](https://docs.digitalocean.com/products/inference/details/models/),
[pricing](https://docs.digitalocean.com/products/inference/details/pricing/),
[limits](https://docs.digitalocean.com/products/inference/details/limits/),
[multimodal guide](https://docs.digitalocean.com/products/inference/how-to/use-multimodal-inference/),
and [quota-header
reference](https://docs.digitalocean.com/products/inference/reference/quota-specific-response-headers/).
Documentation and service behavior can change after the measurement window.

## License and data notice

The code is available under the MIT License. See [DATA-NOTICE.md](DATA-NOTICE.md)
for the narrower terms governing preserved measurement evidence.
