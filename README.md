# DigitalOcean Inference Endpoint Benchmark

An independent, request-level benchmark of 11 DigitalOcean-hosted Serverless
Inference endpoints, measured on 23–25 August 2026. The study covers low-load
latency, open-loop AIMD capacity probes, two-minute offered-load soaks, short
and long inputs, short and long outputs, mixed workloads, recovery, tools,
structured output, vision, parameter behavior, context acceptance/retrieval,
quality, token accounting, and estimated cost.

This repository is not affiliated with or endorsed by DigitalOcean.

## Bottom line

**The planned hosted-only publication matrix is complete.** All 176 planned
endpoint-by-dimension cells are resolved: 169 completed and 7 evidence-backed
unsupported, with no inconclusive matrix cells and no unreconciled requests.
That is 100% of the frozen study design—not an SLA, an eternal model catalog,
or proof of 24-hour/region-wide stability.

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

### Fast production triage

- **Strongest all-round evidence: `qwen3.8-max`.** Its exact short/short soak
  passed, tools and vision passed deterministic checks, and retrieval remained
  correct through the tested ≈1.0M-token prompt point.
- **Hold tool-dependent mixed traffic on `qwen3.5-397b-a17b` and
  `nvidia-nemotron-3-super-120b`.** Matched tool probes repeatedly returned
  provider 500 responses while adjacent controls succeeded; their mixed soaks
  therefore failed at the low-load prerequisite rather than at a discovered
  capacity knee.
- **Treat DeepSeek V4 Flash 0731 long-context acceptance cautiously.** The
  route accepted approximately 1.05M prompt tokens, but the retrieval marker
  was functionally correct only through 2,722 tokens in this synthetic test.
  Acceptance is not the same as useful long-context recall.
- **Gemma is the other guarded pilot candidate with verified vision.** Its
  short/short soak passed and context retrieval succeeded through the tested
  251,873-token point, but high-context latency was slow and variable.
- **No endpoint earned a blanket production recommendation.** In particular,
  no heterogeneous mixed-load cell passed the full acceptance, quality, and
  recovery contract.

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
  until a workload-matched soak passes. No heterogeneous mixed-load cell passed
  the strict composite. Do not infer long-context, long-output, tool, vision,
  or mixed-load capacity from a short/short result.
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
- **40,049 matched hosted request observations and 1,343 hosted load epochs**
  enter the final scientific analysis. Every physical matched-control attempt
  reconciles to its frozen semantic cell; the orphan ledger is empty. Historical
  and quarantined rows remain in the immutable evidence bundle but are excluded
  from these hosted production counts.
- **Two isolated open-loop AIMD campaigns and four workload shapes per
  endpoint.** Twenty-five of the 44 hosted endpoint/shape cells established at
  least one three-separated-confirmation healthy-rate lower bound. Results from
  different campaigns remain separate rather than being pooled into a fictional
  universal ceiling.
- **44 hosted endpoint/shape two-minute soaks:** 42 are scientifically complete.
  The two remaining mixed-load cells—Qwen 3.5 397B and Nemotron 3 Super—are
  resolved as operational failures because their low-load tool prerequisites
  repeatedly returned provider 500 responses while adjacent controls passed.
  Only the three exact short/short cells above passed the full composite.
- **208 endpoint/capability aggregate findings** backed by transport,
  deterministic functional, malformed-input, parameter-boundary, and matched
  control evidence. Vision is functionally verified only on Gemma 4 31B IT and
  Qwen 3.8 Max in this campaign.
- **Fixed context anchors plus adaptive refinements** reached roughly one
  million realized prompt tokens. Qwen 3.8 Max retrieved correctly through the
  tested ≈1.0M point; HTTP acceptance without correct retrieval is reported
  separately for other routes.
- **1,322 matched-control physical attempts** are fully reconciled across the
  closure waves. Expected validation 400s, repeated provider 500s, access-gated
  403/503 incidents, and successful controls remain distinct outcomes.
- **$329.9724 conservative cumulative exposure** under the authorized $400
  cap. The overlapping request-attributed estimate is $272.2202; the two
  figures must not be added. This is an experiment ledger, not a DigitalOcean
  invoice or proof of how credits were applied.

## Start here

- [Engineering encyclopedia PDF](report/DigitalOcean-Inference-Engineering-Encyclopedia-August-2026.pdf)
- [Methods and definitions](METHODS.md)
- [Reproduction guide](REPRODUCE.md)
- [Results guide](results/README.md)
- [Endpoint summary](results/endpoint-summary.csv)
- [Granular coverage ledger](results/coverage-ledger.jsonl)
- [Capacity-cell table](results/capacity-summary.csv)
- [Soak-cell table](results/soak-cell-summary.csv)
- [Capability table](results/capability-evidence.csv)
- [Context-envelope table](results/observed-limits.csv)
- [Time and cost ledger](results/cost-summary.json)
- [Endpoint count and token-cost summary](results/endpoint-summary.csv)

The PDF passes the frozen matrix, sample-unit, interval, schema, reconciliation,
and publication-safety gates. It still labels right-censored bounds and narrow
workload claims honestly; a complete design is not the same thing as a provider
guarantee.

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
