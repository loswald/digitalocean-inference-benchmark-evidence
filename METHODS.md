# Methods

This benchmark describes what a client observed from DigitalOcean Serverless
Inference during bounded experiments on 23–25 August 2026. It does not rank the
underlying models in general, certify production readiness, or isolate model
behavior from hosting behavior. The exact API, model identifier, request shape,
load, and measurement window are part of every result.

Public claims are made only from the canonical normalized request, epoch, and
coverage tables in `results/`. Raw prompts, model responses, response bodies,
and raw headers are intentionally excluded from this public repository.

## Study design

The campaign used exact model identifiers returned by DigitalOcean's live
`/v1/models` catalog and sent chat-completion requests to
`https://inference.do-ai.run/v1`. Standalone direct runners measured breadth,
endpoint-isolated AIMD, two-minute soaks, capability behavior, context
acceptance/retrieval, and targeted completion cells. The final measurement
window ended on 25 August 2026.

The production scope is now the 11 identifiers in DigitalOcean's documented
"DigitalOcean-Hosted Models" table. `arcee-trinity-large-thinking` was
mistakenly included in the original 12-model campaign even though DigitalOcean
lists it in a separate Arcee partner-model section. Its 3,210 historical rows
remain immutable for reconciliation, but are excluded from hosted-only charts,
portfolio KPIs, recommendations, and all future provider sends. Spend-bearing
runners now reject any identifier outside the hosted-only allowlist before
network access.

The direct campaigns shared a $400 cumulative exposure cap. The final
conservative ledger was $240.825554971 after the hosted-only recovery and
closure campaign. Endpoint-isolated capacity work never
overlapped another provider workload. The context runner used one sequential
chain per endpoint and ran those chains concurrently behind an account-wide
RPM/TPM governor; those context timings are diagnostic only and are not reported
as isolated endpoint latency. Unequal endpoint latency, transport failures,
budget guards, and finite windows produced unequal coverage; missing and
inconclusive cells remain visible.

The hosted-only evidence matrix is `11 endpoints × 16 dimensions = 176 cells`.
A cell is resolved only when its dimension-specific evidence rule passes or an
exact, evidence-backed unsupported result exists. The final immutable 12-model
matrix contains 109 completed cells, 8 unsupported cells, and 75 inconclusive
cells before the partner-model exclusion; the hosted-only production view
contains 100 completed, 7 unsupported, and 69 inconclusive cells. Therefore this
repository is an incomplete-evidence report, not a complete endpoint
certification.

The three workload labels are literal request shapes, not general model classes:

- **short / short:** short prompt and short requested response;
- **long / short:** long prompt with a short answer, including exact retrieval;
- **short / long:** short prompt with a long requested response.

Task correctness was determined locally where a machine-checkable answer was
available. Examples include exact values, parsed JSON, selected tool and
arguments, executable code checks, and exact retrieval of a planted nonce.
Request acceptance and answer correctness are always separate outcomes.

## Units of analysis

A **request** is one attempted API call. An **epoch** is a bounded interval at a
fixed offered load. A **repeat** is a separately identified repetition of the
same experimental cell. Requests inside one epoch can share transient service
conditions and are not automatically independent replicates.

The primary comparison key is:

```text
endpoint × workload × request settings × offered load × campaign segment
```

Pooling across any of these fields requires an explicit estimand and weighting
rule. A result must not average unlike tasks merely to produce one endpoint
score.

## Estimands

| Measure | Estimand | Denominator and qualification |
|---|---|---|
| HTTP success | Probability that an attempted request returns the accepted success status | All attempted requests in the named cell |
| Conditional task pass | Probability of an exact task pass given a scorable successful response | Scorable successful responses only |
| End-to-end task success | Probability that an attempt both returns successfully and passes its task | All attempted requests in the named cell |
| Time to first token | Client-observed time from request start to the first streamed content token | Successful streaming responses with a valid first-token timestamp |
| End-to-end latency | Client-observed time from request start through completion | Completed requests; timeouts are reported separately, not assigned an invented duration |
| Aggregate output goodput | Successful server-reported completion tokens divided by the complete measured wall-clock interval | Request groups, AIMD epochs, or soak blocks with explicit elapsed time |
| Post-TTFT service-output proxy | Server-reported completion tokens divided by request end minus streamed TTFT | Streamed single-choice responses with authoritative usage and an interval of at least 100 ms; shorter intervals are timing-unstable and explicitly censored from this per-request rate, but remain in aggregate goodput; not direct decoder speed |
| Legacy SSE chunk-span proxy | Completion tokens divided by first-to-last streamed content-event span | Audit-only; never interpreted as decoder throughput because events batch tokens |
| Achieved throughput | Successful responses divided by the observed epoch duration | One epoch at one offered load |
| Quality-adjusted goodput | Correct responses divided by the observed epoch duration | Epochs whose tasks have a valid correctness oracle |
| Token throughput | Server-reported tokens divided by epoch duration | Unavailable when authoritative usage is absent; absence is not zero |
| Context lower bound | Largest tested prompt that met the stated acceptance or exact-retrieval rule | A tested lower bound, not the endpoint's true maximum |
| Estimated request cost | Server-reported token usage multiplied by the frozen published prices | An estimate, not an invoice or proof of credit application |

The 100 ms denominator qualification is a post-hoc measurement correction
introduced after auditing implausible short-interval ratios. It changes only
whether a per-request service-output proxy is considered timing-stable. It does
not remove the request from reliability, cost, quality, token, or aggregate
goodput accounting.

Latency and throughput are workload- and load-specific. A median measured at
low load cannot be paired with a peak rate measured in a different sweep and
presented as one operating point.

## Uncertainty

- Binary proportions use two-sided 95% Wilson intervals when request-level
  independence is a reasonable approximation and the denominator is shown.
- Load, recovery, and mixed-endpoint comparisons use the epoch or independently
  identified repeat as the resampling unit. Requests nested in an epoch are not
  treated as independent evidence about between-epoch stability.
- Quantile uncertainty is estimated by a nonparametric bootstrap stratified by
  repeat or epoch when enough independent units exist. With fewer than three
  independent units, the report shows observations or their range and labels
  the result exploratory; it does not manufacture a confidence interval.
- A 95% interval quantifies sampling uncertainty under the stated design. It
  does not cover model-version drift, regional routing changes, unobserved
  backend changes, or future service behavior.
- No p99 claim is made without enough observations in the exact comparison cell
  to estimate the tail credibly. The hard publication minimum is 1,000 relevant
  independent sampling units.
- Buffered responses retain end-to-end latency but are censored from TTFT and
  post-TTFT curves. Multi-choice calls retain aggregate cost/goodput but are
  excluded from per-sequence curves.
- Cache hits, explicit cache misses, and missing cache counters are separate
  strata. A missing counter is never treated as a miss.

Every interval is accompanied by its numerator, denominator or independent-unit
count, method, and comparison cell. Multiple exploratory comparisons are not
promoted to discoveries solely because one interval excludes a null value.

## Claim statuses

| Status | Meaning |
|---|---|
| **Confirmed within protocol** | The named cell met its stated analysis rule and required repeats. The report must disclose whether that rule was fixed before outcomes were examined. This is not a service-level guarantee. |
| **Observed only** | A valid measurement exists, but repetition or another confirmation condition is missing. |
| **Inconclusive** | The experiment ran but cannot answer the question because of interruption, insufficient evidence, or incompatible observations. |
| **Unsupported in test** | The endpoint explicitly rejected the tested request or returned no exact success at the tested boundary. This is limited to that test. |
| **Not tested** | No valid attempt supports a result. |
| **Excluded** | A row exists but was removed from the estimand for a documented protocol or data-quality reason. |

An experiment is not called complete merely because requests were issued.
Publication completeness means every planned cell has an explained terminal
status and every claim uses only eligible cells.

The historical rule of three separated healthy AIMD epochs supports only the
phrase **met the benchmark's confirmation rule at _x_ requests/second**. It does
not by itself support **sustainable**, **production-ready**, or a recommended
production ceiling. The former 70% headroom value is an unvalidated heuristic
and is not a measured operating recommendation.

## Matched-control closure

An unresolved capability probe is retried as an endpoint-local sequence:

```text
known-good control before → exact capability probe → known-good control after
```

Only the semantic probe contributes capability evidence. Controls establish
that the same route was operational around the probe. An exact 400/413/422 can
support a route/API-specific rejection only when both controls pass. A 401/403,
402, timeout, 429, or 5xx response never proves that a model lacks the tested
capability. Repeated provider failures may become conclusive reliability
observations, but capability status remains inconclusive.

The late closure wave was stopped after inference calls returned 403 while the
prepaid balance was depleted. Its 52 physical attempts remain in cost and
reliability accounting, while its 12 terminal semantic cells add zero
capability claims. After the balance was replenished, the same credential again
returned HTTP 200 from `/v1/models`; the separate account-control API remained
403 and is not used as the inference-readiness gate.

## Attribution

The campaign had no version-matched model served by another provider. Therefore:

- HTTP status, timeout, client-observed timing, and returned content are direct
  observations of the tested endpoint path;
- wrong but valid answers are model-plus-serving outcomes, not automatically a
  DigitalOcean infrastructure fault;
- randomized within-endpoint comparisons can describe load-associated changes,
  but cannot identify every causal mechanism;
- the mixed-load experiment can report a measured paired difference, but two
  repeats cannot establish the absence of account-wide interference;
- absence of HTTP 402 during the window does not establish future account
  access or how startup credits were applied.

## AIMD and figure rules

Additive-increase/multiplicative-decrease (AIMD) raises offered traffic in steps
and cuts it after congestion. It is a probing algorithm, not a natural workload.

- A chronological diagnostic may connect epochs in time order.
- A plot of outcome against offered load must not connect raw observations.
  Backoff revisits earlier rates, so connecting the points creates loops and
  implies an unmeasured continuous response curve.
- Repeated offered rates are shown as individual marks or as a prespecified
  summary with an interval and `n`.
- Raw transport health and quality-gated health are never placed on one
  unqualified color scale.
- Missing, zero, rejected, interrupted, and unavailable values have distinct
  encodings.
- Conditional correctness and unconditional HTTP success are not drawn as peer
  bars with a shared denominator.
- Cost comparisons use a matched workload and a standardized unit. Total
  campaign spend is not used as an endpoint efficiency score.
- Figures use direct labels or small multiples, show units and sample sizes,
  pair color with shape or text, and remain legible without color.

## Exclusions and missingness

Exclusions are rule-based and recorded with a reason; they are never silent.
The first context sweep used a cross-model character-to-token calibration and
is not eligible for boundary claims. Corrected same-model calibration is the
eligible source for those claims. Interrupted cells, absent usage fields, and
failed requests remain visible in coverage and failure tables.

No failed request is silently replaced by a retry. If a retry occurs, both the
original attempt and retry identity must remain auditable.

## Costs and documentation freeze

Token-cost estimates use server-reported input/output tokens and the official
price snapshot frozen for the campaign. Timeout or no-response estimates must
be identified as conservative assumptions. These values are not DigitalOcean
billing exports and do not prove application of Hatch or other credits.

Catalog, price, and API documentation can change. A rerun with a later catalog,
price, endpoint build, or model identifier is a new campaign and must not be
merged into the August estimand without an explicit longitudinal design.
