# Measurement glossary

This glossary defines the terms used in the benchmark report. The short
explanations are intentionally plain, while the technical definitions state
exactly what is calculated.

## Load and capacity

**Offered requests per minute (offered RPM).** How many request arrivals the
load generator attempted to schedule per minute. This is demand, not completed
work. Arrivals are scheduled from an external clock, so a slow endpoint does
not make the offered rate silently fall.

**Accepted RPM.** Requests per minute for which a send was durably committed
and the server returned a known HTTP response. It includes known HTTP errors.

**Successful RPM.** Requests per campaign-wall minute that returned a complete
successful service response and passed the server-usage evidence gate. A
separate goodput figure additionally requires the deterministic task check to
pass.

**Concurrency.** The number of requests outstanding at once. Concurrency is a
ceiling on in-flight work, not a synonym for request rate. The benchmark varies
offered rate and concurrency independently.

**Open-loop load.** Request arrivals are scheduled from an external clock and
do not wait for earlier responses. This exposes queues and overload. A
closed-loop test, by contrast, can appear healthy merely because slow replies
prevent the client from offering more work.

**Coordinated omission.** A measurement bias in which periods of server delay
also suppress new test requests, thereby omitting the requests that would have
waited in a real arrival stream. The open-loop scheduler records scheduled,
arrival, admission, and completion times to avoid this bias.

**AIMD (additive increase, multiplicative decrease).** A search controller that
raises offered rate in fixed steps while epochs are healthy and cuts the rate,
normally in half, after congestion. It is used to bracket capacity rather than
to assume a rate in advance.

**Healthy epoch.** An independently scheduled load block that passes the frozen
success, error, rate-limit, queue-growth, latency, usage, and task-integrity
rules. The exact rule is published with each campaign.

**Capacity knee.** The transition region where more offered work stops
producing proportional successful work or begins to cause material queueing,
latency inflation, throttling, or failure. It is reported as a confidence
interval or healthy/unhealthy bracket, not a falsely exact number.

**Saturation.** Two consecutive adequately populated epochs that breach the
frozen health rule. A sparse or interrupted epoch is inconclusive, not proof of
saturation.

**AIMD-confirmed healthy offered rate.** The highest tested rate that passed
three separated short confirmation epochs. It is a workload-specific measured
point or lower bound, not sustained capacity, a ceiling, or an SLA. A
short-text result must not be reused for vision, tools, long context, or long
decode.

**Smallest nonbinding concurrency at the tested rate.** The lowest tested
in-flight ceiling for which both replicated epochs and every higher tested
ceiling remain healthy at a fixed sub-knee rate. It answers “how much client
concurrency is enough at this rate”; it is not a universal maximum concurrent
request limit.

**Production headroom.** An operator-selected margin below a workload-matched
soak-verified rate. This benchmark does not manufacture a recommended margin;
choose it from the production SLO, variability, and risk tolerance.

## Tokens and throughput

**Requested input/output tokens.** The target used to construct a request and
the output ceiling sent to the API. Requested values are design inputs, not
proof of what the server processed or generated.

**Realized input/output tokens.** Token counts reported by the server for the
completed request. These are the authoritative x-axis and billing basis in the
report. Missing or inconsistent usage makes a token-throughput observation
inconclusive.

**Offered input/output TPM.** Sum of the conservatively reserved input or output
tokens attached to scheduled arrivals, divided by the offered-window minutes.
It represents potential demand and is not claimed as completed work.

**Accepted input/output TPM.** Server-reported tokens attached to requests with
a known HTTP response, divided by the stated time denominator. This may include
responses that fail the task check.

**Effective input/output TPM.** Server-reported tokens from scientifically
successful requests divided by campaign-wall minutes. Campaign-wall time is
used for the headline capacity figure so queues, errors, and idle drain time do
not disappear. Active-exposure and offered-window rates are also shown and are
labelled separately.

**Aggregate output tokens per second.** Total successfully generated output
tokens divided by the elapsed wall time of the measured epoch. This is the
capacity figure engineers use for fleet-level work.

**Per-request post-TTFT service-output proxy.** A request's server-reported
billed completion tokens divided by `(request end - streamed TTFT)`. It is an
end-to-end delivery proxy that includes network and buffering, not direct model
decoder speed. Intervals shorter than 100 ms are timing-unstable and are
explicitly censored from this per-request rate while their billed tokens remain
in aggregate goodput. Multi-choice responses are excluded from per-sequence
curves because usage aggregates all choices.

**Legacy SSE chunk-span proxy.** Completion tokens divided by the interval from
the first to last semantic SSE event. SSE events are transport chunks and may
batch many tokens, so this value is retained only in the metric-audit table and
never used as a decode-rate claim.

**Seconds per output token.** The reciprocal of per-request output tokens per
second where the measurement is defined. It is useful for slow decodes but is
not the same as inter-token latency.

## Latency

**Time to first token (TTFT).** For a streaming response, elapsed time from the
scheduled request arrival to the first semantic output token. Client queue and
admission time are retained separately. TTFT is undefined for a buffered
non-streaming response; the report uses time to response headers/first byte and
total latency there instead.

**End-to-end latency.** Elapsed time from scheduled request arrival to the
terminal response, including client admission delay, connection/network time,
server queueing, prefill, decode, and stream delivery.

**Inter-token latency (ITL).** Time between adjacent semantic output-token
events in a streaming response. It is summarized per request, then across
requests. ITL is undefined for non-streaming responses.

**End-to-end prefill proxy.** Streaming TTFT after subtracting measured client
queue/admission time where possible. It still includes network and server
queueing and therefore is not labelled direct server-side prompt-processing
speed unless DigitalOcean exposes a server timing for that stage. It is reported
only for an explicit cache miss; observed hits and missing cache counters form
separate strata.

**Recovery time.** Time from an intentional overload epoch ending until the
endpoint again passes the frozen health rule at 50% of the AIMD-confirmed
healthy candidate rate.

## Correctness and usable work

**Transport success.** A known, complete successful HTTP/service response. It
does not imply that the answer is correct.

**Scientific success.** Transport success plus valid server usage and every
measurement-specific integrity gate. Examples include correct long-context
retrieval, a valid hash-chain prefix for forced decode, or exact tool arguments.

**Goodput.** Scientifically successful requests or tokens per campaign-wall
time. Failures, malformed results, and unusable short generations remain in the
denominator through elapsed time.

**Quality-adjusted goodput.** Successful token or request throughput multiplied
by deterministic task credit, or equivalently the rate of correct usable work
for binary tasks. Model/task failures are reported separately from service-path
failures; without a same-model external serving control they are not labelled
DigitalOcean infrastructure failures.

**Context acceptance.** The API accepted and completed a prompt of a given
realized input length. This does not prove that the model used information near
the end or middle of the prompt.

**Context utilization.** The response passes fresh retrieval and checksum
questions placed at multiple positions in the long prompt. Acceptance and
utilization boundaries are reported separately.

**Requested output limit accepted.** The API accepted the output-cap parameter.
This is distinct from the model actually producing that many tokens.

**Realized output length.** Server-reported output tokens generated before EOS,
the requested cap, a combined-context restriction, timeout, cancellation, or
another terminal condition. The stopping reason and deterministic sequence
integrity are reported with the length.

## Uncertainty and comparisons

**95% confidence interval (CI).** An interval produced by the preregistered
sampling model that would cover the target quantity in about 95% of repeated
experiments under that model. It is measurement uncertainty, not a guarantee
that future production values stay inside the interval.

**Sampling unit.** The independently resampled unit. Requests are used for
serial latency; whole load epochs for capacity; paired task roots with their
repeats linked for quality; and time blocks within each fixed observed day for
temporal variation. Individual output tokens are never treated as independent
replicates.

**Bootstrap interval.** A CI formed by repeatedly resampling the appropriate
independent units and recalculating the statistic. Fixed observed campaign days
are not resampled as if seven days represented all possible days.

**Wilson interval.** A binomial proportion interval used for success and error
rates because it behaves better than the simple normal approximation near zero
or one.

**Exact McNemar test.** A paired binary comparison based only on roots that
change outcome between low load and near saturation. It tests whether
regressions and improvements are asymmetric while preserving root pairing.

**p50, p90, p95, p99.** The median, 90th, 95th, and 99th percentiles. Tail
percentiles require many observations. p99 is suppressed unless at least 1,000
relevant independent observations support it; sparse p95 estimates are marked
exploratory.

**Observed versus documented limit.** The documented value is DigitalOcean's
published contract. The observed value is the boundary supported by this
campaign's exact model, region, API version, account, request shape, and time.
Neither is silently substituted for the other.

**Intra-day variation.** Differences observed across the frozen two-hour blocks
in the seven fixed campaign days. It is not called seasonality, a universal
diurnal pattern, or an SLA.

**Cost per million effective tokens.** Settled or conservatively exposed cost
divided by scientifically successful realized tokens, scaled to one million.
Failed, partial, unknown, and retried sends remain in the cost ledger.
