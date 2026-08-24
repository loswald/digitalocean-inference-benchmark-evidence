# Glossary

Terms here describe this benchmark, not universal provider guarantees.

**Accepted request.**
A request for which the API returned the protocol's accepted success status.
Acceptance does not mean the answer was correct.

**Additive-increase/multiplicative-decrease (AIMD).**
A load-probing method that raises traffic gradually and cuts it sharply after
congestion. Its back-and-forth path is why raw AIMD points must not be connected
on an offered-load response plot.

**Attempt.**
One client request, including failures and timeouts. Retries are new attempts.

**Cell.**
One exact combination of endpoint, workload, request settings, load, campaign
segment, and repeat identity.

**Confidence interval (CI).**
A range produced by a stated statistical procedure. A 95% CI describes sampling
uncertainty under that procedure; it is not a 95% guarantee about future service.

**Context lower bound.**
The largest tested prompt that passed a stated rule. It is not the endpoint's
true maximum unless a valid pass/fail boundary was localized.

**Aggregate output goodput.**
Successful server-reported completion tokens divided by the complete measured
wall-clock interval. This is the report's headline output-throughput measure.
It includes reasoning tokens when the provider bills and reports them.

**Post-TTFT service-output proxy.**
Server-reported completion tokens divided by `(request end - streamed TTFT)`.
It includes network and buffering and is not direct decoder speed.

**Legacy SSE chunk-span proxy.**
Completion tokens divided by the time from the first to last streamed content
event. One event may contain many tokens, so this quantity is audit-only and is
never presented as decoder throughput.

**Cache stratum.**
Observed cache hit, observed cache miss, or cache state not reported. Missing
cache counters are not treated as zero. TTFT is not pooled across strata.

**End-to-end latency.**
Client-observed time from starting a request until it completes.

**End-to-end task success.**
The share of all attempts that both returned successfully and passed the local
task check.

**Epoch.**
A bounded measurement interval at one offered load. Requests in the same epoch
may share service conditions and are not independent stability repeats.

**Exact pass.**
A response that satisfies the task's deterministic checker, such as the exact
value, JSON structure, tool arguments, executable result, or planted nonce.

**Goodput.**
Useful completed work per unit time. In this report, quality-adjusted goodput is
correct responses divided by observed time.

**HTTP success rate.**
Successful HTTP responses divided by all attempted requests in the named cell.
It measures transport/API completion, not answer quality.

**Independent repeat.**
A separately identified repetition that can support between-run variation.
Several requests inside one epoch are not independent repeats of service state.

**Observed only.**
A real measurement that did not meet the benchmark's confirmation conditions.

**Offered rate.**
Requests the client attempted to start per second. It is an input to the load
test, not the rate the service completed.

**Quality-adjusted goodput.**
Correct responses completed per unit time. It combines completion and task
correctness for tasks with a valid checker.

**Raw transport health.**
A load-test rule based on response completion, errors, and latency, without a
task-correctness requirement. It must not be described as useful-output capacity.

**Recovery.**
Behavior measured after a stress or congestion phase. One healthy recovery
epoch is an observation, not proof of long-term resilience.

**Requests per minute/second (RPM/RPS).**
Counts of requests per unit time. Labels must say whether the rate was offered,
successful, or correct.

**Server-reported usage.**
Input/output token counts returned by the API. If absent, token throughput and
token-priced cost are unavailable rather than zero.

**Sustainable.**
A durability claim requiring evidence across relevant time, repetitions, and
conditions. The August campaign's three-epoch confirmation rule is not enough
on its own to justify this word.

**Task pass rate.**
The share of responses satisfying the stated checker. The report must say
whether its denominator is successful scorable responses or all attempts.

**Time to first token (TTFT).**
Client-observed time from request start until the first streamed content token.
It includes network and queueing time and is not direct server prefill speed.
Buffered responses expose full-response latency, not token-level TTFT, so their
response-arrival timestamps are censored from TTFT curves.

**Token throughput (TPM/TPS).**
Server-reported tokens divided by an explicit wall-clock denominator. The report
states whether the denominator is a request, AIMD epoch, or 30-second soak block.
Input and output throughput are different quantities and are reported separately.

**Transport support / functional correctness / malformed validation.**
Three separate capability outcomes: whether valid calls complete; whether 2xx
responses solve the task; and whether deliberately invalid inputs are rejected.
A correct 4xx for malformed input does not show that valid calls are unsupported.

**Wilson interval.**
A binomial confidence interval used for proportions. It still depends on a
valid denominator and a defensible independence assumption.
