# Account-access incident — 24–25 August 2026

This note separates a late credential/account failure from model and endpoint results.

## Observed facts

- The only authorized DigitalOcean OAuth token was the same 71-character `doo_v1_…` credential
  sent by the account owner on 14 August 2026 and already used by the successful campaigns.
- A matched-control closure wave began after the main benchmark. Its 52 durable provider attempts
  returned 32 HTTP 403, 4 HTTP 500, and 16 HTTP 503 responses.
- The runner was stopped. It did not replay any request ID. Twelve semantic cells were sealed as
  inconclusive, and zero were promoted to supported or rejected capability evidence.
- A later serial check of `https://inference.do-ai.run/v1/models` returned HTTP 403 while
  the Serverless Inference prepaid balance was depleted.
- After the account owner replenished the balance on 25 August, the same inference
  credential returned HTTP 200 from `/v1/models`.
- The credential still returned HTTP 403 from `https://api.digitalocean.com/v2/account`.
  The control-plane API is a separate surface and is not treated as the inference-readiness
  gate.
- The cumulative conservative campaign exposure after the hosted-only recovery and closure
  wave is `$240.825554971`, below the owner-approved `$400` cap. No HTTP 402 latch occurred
  in those recovery/closure runs.

No credential, prompt, model output, response body, or raw header is retained in this public note.

## Interpretation

DigitalOcean documents the inference and control-plane APIs as separate base URLs. The recovery of
`/v1/models` immediately after prepayment is consistent with DigitalOcean's documented balance-
depletion suspension. It is not evidence that a parameter, vision payload, tool schema, context
size, or model is unsupported.

The audit also found that `arcee-trinity-large-thinking` is documented in a separate Arcee
partner-model section rather than the DigitalOcean-Hosted Models table. Its 3,210 historical rows
carry approximately `$9.486053` of token-attributed usage. The public request ledger cannot prove
that this was the exact balance line item that caused depletion, but it is the identified
passthrough exposure and the most plausible reason promotional credits did not absorb all charges.

The successful evidence collected before the incident remains valid for its measured window. It
does not prove current availability.

## Recovery gate and completion

The following gate was applied before the hosted-only closure wave:

1. The Serverless Inference model-list endpoint succeeds.
2. Two serial streamed exact-marker controls on DigitalOcean-hosted models succeed with positive
   prompt and completion usage.
3. The remaining matched-control plan resumes with its existing deterministic identities and a new
   output directory; the sealed incident IDs are never replayed.
4. Every selected model passes the hosted-only allowlist; Arcee and every other partner/passthrough
   model fail closed before a request is sent.

All four gates passed. The subsequent hosted-only campaign sent 521 provider attempts across
177/177 terminal cells, plus 9 serial recovery/control attempts. It observed 378 HTTP 200, 105
expected validation HTTP 400, and 30 HTTP 500 responses in the main closure wave, with no HTTP
402 and no Arcee selection. Fifty-one cells became conclusive; unresolved cells remain labelled
inconclusive rather than being converted into support or rejection claims.
