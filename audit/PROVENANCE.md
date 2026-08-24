# Provenance

## Source identity

The evidence was produced by operator-run measurements of DigitalOcean
Serverless Inference, with the final direct campaign executed on 23–24 August
2026. Requests used the public `https://inference.do-ai.run/v1` namespace and
exact identifiers returned by the live `/v1/models` catalog with
`owned_by=digitalocean`.

No customer data was used. Prompts were programmatic or synthetic. The public
bundle contains sanitized request-level timing, status, usage, cost, scoring,
and experiment identity—not credentials, prompts, response text, raw bodies, or
raw headers.

## Campaign lineage

| Stage | UTC interval | Terminal status |
|---|---|---|
| Original direct AIMD | 2026-08-23 14:33–17:31 | Complete, right-censored |
| Two-minute soak | 2026-08-23 18:31–21:24 | Execution complete, science incomplete |
| Capability envelope | 2026-08-23 21:41–21:47 | Terminal coverage complete |
| Context envelope | 2026-08-23 21:53–22:08 | Execution complete, science incomplete |
| Fresh endpoint-isolated AIMD | 2026-08-24 04:36–06:48 | Complete, right-censored |
| Targeted completion | 2026-08-24 06:58–10:31 | Incomplete or censored |

The final conservative exposure was `$237.358277971` under a `$400` cap. One
historical context stage latched HTTP 402; later stages preserve that fact but
do not attribute it to model quality. The public analysis contains 41,595
request rows and 1,453 epoch rows.

The strict evidence matrix is 192 cells. It resolves 105 cells and leaves 87
inconclusive, so the publication status is `draft_incomplete_coverage`.

## Transformation lineage

```text
catalog + immutable plan + request/epoch/reservation journals
    -> source-identity and reconciliation checks
    -> normalized public request and epoch tables
    -> dimension-specific eligibility and coverage rules
    -> matched estimand tables and 95% intervals
    -> figure-ready data, figures, and PDF
    -> public safety scan and checksum manifests
```

Report prose, plots, and PDFs are terminal derivatives. They are never read
back as evidence or used to repair missing values. Missing, unsupported,
censored, and inconclusive are distinct states.

## Documentation and pricing provenance

The final documentation freeze used DigitalOcean's official model, pricing,
limit, multimodal, and quota-header pages as verified on 24 August 2026. Token
cost is calculated from server-reported usage and frozen prices. It is an
estimate, not a billing export, and does not establish how Hatch or other
credits were applied.

## Sanitization and disclosure

The final public safety scan reported zero findings across its allow-listed
bundle. Provider credentials and authorization headers were never published.
The repository is operator-run and is not affiliated with or endorsed by
DigitalOcean. Any DigitalOcean review after publication does not retroactively
change the evidence or make the study sponsored.
