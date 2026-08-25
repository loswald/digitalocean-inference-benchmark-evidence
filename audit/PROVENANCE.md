# Provenance

## Source identity

The evidence was produced by operator-run measurements of DigitalOcean
Serverless Inference, with direct campaigns executed on 23–25 August 2026.
Requests used the public `https://inference.do-ai.run/v1` namespace and exact
identifiers returned by the live `/v1/models` catalog. The current production
scope is the 11 identifiers in DigitalOcean's documented hosted-model table.
One historical identifier, `arcee-trinity-large-thinking`, is a partner model;
its rows are retained only for reconciliation and excluded from production
comparisons and recommendations.

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
| Interrupted matched closure | 2026-08-24 23:16–23:19 | Sealed inconclusive; no replay |
| Hosted recovery controls | 2026-08-25 00:12–00:24 | Complete; access restored |
| Hosted-only closure | 2026-08-25 00:25–00:53 | 177/177 terminal; 51 conclusive |
| Hosted-only closure v2 | 2026-08-25 02:22–02:44 | Complete; matched controls |
| Hosted context closure | 2026-08-25 03:44–04:21 | 199/199 terminal |
| Final matched closure | 2026-08-25 04:31–04:40 | 57/57 semantic records; 196 attempts |
| Targeted Gemma context | 2026-08-25 04:42–04:50 | Complete; nonmonotonic retrieval retained |
| Fixed Nemotron Super soak | 2026-08-25 04:55–04:57 | 139/139 transport success; quality recovery failed |

The final conservative exposure was `$329.972391327` under a `$400` cap. One
historical context stage latched HTTP 402; later stages preserve that fact but
do not attribute it to model quality. The public analysis contains 42,864
normalized request/attempt rows and 1,435 load epochs.

The immutable partner-inclusive historical matrix is retained only for forensic
scope reconciliation. The hosted-only publication view is 176 cells: 169
completed and 7 unsupported. Publication status is `publication_ready`.

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
