# Publication sanitization

The public evidence preserves request IDs, cell IDs, timestamps, status codes,
sanitized error classes, timing, server-reported usage, task scores, and cost
fields. It excludes prompt text, model response text, raw response bodies, raw
headers, and credentials.

The publication transformations are:

- Authorization headers and credential values were never recorded by the
  direct harness.
- Input journals were normalized through an allow-list. Content fields were
  dropped; deterministic payload, response, task, scorer, and request hashes
  retain traceability without publishing the content.
- Internal backup locations, machine names, credentials, account identifiers,
  and unrelated workspace paths are excluded.

These transformations do not change reported HTTP status, latency, throughput,
quality-pass, token, or cost values. The final public safety scan records zero
findings in the canonical analysis bundle.
