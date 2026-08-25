# Reproduce and verify

This guide separates offline regeneration of the published evidence from a new
billable benchmark. Offline commands need no DigitalOcean key. Every live
runner creates a new campaign and must use a new output directory.

## Release status

The public artifacts are a **draft incomplete-evidence release**. The strict
matrix resolves 105 of 192 planned cells, so the final-publication gate fails by
design. Offline regeneration should reproduce that failure; it must not coerce
inconclusive cells into completed ones.

## Verify the package offline

Use Python 3.12:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python scripts/build-digitalocean-benchmark-manifest.py .
```

All `--plan-only` modes below must finish without loading a credential or
sending a request.

## Experiment entry points

The direct runners used for the 23–24 August closure are:

```text
scripts/run-digitalocean-direct-aimd.py
scripts/run-digitalocean-direct-soak.py
scripts/run-digitalocean-direct-capability.py
scripts/run-digitalocean-direct-context.py
scripts/run-digitalocean-direct-completion.py
scripts/run-digitalocean-matched-closure.py
```

Example no-send capacity plan:

```bash
python scripts/run-digitalocean-direct-aimd.py \
  --output-dir tmp/aimd-plan \
  --duration-minutes 60 \
  --max-cost-usd 200 \
  --prior-cost-usd 0 \
  --plan-only
```

Before a live run, inspect `--help` for that runner and set all duration,
cutoff, cumulative-cost, prior-exposure, concurrency, and timeout values
explicitly. `--prior-cost-usd` must include every earlier reservation and
settled request inside the same campaign cap.

## Offline analysis

The checked-in public bundle keeps the 105 MB canonical analysis JSON compressed
to stay below Git hosting limits. The PDF builder reads it directly:

```bash
python scripts/build-direct-public-report-pdf.py \
  --artifacts results \
  --output build/DigitalOcean-Inference-Endpoint-Benchmark-August-2026.pdf \
  --mode draft \
  --title "DigitalOcean Inference Endpoint Technical Benchmark" \
  --subtitle "Incomplete evidence report — 23–25 August 2026"
```

The decompressed `results/analysis.json` hash is recorded in
`results/analysis-manifest.json`. It is intentionally not committed in its
uncompressed form.

The canonical analyzer accepts one or more preserved directories for each
campaign family and writes normalized request/epoch tables, endpoint summaries,
coverage, confidence intervals, figure-ready tables, charts, a contract-gate
report, and an artifact manifest:

```bash
python scripts/analyze-direct-public-report.py \
  --breadth-dir /evidence/breadth \
  --aimd-dir /evidence/aimd \
  --soak-dir /evidence/soak \
  --completion-dir /evidence/completion \
  --closure-dir /evidence/matched-closure \
  --output-dir build/analysis \
  --bootstrap-replicates 2000 \
  --publication-mode draft
```

Paths above are placeholders. The private raw campaign journals are not
distributed in this public repository. Full request normalization and
re-analysis therefore require those separately preserved source directories;
repeat a flag to load multiple eligible campaigns. The analyzer verifies each
source identity and fails closed on incompatible or incomplete journals. The
public bundle itself supports checksum, schema, contract, table, and PDF
verification from its sanitized canonical analysis and derived tables, but it
does not pretend to reconstruct private source journals that are not shipped.

Build the concise engineering encyclopedia from a newly generated analysis:

```bash
python scripts/build-digitalocean-encyclopedia.py \
  --artifacts build/analysis \
  --output build/DigitalOcean-Inference-Engineering-Encyclopedia-August-2026.pdf
```

The older long-form builder remains available for forensic comparison, but it
is not the primary engineering handoff. The encyclopedia uses matched workload
small multiples, explicit missing/censored cells, a deterministic timing
outlier audit, and per-endpoint operating tables.

Build the legacy detailed PDF only when needed:

```bash
python scripts/build-direct-public-report-pdf.py \
  --artifacts build/analysis \
  --output build/DigitalOcean-Inference-Endpoint-Benchmark-August-2026.pdf \
  --mode draft \
  --title "DigitalOcean Inference Endpoint Technical Benchmark" \
  --subtitle "Incomplete evidence report — 23–24 August 2026"
```

`--mode final` is intentionally expected to fail until all 192 cells are
resolved by their exact evidence contracts. Do not edit the matrix status to
make that gate pass.

## Trace one result

For every table row or plotted point:

1. identify endpoint, workload, request settings, offered load, campaign, and
   repeat or epoch ID;
2. show included, excluded, failed, censored, and missing counts;
3. recompute the estimand and interval specified in [METHODS.md](METHODS.md);
4. trace the displayed value to request or epoch rows;
5. ensure unavailable values remain unavailable rather than becoming zero;
6. ensure every retry preserves the original failed attempt;
7. verify the same canonical value is used in CSV, chart, and PDF.

## Billable reruns

A rerun is a new study. Before sending traffic:

1. preserve a fresh model catalog and the current official documentation;
2. freeze exact model IDs, prices, client/dependency versions, seeds, UTC
   window, parameters, source commit, and region/account identity;
3. predeclare estimands, exclusions, confirmation rules, coverage target,
   time/cost caps, request timeout, and stop conditions;
4. run no-send planning, then a small labelled smoke;
5. keep the key only in the process environment;
6. journal every reservation, send, failure, retry, epoch, and settlement;
7. use a fresh output directory and never replay an existing request ID;
8. revoke the temporary credential after the campaign.

Never place a credential in a command argument, shell history, prompt, issue,
report, raw journal, or commit.
