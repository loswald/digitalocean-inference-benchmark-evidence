"""Dependency-free statistical primitives for the direct benchmark.

The module is deliberately small and auditable.  Requests are the sampling
unit for serial work, load epochs are the sampling unit for capacity work, and
task roots are the sampling unit for paired quality work.  Output tokens are
never resampled as though they were independent observations.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


STATISTICS_SCHEMA_VERSION = "digitalocean_direct_statistics_v1"
DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_REPLICATES = 10_000


class StatisticsError(ValueError):
    """Raised when an estimand has no valid statistical interpretation."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def deterministic_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def finite_values(values: Iterable[float | int | None]) -> list[float]:
    output: list[float] = []
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        parsed = float(value)
        if math.isfinite(parsed):
            output.append(parsed)
    return output


def nearest_rank(
    values: Iterable[float | int | None], probability: float
) -> float | None:
    if not 0 <= probability <= 1:
        raise StatisticsError("quantile probability must be in [0,1]")
    ordered = sorted(finite_values(values))
    if not ordered:
        return None
    if probability == 0:
        return ordered[0]
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def mean(values: Iterable[float | int | None]) -> float | None:
    rows = finite_values(values)
    return sum(rows) / len(rows) if rows else None


def _ci_bounds(confidence: float) -> tuple[float, float]:
    if not 0 < confidence < 1:
        raise StatisticsError("confidence must be in (0,1)")
    alpha = 1.0 - confidence
    return alpha / 2.0, 1.0 - alpha / 2.0


def percentile_interval(
    draws: Sequence[float], *, confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float | None, float | None]:
    lower, upper = _ci_bounds(confidence)
    return nearest_rank(draws, lower), nearest_rank(draws, upper)


def bootstrap_interval(
    values: Sequence[float | int],
    statistic: Callable[[Sequence[float]], float | None],
    *,
    seed: int,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    rows = finite_values(values)
    if not rows:
        return {
            "estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "n_units": 0,
            "bootstrap_replicates": 0,
            "qualified": False,
        }
    if replicates <= 0:
        raise StatisticsError("bootstrap replicates must be positive")
    rng = random.Random(int(seed))
    draws: list[float] = []
    for _ in range(replicates):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        value = statistic(sample)
        if value is not None and math.isfinite(float(value)):
            draws.append(float(value))
    low, high = percentile_interval(draws, confidence=confidence)
    estimate = statistic(rows)
    return {
        "estimate": None if estimate is None else float(estimate),
        "ci95_low": low,
        "ci95_high": high,
        "n_units": len(rows),
        "bootstrap_replicates": len(draws),
        "qualified": bool(draws),
    }


def cluster_bootstrap_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    cluster_key: str,
    value_key: str,
    statistic: Callable[[Sequence[float]], float | None],
    seed: int,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        parsed = float(value)
        if not math.isfinite(parsed):
            continue
        cluster = str(row.get(cluster_key, ""))
        if cluster:
            clusters[cluster].append(parsed)
    keys = sorted(clusters)
    flattened = [value for key in keys for value in clusters[key]]
    estimate = statistic(flattened)
    if not keys:
        return {
            "estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "n_clusters": 0,
            "n_observations": 0,
            "bootstrap_replicates": 0,
            "qualified": False,
        }
    rng = random.Random(int(seed))
    draws: list[float] = []
    for _ in range(replicates):
        sample: list[float] = []
        for _ in keys:
            sample.extend(clusters[keys[rng.randrange(len(keys))]])
        value = statistic(sample)
        if value is not None and math.isfinite(float(value)):
            draws.append(float(value))
    low, high = percentile_interval(draws, confidence=confidence)
    return {
        "estimate": None if estimate is None else float(estimate),
        "ci95_low": low,
        "ci95_high": high,
        "n_clusters": len(keys),
        "n_observations": len(flattened),
        "bootstrap_replicates": len(draws),
        "qualified": bool(draws) and len(keys) >= 2,
    }


def fixed_day_block_bootstrap_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    day_key: str = "day_id",
    block_key: str = "time_block_id",
    statistic: Callable[[Sequence[float]], float | None] = mean,
    seed: int,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Resample time blocks within each observed day; never resample three days.

    The campaign days are fixed design strata, not a plausible random sample of
    calendar days.  This interval therefore measures within-campaign block
    variability and cannot support a seasonal or universal diurnal claim.
    """

    days: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = row.get(value_key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        parsed = float(value)
        if not math.isfinite(parsed):
            continue
        day = str(row.get(day_key, ""))
        block = str(row.get(block_key, ""))
        if day and block:
            days[day][block].append(parsed)
    observed = [
        value
        for blocks in days.values()
        for values in blocks.values()
        for value in values
    ]
    estimate = statistic(observed)
    block_count = sum(len(blocks) for blocks in days.values())
    if not days or block_count < 2:
        return {
            "estimate": None if estimate is None else float(estimate),
            "ci95_low": None,
            "ci95_high": None,
            "fixed_day_count": len(days),
            "n_time_blocks": block_count,
            "n_observations": len(observed),
            "bootstrap_replicates": 0,
            "qualified": False,
            "claim_scope": "within_campaign_fixed_days",
        }
    rng = random.Random(int(seed))
    draws: list[float] = []
    for _ in range(replicates):
        sample: list[float] = []
        for day in sorted(days):
            block_ids = sorted(days[day])
            for _ in block_ids:
                sample.extend(days[day][block_ids[rng.randrange(len(block_ids))]])
        value = statistic(sample)
        if value is not None and math.isfinite(float(value)):
            draws.append(float(value))
    low, high = percentile_interval(draws, confidence=confidence)
    return {
        "estimate": None if estimate is None else float(estimate),
        "ci95_low": low,
        "ci95_high": high,
        "fixed_day_count": len(days),
        "n_time_blocks": block_count,
        "n_observations": len(observed),
        "bootstrap_replicates": len(draws),
        "qualified": bool(draws),
        "claim_scope": "within_campaign_fixed_days",
    }


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    if successes < 0 or total < 0 or successes > total:
        raise StatisticsError("invalid binomial counts")
    if total == 0:
        return {"estimate": None, "ci95_low": None, "ci95_high": None, "n": 0}
    # The report contract freezes 95% intervals. Avoid a large inverse-normal
    # implementation and fail closed if a caller silently changes confidence.
    if not math.isclose(confidence, 0.95, rel_tol=0, abs_tol=1e-12):
        raise StatisticsError("Wilson implementation is frozen to 95% confidence")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return {
        "estimate": p,
        "ci95_low": max(0.0, center - half),
        "ci95_high": min(1.0, center + half),
        "n": total,
    }


def mcnemar_exact(
    discordant_low_only: int, discordant_high_only: int
) -> dict[str, Any]:
    """Two-sided exact McNemar test, conditioning on discordant pairs."""

    if discordant_low_only < 0 or discordant_high_only < 0:
        raise StatisticsError("discordant counts must be non-negative")
    n = discordant_low_only + discordant_high_only
    if n == 0:
        return {
            "p_value_two_sided": 1.0,
            "discordant_pairs": 0,
            "low_only_success": discordant_low_only,
            "high_only_success": discordant_high_only,
        }
    tail = min(discordant_low_only, discordant_high_only)
    probability = sum(math.comb(n, index) for index in range(tail + 1)) / (2**n)
    return {
        "p_value_two_sided": min(1.0, 2.0 * probability),
        "discordant_pairs": n,
        "low_only_success": discordant_low_only,
        "high_only_success": discordant_high_only,
    }


def paired_root_analysis(
    pairs: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence: float = DEFAULT_CONFIDENCE,
    binary_threshold: float = 0.5,
) -> dict[str, Any]:
    roots: list[str] = []
    differences: list[float] = []
    low_only = 0
    high_only = 0
    for row in pairs:
        low = row.get("low_load_score")
        high = row.get("near_saturation_score")
        root = str(row.get("root_id", ""))
        if (
            not root
            or isinstance(low, bool)
            or isinstance(high, bool)
            or not isinstance(low, (int, float))
            or not isinstance(high, (int, float))
            or not math.isfinite(float(low))
            or not math.isfinite(float(high))
        ):
            continue
        low_value, high_value = float(low), float(high)
        roots.append(root)
        differences.append(high_value - low_value)
        low_success = low_value >= binary_threshold
        high_success = high_value >= binary_threshold
        low_only += int(low_success and not high_success)
        high_only += int(high_success and not low_success)
    interval = bootstrap_interval(
        differences,
        mean,
        seed=seed,
        replicates=replicates,
        confidence=confidence,
    )
    return {
        "paired_root_count": len(roots),
        "mean_near_minus_low": interval,
        "mcnemar_exact": mcnemar_exact(low_only, high_only),
        "pairing_unit": "task_root",
        "qualified": len(roots) >= 2,
    }


@dataclass(frozen=True)
class IsotonicPoint:
    offered_rps: float
    observations: int
    raw_congestion_rate: float
    fitted_congestion_rate: float


def _pava(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    if len(values) != len(weights) or not values:
        raise StatisticsError("PAVA requires equally sized non-empty vectors")
    blocks: list[list[float]] = []  # [start, end, weight, weighted value]
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        if weight <= 0 or not math.isfinite(value) or not math.isfinite(weight):
            raise StatisticsError("PAVA values and weights must be finite and weighted")
        blocks.append(
            [float(index), float(index), float(weight), float(value * weight)]
        )
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[3] / left[2] <= right[3] / right[2] + 1e-15:
                break
            blocks[-2:] = [[left[0], right[1], left[2] + right[2], left[3] + right[3]]]
    fitted = [0.0] * len(values)
    for start, end, weight, weighted in blocks:
        value = weighted / weight
        for index in range(int(start), int(end) + 1):
            fitted[index] = value
    return fitted


def isotonic_congestion_curve(
    epoch_rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.01,
    seed: int | None = None,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Fit monotone congestion and optionally bootstrap its tested-grid knee.

    Input rows are science epochs.  Bootstrap draws resample whole
    ``science_epoch_id`` clusters, never the requests inside an epoch.  The
    resulting interval is conditional on the offered-rate grid actually tested.
    """

    if not 0 <= threshold <= 1:
        raise StatisticsError("congestion threshold must be in [0,1]")

    def fit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        by_rate: dict[float, list[tuple[int, int]]] = defaultdict(list)
        for row in rows:
            rate = row.get("offered_rps")
            congested = row.get("congestion_count")
            total = row.get("terminal_count")
            if (
                isinstance(rate, bool)
                or isinstance(congested, bool)
                or isinstance(total, bool)
                or not isinstance(rate, (int, float))
                or not isinstance(congested, int)
                or not isinstance(total, int)
                or rate <= 0
                or total <= 0
                or not 0 <= congested <= total
            ):
                continue
            by_rate[float(rate)].append((congested, total))
        rates = sorted(by_rate)
        raw: list[float] = []
        weights: list[float] = []
        counts: list[int] = []
        for rate in rates:
            congestion = sum(value[0] for value in by_rate[rate])
            total = sum(value[1] for value in by_rate[rate])
            raw.append(congestion / total)
            weights.append(float(total))
            counts.append(total)
        if not rates:
            return {"points": [], "knee_bracket": None, "qualified": False}
        fitted = _pava(raw, weights)
        points = [
            IsotonicPoint(rate, count, observed, predicted).__dict__
            for rate, count, observed, predicted in zip(
                rates, counts, raw, fitted, strict=True
            )
        ]
        healthy_rates = [
            rate
            for rate, value in zip(rates, fitted, strict=True)
            if value <= threshold
        ]
        unhealthy_rates = [
            rate for rate, value in zip(rates, fitted, strict=True) if value > threshold
        ]
        return {
            "points": points,
            "knee_bracket": {
                "highest_fitted_at_or_below_threshold_rps": (
                    max(healthy_rates) if healthy_rates else None
                ),
                "lowest_fitted_above_threshold_rps": (
                    min(unhealthy_rates) if unhealthy_rates else None
                ),
                "congestion_threshold": threshold,
            },
            "qualified": len(rates) >= 2,
            "fit": "weighted_non_decreasing_PAVA",
        }

    observed_fit = fit(epoch_rows)
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(epoch_rows):
        cluster_id = str(row.get("science_epoch_id") or f"row-{index}")
        clusters[cluster_id].append(row)
    cluster_ids = sorted(clusters)
    if seed is None:
        observed_fit["knee_epoch_cluster_bootstrap"] = {
            "sampling_unit": "science_epoch_id",
            "n_clusters": len(cluster_ids),
            "bootstrap_replicates": 0,
            "qualified": False,
            "reason": "seed_not_supplied",
        }
        return observed_fit
    if replicates <= 0:
        raise StatisticsError("bootstrap replicates must be positive")

    rng = random.Random(int(seed))
    healthy_draws: list[float] = []
    unhealthy_draws: list[float] = []
    joint_draws = 0
    for _ in range(replicates if cluster_ids else 0):
        sample: list[Mapping[str, Any]] = []
        for _ in cluster_ids:
            sample.extend(clusters[cluster_ids[rng.randrange(len(cluster_ids))]])
        bracket = fit(sample).get("knee_bracket") or {}
        healthy = bracket.get("highest_fitted_at_or_below_threshold_rps")
        unhealthy = bracket.get("lowest_fitted_above_threshold_rps")
        healthy_ok = isinstance(healthy, (int, float)) and not isinstance(healthy, bool)
        unhealthy_ok = isinstance(unhealthy, (int, float)) and not isinstance(
            unhealthy, bool
        )
        if healthy_ok:
            healthy_draws.append(float(healthy))
        if unhealthy_ok:
            unhealthy_draws.append(float(unhealthy))
        joint_draws += int(healthy_ok and unhealthy_ok)

    observed_bracket = observed_fit.get("knee_bracket") or {}

    def endpoint_interval(key: str, draws: Sequence[float]) -> dict[str, Any]:
        low, high = percentile_interval(draws, confidence=confidence)
        estimate = observed_bracket.get(key)
        return {
            "estimate": (
                float(estimate) if isinstance(estimate, (int, float)) else None
            ),
            "ci95_low": low,
            "ci95_high": high,
            "finite_draws": len(draws),
            "qualified": bool(
                len(cluster_ids) >= 2 and len(draws) >= math.ceil(0.80 * replicates)
            ),
        }

    observed_fit["knee_epoch_cluster_bootstrap"] = {
        "sampling_unit": "science_epoch_id",
        "n_clusters": len(cluster_ids),
        "bootstrap_replicates": replicates if cluster_ids else 0,
        "joint_finite_bracket_draws": joint_draws,
        "highest_fitted_at_or_below_threshold_rps": endpoint_interval(
            "highest_fitted_at_or_below_threshold_rps", healthy_draws
        ),
        "lowest_fitted_above_threshold_rps": endpoint_interval(
            "lowest_fitted_above_threshold_rps", unhealthy_draws
        ),
        "qualified": bool(
            observed_fit.get("qualified")
            and len(cluster_ids) >= 2
            and joint_draws >= math.ceil(0.80 * replicates)
        ),
        "qualification": "conditional_on_observed_offered_rate_grid",
    }
    return observed_fit


def qualified_quantiles(
    values: Sequence[float | int],
    *,
    seed: int,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    rows = finite_values(values)
    output: dict[str, Any] = {"n": len(rows)}
    for label, probability in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95)):
        output[label] = bootstrap_interval(
            rows,
            lambda sample, probability=probability: nearest_rank(sample, probability),
            seed=deterministic_seed(seed, label),
            replicates=replicates,
            confidence=confidence,
        )
    if len(rows) >= 1_000:
        output["p99"] = bootstrap_interval(
            rows,
            lambda sample: nearest_rank(sample, 0.99),
            seed=deterministic_seed(seed, "p99"),
            replicates=replicates,
            confidence=confidence,
        )
        output["p99"]["qualification"] = "qualified_n_at_least_1000"
    else:
        output["p99"] = {
            "estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "n_units": len(rows),
            "bootstrap_replicates": 0,
            "qualified": False,
            "qualification": "suppressed_n_below_1000",
        }
    return output


__all__ = [
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_CONFIDENCE",
    "STATISTICS_SCHEMA_VERSION",
    "StatisticsError",
    "bootstrap_interval",
    "canonical_json",
    "cluster_bootstrap_interval",
    "deterministic_seed",
    "fixed_day_block_bootstrap_interval",
    "isotonic_congestion_curve",
    "mcnemar_exact",
    "mean",
    "nearest_rank",
    "paired_root_analysis",
    "percentile_interval",
    "qualified_quantiles",
    "wilson_interval",
]
