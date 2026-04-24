"""Data sanity checks: calibration fail-fast, split verification, sample logging."""

import logging
from typing import List

logger = logging.getLogger(__name__)

MIN_CALIB_SAMPLES = 50


def check_calibration_data(
    texts: List[str],
    domain: str,
    min_samples: int = MIN_CALIB_SAMPLES,
    source: str = "unknown",
) -> None:
    """Fail-fast if calibration data is insufficient.

    Raises ValueError if fewer than min_samples valid texts are available.
    Prevents FLC/CARR calibration from silently using empty or near-empty data.
    """
    valid = [t for t in texts if t and len(t.strip()) > 10]
    if len(valid) < min_samples:
        raise ValueError(
            f"Calibration data insufficient for domain '{domain}' from {source}: "
            f"got {len(valid)} valid texts (need >= {min_samples}). "
            f"Total provided: {len(texts)}, empty/short: {len(texts) - len(valid)}. "
            f"Do NOT synthesize fallback examples — fix the data source."
        )
    logger.info("Calibration check PASS: %s has %d valid texts (min=%d)", domain, len(valid), min_samples)
    return valid


def check_no_overlap(
    train_ids: set,
    eval_ids: set,
    domain: str,
) -> None:
    """Fail if any sample ID appears in both train and eval sets."""
    overlap = train_ids & eval_ids
    if overlap:
        raise ValueError(
            f"Train/eval overlap detected for domain '{domain}': "
            f"{len(overlap)} shared samples. IDs: {list(overlap)[:5]}..."
        )
    logger.info("Overlap check PASS: %s has 0 shared samples", domain)
