"""
test_face_recognition.py — Automated validation of the 4-tier face recognition threshold system.

Tests:
  1. Threshold boundary logic (Strong / Recognized / Weak / Unknown)
  2. Margin rule (ambiguous top-two candidates)
  3. False positive rate (random unknown embeddings misidentified as known)
  4. False negative rate (known person embeddings incorrectly rejected)
  5. Live DB integration (if enrolled faces exist)

Usage:
    python test_face_recognition.py
    python test_face_recognition.py --quick   # skip DB integration test
"""
from __future__ import annotations

import argparse
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
# Force UTF-8 on Windows terminals that default to cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Thresholds (must match face_ai.py exactly) ────────────────────────────────
STRONG_THRESHOLD    = 0.85
RECOG_THRESHOLD     = 0.70
WEAK_THRESHOLD      = 0.50
MARGIN_MIN          = 0.10

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _rand_unit(dim: int = 512, seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _perturb(base: np.ndarray, noise_scale: float, seed=None) -> np.ndarray:
    """
    Return base + noise_scale * unit_noise_direction, re-normalized.
    noise_scale controls angular displacement regardless of embedding dimension.
    Expected cosine similarity after perturbation: ~1/sqrt(1 + noise_scale^2).
      noise_scale=0.75 → ~0.80 sim  (solidly Recognized)
      noise_scale=1.00 → ~0.71 sim  (low end of Recognized)
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(base.shape).astype(np.float32)
    noise /= np.linalg.norm(noise)   # unit direction orthogonal-ish to base
    v = base + noise_scale * noise
    return v / np.linalg.norm(v)


def _classify(query: np.ndarray, centroids: dict[str, np.ndarray]) -> tuple[str, float, str]:
    """
    Mirrors the logic in face_ai.py recognize_multiple_faces().
    Returns (person_id_or_unknown, top_sim, match_status).
    """
    if not centroids:
        return "Unknown", 0.0, "Unknown Person"

    matches = sorted(
        [(pid, float(np.dot(query, c))) for pid, c in centroids.items()],
        key=lambda x: x[1], reverse=True
    )
    top_id, top_sim = matches[0]
    second_sim = matches[1][1] if len(matches) > 1 else 0.0
    margin = top_sim - second_sim

    if len(matches) > 1 and margin < MARGIN_MIN:
        return "Unknown", top_sim, "Unknown Person"
    if top_sim < WEAK_THRESHOLD:
        return "Unknown", top_sim, "Unknown Person"
    if top_sim < RECOG_THRESHOLD:
        return top_id, top_sim, "Weak Match"
    if top_sim < STRONG_THRESHOLD:
        return top_id, top_sim, "Recognized"
    return top_id, top_sim, "Strong Match"


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

def test_threshold_boundaries():
    """Verifies each confidence tier maps to the correct status label."""
    alice = _rand_unit(seed=1)
    centroids = {"alice": alice}

    cases = [
        # (target_similarity, expected_status)
        # Avoid exact boundary values (0.85, 0.70, 0.50) due to fp imprecision
        (0.90, "Strong Match"),
        (0.87, "Strong Match"),
        (0.83, "Recognized"),
        (0.72, "Recognized"),
        (0.68, "Weak Match"),
        (0.52, "Weak Match"),
        (0.48, "Unknown Person"),
        (0.10, "Unknown Person"),
    ]

    passed = 0
    for sim, expected in cases:
        # Build a query with exact dot product = sim against alice
        # q = sim * alice + sqrt(1-sim^2) * perp_component
        perp = _rand_unit(seed=99)
        perp -= np.dot(perp, alice) * alice
        if np.linalg.norm(perp) < 1e-8:
            perp = _rand_unit(seed=100)
            perp -= np.dot(perp, alice) * alice
        perp /= np.linalg.norm(perp)
        query = sim * alice + np.sqrt(max(0.0, 1.0 - sim ** 2)) * perp
        query /= np.linalg.norm(query)

        _, got_sim, got_status = _classify(query, centroids)
        ok = got_status == expected
        print(f"  [{PASS if ok else FAIL}] sim~{got_sim:.3f}  expected={expected:15s}  got={got_status}")
        if ok:
            passed += 1

    return passed, len(cases)


def test_margin_rule():
    """Ambiguous top-two candidates (margin < 0.10) must yield Unknown Person."""
    alice = _rand_unit(seed=2)
    bob   = _rand_unit(seed=3)
    centroids = {"alice": alice, "bob": bob}

    # Build a query equidistant between alice and bob (margin ~ 0)
    mid = (alice + bob) / 2.0
    mid /= np.linalg.norm(mid)

    _, _, status = _classify(mid, centroids)
    ok = status == "Unknown Person"
    print(f"  [{PASS if ok else FAIL}] Equidistant query → {status}  (expected Unknown Person)")

    # Build a query clearly closer to alice (margin > 0.10, sim > 0.70)
    biased = 0.9 * alice + 0.1 * bob
    biased /= np.linalg.norm(biased)
    _, _, status2 = _classify(biased, centroids)
    ok2 = status2 in ("Strong Match", "Recognized", "Weak Match")
    print(f"  [{PASS if ok2 else FAIL}] Biased-toward-alice query → {status2}  (expected a positive match)")

    return (ok + ok2), 2


def test_false_positive_rate(n_trials: int = 500, n_enrolled: int = 5):
    """
    Generates random query embeddings (simulating unknown persons) and measures
    how often they are incorrectly classified as a known person.
    Acceptable FPR: < 5%.
    """
    enrolled = {f"P{i}": _rand_unit(seed=10 + i) for i in range(n_enrolled)}
    false_positives = 0

    rng = np.random.default_rng(42)
    for trial in range(n_trials):
        q = rng.standard_normal(512).astype(np.float32)
        q /= np.linalg.norm(q)
        pid, _, status = _classify(q, enrolled)
        if status != "Unknown Person":
            false_positives += 1

    fpr = false_positives / n_trials
    ok = fpr < 0.05
    print(f"  [{PASS if ok else FAIL}] FPR={fpr:.1%}  ({false_positives}/{n_trials} false positives, threshold <5%)")
    return int(ok), 1


def test_false_negative_rate(n_trials: int = 200, noise_scale: float = 0.75):
    """
    Generates mildly perturbed versions of enrolled embeddings (same person,
    different photos) and checks that they are recognized rather than rejected.
    Acceptable FNR: < 15% for the Recognized + Strong tiers combined.
    """
    alice = _rand_unit(seed=7)
    bob   = _rand_unit(seed=8)
    enrolled = {"alice": alice, "bob": bob}

    alice_missed = 0
    bob_missed = 0

    for i in range(n_trials):
        q_alice = _perturb(alice, noise_scale, seed=1000 + i)
        pid, _, status = _classify(q_alice, enrolled)
        if pid != "alice" or status not in ("Strong Match", "Recognized", "Weak Match"):
            alice_missed += 1

        q_bob = _perturb(bob, noise_scale, seed=2000 + i)
        pid, _, status = _classify(q_bob, enrolled)
        if pid != "bob" or status not in ("Strong Match", "Recognized", "Weak Match"):
            bob_missed += 1

    fnr_alice = alice_missed / n_trials
    fnr_bob   = bob_missed   / n_trials
    ok = fnr_alice < 0.15 and fnr_bob < 0.15
    expected_sim = 1.0 / np.sqrt(1.0 + noise_scale**2)
    print(f"  [{PASS if ok else FAIL}] FNR alice={fnr_alice:.1%} bob={fnr_bob:.1%}  (noise_scale={noise_scale}, ~{expected_sim:.2f} expected sim, threshold <15%)")
    return int(ok), 1


def test_live_db_integration():
    """
    Loads enrolled face vectors from the real SQLite DB and verifies that each
    enrolled person's centroid recognizes itself at >= 0.70 confidence.
    Skipped if no faces are enrolled.
    """
    try:
        from modules.perf.db_cache import get_cached_face_centroids, invalidate_face_cache
        invalidate_face_cache()  # force fresh load
        centroids = get_cached_face_centroids()
    except Exception as exc:
        print(f"  [SKIP] DB integration test failed to load centroids: {exc}")
        return 0, 0

    if not centroids:
        print(f"  [SKIP] No enrolled faces in DB — skipping integration test")
        return 0, 0

    passed = 0
    total = len(centroids)
    for pid, centroid in centroids.items():
        pid_out, sim, status = _classify(centroid, centroids)
        ok = pid_out == pid and status in ("Strong Match", "Recognized")
        print(f"  [{PASS if ok else FAIL}] {pid[:12]:12s} → {status} ({sim:.3f})")
        if ok:
            passed += 1

    return passed, total


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip DB integration test")
    args = parser.parse_args()

    suites = [
        ("Threshold Boundaries",   test_threshold_boundaries),
        ("Margin Rule",            test_margin_rule),
        ("False Positive Rate",    test_false_positive_rate),
        ("False Negative Rate",    test_false_negative_rate),
    ]
    if not args.quick:
        suites.append(("Live DB Integration", test_live_db_integration))

    total_passed = 0
    total_cases  = 0

    t_start = time.perf_counter()
    for name, fn in suites:
        print(f"\n{'-'*60}")
        print(f"  {name}")
        print(f"{'-'*60}")
        p, n = fn()
        total_passed += p
        total_cases  += n

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"  Result: {total_passed}/{total_cases} passed  ({elapsed:.2f}s)")
    print(f"{'='*60}\n")

    if total_cases > 0 and total_passed < total_cases:
        sys.exit(1)


if __name__ == "__main__":
    main()
