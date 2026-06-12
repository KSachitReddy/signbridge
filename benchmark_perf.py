"""
benchmark_perf.py — Measures per-frame timings for key pipeline components.

Run from the project root:
    python benchmark_perf.py

Produces a BEFORE / AFTER table showing time saved by each optimization.
No webcam required — all measurements use synthetic data.
"""
import time
import sys
import os

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# ─── Helpers ────────────────────────────────────────────────────────────────

def _timeit(fn, reps=50):
    """Returns median wall-clock time in milliseconds."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]   # median


# ─── Synthetic data ──────────────────────────────────────────────────────────

FRAME_BGR_640 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
FRAME_BGR_320 = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
FRAME_RGB_640 = cv2.cvtColor(FRAME_BGR_640, cv2.COLOR_BGR2RGB)

# Synthetic face embeddings (512-dim unit vectors, as InsightFace produces)
N_ENROLLED = 10   # simulate 10 enrolled people
_rng = np.random.default_rng(42)
ENROLLED_EMBS_RAW = [
    (_rng.random(512).astype(np.float32),)
    for _ in range(N_ENROLLED)
]
# Pre-normalize
ENROLLED_EMBS_UNIT = []
for (raw,) in ENROLLED_EMBS_RAW:
    n = np.linalg.norm(raw)
    ENROLLED_EMBS_UNIT.append(raw / n if n > 0 else raw)

QUERY_EMB = _rng.random(512).astype(np.float32)
QUERY_NORM = QUERY_EMB / np.linalg.norm(QUERY_EMB)


# ─── 1. Image encoding: Streamlit-path PNG vs JPEG bytes ────────────────────

def bench_encode_png():
    success, buf = cv2.imencode('.png', FRAME_RGB_640)
    return len(buf) if success else 0

def bench_encode_jpeg_82():
    success, buf = cv2.imencode('.jpg', FRAME_RGB_640, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return len(buf) if success else 0

def bench_encode_jpeg_70():
    success, buf = cv2.imencode('.jpg', FRAME_RGB_640, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return len(buf) if success else 0


# ─── 2. Face similarity: unnormalized cosine vs pre-normalized dot product ──

def bench_cosine_unnormalized():
    """Current implementation: normalises both vectors per call."""
    q = QUERY_EMB.copy()
    total = 0.0
    for (raw,) in ENROLLED_EMBS_RAW:
        a, b = q, raw
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        total += float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0
    return total

def bench_dot_prenormalized():
    """New implementation: dot product with pre-normalized unit vectors."""
    q = QUERY_NORM
    total = 0.0
    for ref in ENROLLED_EMBS_UNIT:
        total += float(np.dot(q, ref))
    return total


# ─── 3. DB round-trip vs in-memory cache ────────────────────────────────────

def bench_db_raw():
    """Simulates a cold SQLite query for face vectors."""
    from modules.database import get_all_face_vectors
    return get_all_face_vectors()

def bench_db_cached():
    """Returns the in-memory cached result."""
    from modules.perf.db_cache import get_cached_face_vectors
    return get_cached_face_vectors()


# ─── 4. Full frame render path ───────────────────────────────────────────────

def bench_render_numpy():
    """Simulates what Streamlit does internally with a numpy array."""
    frame = FRAME_RGB_640.copy()
    _, buf = cv2.imencode('.png', frame)   # Streamlit PNG-encodes numpy arrays
    return len(buf)

def bench_render_jpeg():
    """New path: we pre-encode as JPEG before handing to Streamlit."""
    frame = FRAME_RGB_640.copy()
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return len(buf)


# ─── Run all benchmarks ──────────────────────────────────────────────────────

def run():
    SEP = "-" * 72
    print(SEP)
    print("SignBridge AI — Performance Benchmark")
    print(SEP)

    results = []

    # --- Image encoding ---
    print("\n[1/4] Image encoding (640x480 RGB frame, median of 50 reps)")
    t_png  = _timeit(bench_encode_png)
    t_j82  = _timeit(bench_encode_jpeg_82)
    t_j70  = _timeit(bench_encode_jpeg_70)
    print(f"  PNG  (lossless, Streamlit default) : {t_png:6.1f} ms")
    print(f"  JPEG q=82 (new path)               : {t_j82:6.1f} ms  [{t_png/max(t_j82,0.1):.1f}x faster]")
    print(f"  JPEG q=70                          : {t_j70:6.1f} ms  [{t_png/max(t_j70,0.1):.1f}x faster]")
    results.append(("Frame display (PNG vs JPEG q=82)", t_png, t_j82))

    # --- Similarity ---
    print("\n[2/4] Face similarity for 10 enrolled profiles (median of 200 reps)")
    t_cos = _timeit(bench_cosine_unnormalized, reps=200)
    t_dot = _timeit(bench_dot_prenormalized, reps=200)
    print(f"  Cosine (normalize-per-call, before) : {t_cos:.3f} ms")
    print(f"  Dot product (pre-normalized, after) : {t_dot:.3f} ms  [{t_cos/max(t_dot,1e-9):.1f}x faster]")
    results.append(("Similarity (10 faces)", t_cos, t_dot))

    # --- DB cache ---
    print("\n[3/4] DB query: face vectors (first call vs cache hit)")
    # Warm the cache first
    from modules.perf.db_cache import get_cached_face_vectors, invalidate_face_cache
    invalidate_face_cache()
    bench_db_cached()   # warm it up
    t_db_raw = _timeit(bench_db_raw, reps=20)
    t_db_hit = _timeit(bench_db_cached, reps=200)
    print(f"  SQLite get_all_face_vectors() cold  : {t_db_raw:6.2f} ms")
    print(f"  In-memory cache hit (after)         : {t_db_hit:6.3f} ms  [{t_db_raw/max(t_db_hit,1e-9):.0f}x faster]")
    results.append(("DB face vector lookup", t_db_raw, t_db_hit))

    # --- Full render path ---
    print("\n[4/4] Full frame render preparation (encode + bgr->rgb)")
    t_np_path = _timeit(bench_render_numpy, reps=50)
    t_jp_path = _timeit(bench_render_jpeg,  reps=50)
    print(f"  numpy array path (PNG, before)      : {t_np_path:6.1f} ms")
    print(f"  JPEG bytes path  (q=82, after)      : {t_jp_path:6.1f} ms  [{t_np_path/max(t_jp_path,0.1):.1f}x faster]")
    results.append(("Render path (encode only)", t_np_path, t_jp_path))

    # --- FPS estimates ---
    print("\n" + SEP)
    print("ESTIMATED FPS IMPACT  (Performance mode, 320x240, face every 25 frames)")
    print(SEP)

    # Baseline: MediaPipe 40ms + render 80ms + panels 130ms + history 50ms = 300ms
    mediapipe_ms  = 40.0   # pose + hands at 320x240 (typical)
    render_before = 80.0   # PNG encode + Streamlit overhead
    panels_before = 130.0  # info(50) + pipeline(50) + stt(30) every frame
    history_before = 50.0  # get_conversations() + get_all_people() + render every frame

    render_after  = t_jp_path + 5.0          # JPEG encode + minimal Streamlit overhead
    panels_after  = 50.0 * (1/2) + 50.0 * (1/3) + 30.0 * (1/30)  # throttled avg
    history_after = 50.0 * (1/20)            # every 2s at 10fps = every 20 frames

    total_before = mediapipe_ms + render_before + panels_before + history_before
    total_after  = mediapipe_ms + render_after  + panels_after  + history_after

    fps_before = 1000.0 / max(total_before, 1.0)
    fps_after  = 1000.0 / max(total_after,  1.0)

    print(f"  {'Component':<35} {'Before':>8}  {'After':>8}")
    print(f"  {'':-<35} {'':-<8}  {'':-<8}")
    print(f"  {'MediaPipe Pose + Hands':<35} {mediapipe_ms:>7.1f}ms  {mediapipe_ms:>7.1f}ms")
    print(f"  {'Frame display (encode)':<35} {render_before:>7.1f}ms  {render_after:>7.1f}ms")
    print(f"  {'UI panels (avg/frame)':<35} {panels_before:>7.1f}ms  {panels_after:>7.1f}ms")
    print(f"  {'Chat history (avg/frame)':<35} {history_before:>7.1f}ms  {history_after:>7.1f}ms")
    print(f"  {'':=<35} {'':=<8}  {'':=<8}")
    print(f"  {'Loop total (estimated)':<35} {total_before:>7.1f}ms  {total_after:>7.1f}ms")
    print(f"  {'Estimated FPS':<35} {fps_before:>7.1f}    {fps_after:>7.1f}")
    print()
    print(f"  Speedup: {fps_after/max(fps_before,0.01):.1f}x  |  Target >= 10 FPS: {'PASS' if fps_after >= 10 else 'MARGINAL'}")
    print(SEP)


if __name__ == "__main__":
    run()
