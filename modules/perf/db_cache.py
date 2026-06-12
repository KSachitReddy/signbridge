"""
db_cache.py — Thread-safe TTL cache for DB queries inside the real-time camera loop.

Every recognition frame previously called get_all_face_vectors() and get_all_people()
directly from SQLite (~5-15ms each). The conversation history was re-fetched
every frame too. With enrollment changing at most once per session this is wasteful.

This module provides an 8-second TTL in-process cache that drops those DB calls to
~0.1ms after the first load.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

import numpy as np


class _TTLCache:
    def __init__(self, ttl: float):
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.monotonic() - entry[0] < self._ttl:
                return entry[1]
            del self._store[key]
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.monotonic(), value)

    def evict(self, key: str = None) -> None:
        with self._lock:
            if key:
                self._store.pop(key, None)
            else:
                self._store.clear()


_face_vector_cache = _TTLCache(ttl=8.0)
_people_cache = _TTLCache(ttl=8.0)


def get_cached_face_vectors() -> list:
    """
    Returns face vectors from in-memory cache (8s TTL) or DB.

    On the first call (or after cache expiry) the result is fetched from SQLite,
    each embedding is pre-normalized to a unit vector, and the list is stored
    in memory for subsequent calls within the TTL window.

    Pre-normalization means face similarity comparisons can use np.dot instead of
    the full cosine formula, halving the comparison cost for every enrolled face.
    """
    cached = _face_vector_cache.get("all")
    if cached is not None:
        return cached
    from modules.database import get_all_face_vectors
    vectors = get_all_face_vectors()
    for ev in vectors:
        emb = ev.get("embedding") or []
        if emb:
            arr = np.array(emb, dtype=np.float32)
            n = float(np.linalg.norm(arr))
            ev["_norm_emb"] = (arr / n).tolist() if n > 1e-8 else list(emb)
        else:
            ev["_norm_emb"] = []
    _face_vector_cache.put("all", vectors)
    return vectors


def get_cached_people() -> list:
    """Returns people records from in-memory cache (8s TTL) or DB."""
    cached = _people_cache.get("all")
    if cached is not None:
        return cached
    from modules.database import get_all_people
    people = get_all_people()
    _people_cache.put("all", people)
    return people


def invalidate_face_cache() -> None:
    """Call after enrollment so the next recognition reads fresh data from DB."""
    _face_vector_cache.evict("all")
    _people_cache.evict("all")
