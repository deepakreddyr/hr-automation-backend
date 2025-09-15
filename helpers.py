# helpers.py (or near your shortlist code)
import re
import json

def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', str(s).lower()).strip()

def _blobify(candidate) -> str:
    if isinstance(candidate, str):
        return _norm(candidate)
    try:
        return _norm(" ".join(map(str, candidate)))
    except Exception:
        return _norm(str(candidate))

def find_true_index_by_name(scraped_candidates, name: str):
    """
    Try to locate the candidate's true index in scraped_candidates by name.
    Strategy: substring match on normalized joined blob; fallback token check.
    Returns an int index or None if not found.
    """
    norm_name = _norm(name)
    # 1) direct substring match
    for idx, cand in enumerate(scraped_candidates):
        if norm_name and norm_name in _blobify(cand):
            return idx

    # 2) token fallback (handles minor differences)
    tokens = [t for t in re.split(r'[^a-z0-9]+', norm_name) if t]
    tokens = [t for t in tokens if t not in {"mr", "ms", "mrs"}]
    if not tokens:
        return None

    for idx, cand in enumerate(scraped_candidates):
        blob = _blobify(cand)
        if all(tok in blob for tok in tokens):
            return idx

    return None

def correct_shortlisted_indices(shortlisted, scraped_candidates):
    """
    shortlisted: list of {"index": int, "name": str, "email": str}
    scraped_candidates: the list returned by your `scrape()` function

    Returns a new list with corrected "index" and an "index_verified" flag.
    Deduplicates indices to avoid repeats.
    """
    corrected = []
    used = set()

    for item in shortlisted:
        name = item.get("name") or ""
        old_idx = item.get("index")
        true_idx = find_true_index_by_name(scraped_candidates, name)

        if true_idx is None:
            # keep old index if present but mark unverified
            corrected.append({
                **item,
                "index_verified": False
            })
            continue

        # de-dup
        if true_idx in used:
            # keep but mark duplicate; you can decide to skip instead
            corrected.append({
                **item,
                "index": true_idx,
                "index_verified": True,
                "duplicate_index": True
            })
        else:
            corrected.append({
                **item,
                "index": true_idx,
                "index_verified": True
            })
            used.add(true_idx)

    return corrected