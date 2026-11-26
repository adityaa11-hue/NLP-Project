from __future__ import annotations

import logging
import hashlib
import os
from functools import lru_cache
from pathlib import Path
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
import json
import re
import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from misinfo.xfact import read_xfact

logger = logging.getLogger("misinfo.evidence")
ROOT = Path(__file__).resolve().parents[1]
XFACT_DIR = ROOT / "data" / "x-fact"
FILES = ["train.all.tsv", "dev.all.tsv", "test.all.tsv", "ood.tsv"]
MIN_RELEVANCE = 0.28
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
NUMBER_RE = re.compile(r"(?<!\w)\d[\d,]*(?:\.\d+)?")
ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9&.'’/-]*)(?:\s+[A-Z][A-Za-z0-9&.'’/-]*)*\b")
ENTITY_STOP = {"a", "an", "and", "but", "he", "it", "she", "the", "they", "this", "we"}
INDEX_CACHE_DIR = ROOT / ".cache" / "xfact"
_index_build_lock = threading.Lock()
_INDEX_STATUS = {"en": "not_started", "hi": "not_started"}


def _entity_terms(text: str) -> set[str]:
    values = set()
    for match in ENTITY_RE.finditer(text):
        value = " ".join(match.group(0).casefold().split())
        if value.split()[0] in ENTITY_STOP and len(value.split()) == 1:
            continue
        if match.start() == 0 and len(value.split()) == 1 and not value.isupper():
            continue
        if len(value.split()) > 1 or value.isupper() or match.start() > 0:
            values.add(value)
    return values


def _canonical(text: str) -> str:
    value = unicodedata.normalize("NFKC", URL_RE.sub(" ", str(text))).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE).split())


def _context_alignment(query: str, candidate: str) -> dict[str, object]:
    query_entities, candidate_entities = _entity_terms(query), _entity_terms(candidate)
    query_numbers, candidate_numbers = set(NUMBER_RE.findall(query)), set(NUMBER_RE.findall(candidate))
    return {
        "shared_entities": sorted(query_entities & candidate_entities),
        "entity_conflict": bool(query_entities and candidate_entities and query_entities.isdisjoint(candidate_entities)),
        "shared_quantities": sorted(query_numbers & candidate_numbers),
        "quantity_conflict": bool(query_numbers and candidate_numbers and query_numbers.isdisjoint(candidate_numbers)),
    }


def _pair_similarity(query: str, candidate: str) -> float:
    """Blend lexical word/character overlap with date and quantity agreement."""
    if not query.strip() or not candidate.strip():
        return 0.0
    char_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True)
    word_vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    char_mat = char_vec.fit_transform([query, candidate])
    word_mat = word_vec.fit_transform([query, candidate])
    char_score = float(cosine_similarity(char_mat[0], char_mat[1])[0, 0])
    word_score = float(cosine_similarity(word_mat[0], word_mat[1])[0, 0])
    score = 0.42 * char_score + 0.58 * word_score
    query_entities = _entity_terms(query)
    candidate_entities = _entity_terms(candidate)
    if query_entities and candidate_entities:
        if query_entities.isdisjoint(candidate_entities):
            score *= 0.82
        else:
            score = min(1.0, score + 0.035)
    query_numbers = set(NUMBER_RE.findall(query))
    candidate_numbers = set(NUMBER_RE.findall(candidate))
    if query_numbers and candidate_numbers:
        if query_numbers.isdisjoint(candidate_numbers):
            score *= 0.65
        else:
            score = min(1.0, score + 0.04)
    return max(0.0, min(1.0, score))


@lru_cache(maxsize=2)
def _local_index(language: str):
    """Load the local TF-IDF index from disk when corpus inputs are unchanged."""
    fingerprint_parts = ["xfact-index-v2", language, SKLEARN_VERSION]
    for filename in FILES:
        path = XFACT_DIR / filename
        if path.exists():
            stat = path.stat()
            fingerprint_parts.append(f"{filename}:{stat.st_size}:{stat.st_mtime_ns}")
        else:
            fingerprint_parts.append(f"{filename}:missing")
    fingerprint = hashlib.sha256("|".join(fingerprint_parts).encode("utf-8")).hexdigest()[:20]
    cache_path = INDEX_CACHE_DIR / f"{language}-{fingerprint}.joblib"

    # Serialize concurrent cold starts and recheck after acquiring the lock.
    with _index_build_lock:
        _INDEX_STATUS[language] = "loading"
        try:
            if cache_path.is_file():
                try:
                    cached = joblib.load(cache_path)
                    if isinstance(cached, tuple) and len(cached) == 5:
                        _INDEX_STATUS[language] = "ready"
                        logger.info("Loaded %s X-FACT search index from disk cache", language)
                        return cached
                except (OSError, ValueError, EOFError, AttributeError, ImportError) as exc:
                    logger.warning("Ignoring unusable X-FACT index cache (%s)", type(exc).__name__)

            index = _build_local_index(language)
            try:
                INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                temp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp")
                joblib.dump(index, temp_path)
                os.replace(temp_path, cache_path)
                logger.info("Saved %s X-FACT search index to disk cache", language)
            except OSError as exc:
                logger.warning("Could not save X-FACT search index cache (%s)", type(exc).__name__)
            _INDEX_STATUS[language] = "ready"
            return index
        except Exception:
            _INDEX_STATUS[language] = "error"
            raise


def index_readiness() -> dict[str, str]:
    return dict(_INDEX_STATUS)


def _build_local_index(language: str):
    frames = []
    problems = []
    present_files = 0
    for filename in FILES:
        path = XFACT_DIR / filename
        if path.exists():
            present_files += 1
            try:
                frame = read_xfact(path, {language})
            except (OSError, ValueError, UnicodeError) as exc:
                logger.warning("Could not parse local X-FACT file %s: %s", filename, type(exc).__name__)
                problems.append(filename)
                continue
            if not frame.empty:
                frames.append(frame)
    if not frames:
        reason = ", ".join(problems) or "no data files found" if present_files or problems else "no X-FACT files found"
        raise RuntimeError(f"No parseable local X-FACT index for {language}: {reason}.")
    frame = pd.concat(frames, ignore_index=True).dropna(subset=["claim"])
    frame["_canonical"] = frame.claim.astype(str).map(_canonical)
    frame = frame[frame._canonical.str.len() >= 12].drop_duplicates(subset=["_canonical"], keep="first").reset_index(drop=True)
    # Entity and quantity features are constant per corpus claim. Cache them with
    # the matrices to avoid rescanning every corpus row for every user query.
    claim_texts = frame.claim.astype(str)
    frame["_entities"] = claim_texts.map(_entity_terms)
    frame["_quantities"] = claim_texts.map(lambda value: set(NUMBER_RE.findall(value)))
    char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, max_features=180_000, sublinear_tf=True)
    char_matrix = char_vectorizer.fit_transform(frame.claim.astype(str).tolist())
    word_vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=1, max_features=180_000, sublinear_tf=True)
    word_matrix = word_vectorizer.fit_transform(frame.claim.astype(str).tolist())
    return frame, char_vectorizer, char_matrix, word_vectorizer, word_matrix


def _similarity(query: str, candidate: str) -> float:
    return _pair_similarity(query, candidate)


def search_local(claim: str, language: str, limit: int = 5) -> list[dict]:
    index = _local_index(language)
    if len(_canonical(claim)) < 12:
        return []
    frame, char_vectorizer, char_matrix, word_vectorizer, word_matrix = index
    char_scores = cosine_similarity(char_vectorizer.transform([claim]), char_matrix).ravel()
    word_scores = cosine_similarity(word_vectorizer.transform([claim]), word_matrix).ravel()
    scores = 0.42 * char_scores + 0.58 * word_scores
    query_numbers = set(NUMBER_RE.findall(claim))
    query_entities = _entity_terms(claim)
    candidate_entities = frame["_entities"].tolist()
    candidate_quantities = frame["_quantities"].tolist()
    if query_numbers:
        for position, (entities, quantities) in enumerate(zip(candidate_entities, candidate_quantities)):
            if query_entities and entities:
                if query_entities.isdisjoint(entities):
                    scores[position] *= 0.82
                else:
                    scores[position] = min(1.0, scores[position] + 0.035)
            if quantities:
                if query_numbers.isdisjoint(quantities):
                    scores[position] *= 0.65
                else:
                    scores[position] = min(1.0, scores[position] + 0.04)
    elif query_entities:
        for position, entities in enumerate(candidate_entities):
            if entities and query_entities.isdisjoint(entities):
                scores[position] *= 0.82
            elif query_entities & entities:
                scores[position] = min(1.0, scores[position] + 0.035)
    indices = scores.argsort()[::-1]
    results = []
    for index_num in indices:
        score = float(scores[index_num])
        if score < MIN_RELEVANCE:
            break
        row = frame.iloc[int(index_num)]
        link = str(row.get("link_1", "") or "")
        evidence = str(row.get("evidence_1", "") or "")
        if evidence in {"NO_EVIDENCE", "<DUMMY_EVIDENCE>", "none"}:
            evidence = ""
        results.append({
            "claim": str(row.claim), "rating": str(row.get("label", "Rating unavailable")),
            "publisher": str(row.get("site", "Fact-check source")), "title": str(row.claim),
            "url": link if link.startswith(("http://", "https://")) else "",
            "excerpt": evidence[:900], "review_date": str(row.get("reviewDate", "")),
            "similarity": round(score, 3), "match_type": "Word/character and numeric-context match",
            "context_alignment": _context_alignment(claim, str(row.claim)),
            "provider": "X-FACT local corpus",
        })
        if len(results) >= limit:
            break
    return results


def _google_error_message(code: int, reason: str = "") -> str:
    if code in {401, 403}:
        return "Google Fact Check API rejected the request. Check that the API is enabled, the key is valid, and quota is available."
    if code == 429:
        return "Google Fact Check API quota or rate limit was reached. Try again later."
    if code >= 500:
        return "Google Fact Check API is temporarily unavailable. Try again later."
    return f"Google Fact Check API returned HTTP {code}." + (f" {reason[:160]}" if reason else "")


def _google_api_key() -> str:
    """Read the key from process env or Streamlit Cloud secrets without logging it."""
    key = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st
        for name in ("GOOGLE_FACT_CHECK_API_KEY", "google_fact_check_api_key"):
            value = st.secrets.get(name, "")
            if value:
                return str(value).strip()
    except Exception as exc:
        # Streamlit raises a secrets-file exception in local/API contexts when
        # no secrets are configured. The evidence path remains usable offline.
        logger.debug("Streamlit secrets unavailable for Google Fact Check (%s)", type(exc).__name__)
    return ""


def search_google(claim: str, language: str, api_key: str | None = None) -> dict:
    key = api_key or _google_api_key()
    if not key:
        return {"status": "not_configured", "results": [], "error": "Live Google Fact Check search is not configured."}
    if len(_canonical(claim)) < 12:
        return {"status": "success_empty", "results": [], "error": "Search query is too short."}
    params = urlencode({"query": claim[:500], "languageCode": language, "pageSize": 5, "key": key})
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?{params}"
    try:
        with urlopen(url, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        logger.warning("Google Fact Check request failed with HTTP %s", exc.code)
        detail = ""
        try:
            body = json.loads(exc.read().decode("utf-8"))
            detail = body.get("error", {}).get("message", "")
        except (OSError, UnicodeError, ValueError):
            pass
        return {"status": "error", "results": [], "error": _google_error_message(exc.code, detail)}
    except (URLError, TimeoutError, OSError) as exc:
        logger.warning("Google Fact Check network request failed: %s", type(exc).__name__)
        return {"status": "error", "results": [], "error": "Could not reach Google Fact Check API (network or timeout error). Local X-FACT search can still be used."}
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Google Fact Check returned an unreadable response: %s", type(exc).__name__)
        return {"status": "error", "results": [], "error": "Google Fact Check returned an unreadable response. Check the API response and try again."}

    if not isinstance(payload, dict) or not isinstance(payload.get("claims", []), list):
        logger.warning("Google Fact Check returned an unexpected response structure")
        return {"status": "error", "results": [], "error": "Google Fact Check returned an unexpected response structure. No evidence results were used."}

    matches = []
    for item in payload.get("claims", []):
        if not isinstance(item, dict):
            continue
        reviews = item.get("claimReview", [])
        if not isinstance(reviews, list):
            continue
        for review in reviews[:3]:
            if not isinstance(review, dict):
                continue
            match_claim = str(item.get("text", ""))
            score = _similarity(claim, match_claim)
            if score < MIN_RELEVANCE:
                continue
            publisher = review.get("publisher") or {}
            if not isinstance(publisher, dict):
                publisher = {}
            url = str(review.get("url", ""))
            if not url.startswith(("http://", "https://")):
                url = ""
            matches.append({
                "claim": match_claim, "rating": str(review.get("textualRating", "Rating not provided")),
                "publisher": str(publisher.get("name", "Fact-check publisher")),
                "title": str(review.get("title", match_claim)), "url": url,
                "excerpt": "", "review_date": str(review.get("reviewDate", "")),
            "similarity": round(score, 3), "match_type": "Google result ranked by word/character and numeric-context match",
                "context_alignment": _context_alignment(claim, match_claim),
                "provider": "Google Fact Check Tools API",
            })
    matches.sort(key=lambda match: match["similarity"], reverse=True)
    return {"status": "success" if matches else "success_empty", "results": matches[:5], "error": None}


def retrieve_evidence(claim: str, language: str) -> dict:
    try:
        local = search_local(claim, language)
        local_error = None
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("Local X-FACT retrieval failed")
        local, local_error = [], f"Local X-FACT search failed ({type(exc).__name__}). Check the downloaded corpus."
    live = search_google(claim, language)
    combined = local + live["results"]
    deduped = {}
    for match in combined:
        key = (match.get("url") or "", _canonical(match.get("claim", "")))
        if key not in deduped or (match.get("similarity") or 0) > (deduped[key].get("similarity") or 0):
            deduped[key] = match
    ranked = sorted(deduped.values(), key=lambda item: item.get("similarity", 0), reverse=True)
    if ranked:
        status = "matches_found"
        message = "Related fact-check records found. Compare the claim wording, context, date, and publisher before relying on a result."
    elif live["status"] == "error":
        status = "search_error"
        message = live["error"]
    elif local_error:
        status = "search_error"
        message = local_error
    elif live["status"] == "not_configured":
        status = "local_empty_live_off"
        message = "No sufficiently relevant fact-check record was found in the local corpus. Live Google search is not configured."
    else:
        status = "no_results"
        message = "No sufficiently relevant fact-check record was found for this claim. Absence of a match is not evidence that it is true or false."
    return {
        "status": status, "message": message, "matches": ranked[:5],
        "local_matches": local, "google_matches": live["results"],
        "google_status": live["status"], "google_error": live["error"], "local_error": local_error,
        "live_enabled": live["status"] != "not_configured",
    }
