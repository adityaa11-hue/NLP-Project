from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import joblib

from misinfo.evidence import retrieve_evidence

logger = logging.getLogger("misinfo.pipeline")
LABELS = {"Likely Genuine", "Potentially Misleading", "Unable to Assess"}
STRONG_EVIDENCE_SCORE = 0.78
NEGATIVE_RATING = re.compile(r"\b(false|mostly false|fake|misleading|fabricated|incorrect|debunked|pants on fire)\b", re.I)
POSITIVE_RATING = re.compile(r"\b(true|mostly true|accurate|correct|verified|supported)\b", re.I)
MIXED_RATING = re.compile(r"\b(partly|partially|half true|barely true|mixed|unproven|unverified|unsupported|inconclusive|unclear|not rated)\b", re.I)
SUBJECTIVE_START = re.compile(r"^(i think|i feel|i believe|in my opinion|in my view|it seems|यह मुझे लगता है|मेरी राय में|मेरा मानना है)", re.I)
FACT_CUES = re.compile(r"\b(announced|reported|confirmed|claimed|said|increased|decreased|won|lost|approved|banned|killed|arrested|launched|opened|closed|caused|found|shows|recorded|percent|million|billion|crore|lakh)\b|[₹$€£%]|\d", re.I)
OPINION_CUES = re.compile(r"\b(should|ought to|must|i think|i believe|in my opinion|in my view|probably|perhaps|arguably|it seems|appears to be)\b|यह मेरी राय|मेरा मानना है|चाहिए|शायद", re.I)
HEADING_CUES = re.compile(r"^(breaking|opinion|editorial|commentary|analysis|watch|read more|subscribe|advertisement|sponsored|शीर्षक|शिर्षक|संपादकीय|विज्ञापन|राय)\b", re.I)
NUMBER_RE = re.compile(r"(?<!\w)(?:[$₹€£]\s*)?\d[\d,]*(?:\.\d+)?(?:\s?(?:%|percent|million|billion|crore|lakh|thousand|km|kms|kg|tonnes?|years?|people|homes|jobs))?", re.I)
DATE_RE = re.compile(r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,?\s+\d{4})?|\d{1,2}\s+(?:जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)\s+\d{4}|\d{4})\b", re.I)
ENGLISH_STOP_INITIAL = {"a", "an", "and", "but", "he", "it", "officials", "she", "the", "they", "this", "we"}


@lru_cache(maxsize=1)
def _english_tokenizer():
    import spacy
    return spacy.blank("en").tokenizer


def tokenize_english(text: str) -> list[str]:
    """spaCy word tokenization; retain stop words and negation cues."""
    return [token.text.lower() for token in _english_tokenizer()(text) if not token.is_space and not token.is_punct]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", text)
    return " ".join(text.split())


@lru_cache(maxsize=4)
def _load_model_cached(model_path: str) -> Any | None:
    path = Path(model_path)
    if not path.is_file():
        return None
    try:
        return joblib.load(path)
    except (OSError, ValueError, EOFError, AttributeError) as exc:
        logger.exception("Could not load model artifact (%s)", type(exc).__name__)
        return None


def load_model(path: str | None = None) -> Any | None:
    model_path = Path(path or os.getenv("MISINFO_MODEL_PATH", "models/misinformation_tfidf.joblib"))
    models = _load_model_cached(str(model_path.resolve()))
    # This experimental Hindi artifact was trained on source-confounded data.
    # It stays opt-in and is not loaded as the default Hindi news classifier.
    hindi_path = model_path.with_name("hindi_misinfo_experimental.joblib")
    if os.getenv("MISINFO_ENABLE_EXPERIMENTAL_HINDI", "").strip().lower() in {"1", "true", "yes"} and isinstance(models, dict) and hindi_path.is_file():
        hindi_package = _load_model_cached(str(hindi_path.resolve()))
        if isinstance(hindi_package, dict) and "estimator" in hindi_package:
            models["hi"] = hindi_package
    return models


def _sentence_score(sentence: str) -> float:
    words = re.findall(r"\w+", sentence, flags=re.UNICODE)
    if len(words) < 7 or len(sentence) < 35 or len(sentence) > 900:
        return -1.0
    if sentence.strip().endswith("?") or SUBJECTIVE_START.search(sentence.strip()) or OPINION_CUES.search(sentence):
        return -1.0
    score = min(len(words), 40) / 40
    score += 1.15 if FACT_CUES.search(sentence) else 0.0
    if re.search(r"\b(is|are|was|were|has|have|will|did|does|can|से|है|हैं|था|थी|होगा|किया|कहा|मिला|बना|जारी)\b", sentence, re.I):
        score += 0.35
    if HEADING_CUES.search(sentence.strip()):
        return -1.0
    return score


def _split_sentences(text: str) -> list[str]:
    """Split at sentence boundaries while preserving decimals and common abbreviations."""
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DECIMAL_POINT>", text)
    abbreviations = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Gov|Sen|Rep|St|vs|etc|U\.S|U\.K)\.", re.I)
    protected = abbreviations.sub(lambda m: m.group(0).replace(".", "<ABBR_POINT>"), protected)
    parts = re.split(r"(?<=[.!?।])\s+|\n+", protected)
    return [part.replace("<DECIMAL_POINT>", ".").replace("<ABBR_POINT>", ".").strip() for part in parts if part.strip()]


def _claim_metadata(sentence: str) -> dict[str, list[str]]:
    """Extract literal cues only; this is not named-entity recognition."""
    date_matches = list(DATE_RE.finditer(sentence))
    dates = list(dict.fromkeys(match.group(0).strip() for match in date_matches))
    quantities = list(dict.fromkeys(match.group(0).strip() for match in NUMBER_RE.finditer(sentence)
                                    if not any(date.start() <= match.start() and match.end() <= date.end() for date in date_matches)))
    entities: list[str] = []
    for match in re.finditer(r"\b(?:[A-Z][A-Za-z0-9&.'’/-]*)(?:\s+[A-Z][A-Za-z0-9&.'’/-]*)*\b", sentence):
        value = match.group(0).strip()
        if value.split()[0].casefold() in ENGLISH_STOP_INITIAL and len(value.split()) == 1:
            continue
        if match.start() == 0 and len(value.split()) == 1 and not value.isupper():
            continue
        if len(value.split()) > 1 or value.isupper() or match.start() > 0:
            if value not in entities:
                entities.append(value)
    hindi_entity_patterns = [
        r"(?:[\u0900-\u097F]+\s+){0,3}(?:संगठन|मंत्रालय|सरकार|विभाग|निगम|पार्टी|बैंक|संस्थान|आयोग|संसद)",
        r"[\u0900-\u097F]+-[A-Za-z0-9]+",
    ]
    for pattern in hindi_entity_patterns:
        for match in re.finditer(pattern, sentence):
            value = " ".join(match.group(0).split())
            if value not in entities:
                entities.append(value)
    return {"entities": entities, "dates": dates, "quantities": quantities}


def extract_article_claims(text: str, limit: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Select factual-looking statements and separately report opinion/commentary.

    This is a rules-based screening step, not a trained fact detector. Extracted
    text is never rewritten and no missing context is invented.
    """
    normalized = _normalize(text)
    sentences = [re.sub(r"\s+", " ", item).strip(" \t\r\n\"'“”‘’") for item in _split_sentences(normalized)]
    eligible: list[tuple[int, str, float]] = []
    excluded: list[dict[str, str]] = []
    for index, sentence in enumerate(sentences):
        score = _sentence_score(sentence)
        if score >= 0:
            eligible.append((index, sentence, score))
        else:
            reason = "opinion_or_commentary" if OPINION_CUES.search(sentence) or SUBJECTIVE_START.search(sentence) else "not_selected_as_verifiable"
            excluded.append({"text": sentence, "reason": reason})
    eligible.sort(key=lambda item: (-item[2], item[0]))
    chosen = sorted(eligible[:limit], key=lambda item: item[0])
    claims = [{"text": sentence, "selection_score": round(score, 3), **_claim_metadata(sentence)} for _, sentence, score in chosen]
    return claims, excluded


def extract_candidate_claims(text: str, limit: int = 5) -> list[str]:
    claims, _ = extract_article_claims(text, limit)
    return [claim["text"] for claim in claims]


def _rating_direction(rating: str) -> str:
    rating = " ".join(str(rating or "").casefold().replace("-", " ").split())
    if not rating or MIXED_RATING.search(rating):
        return "mixed"
    if re.search(r"\bnot\s+(?:false|fake|misleading|incorrect)\b", rating):
        return "unknown"
    if re.search(r"\bnot\s+(?:true|accurate|correct|verified)\b|\b(?:no evidence|unsubstantiated)\b", rating):
        return "unknown"
    if NEGATIVE_RATING.search(rating):
        return "refutes"
    if POSITIVE_RATING.search(rating):
        return "supports"
    return "unknown"


def _strong_evidence(evidence: dict) -> dict:
    strong = [match for match in evidence.get("matches", [])
              if (match.get("similarity") or 0) >= STRONG_EVIDENCE_SCORE
              and not match.get("context_alignment", {}).get("entity_conflict", False)
              and not match.get("context_alignment", {}).get("quantity_conflict", False)]
    votes: dict[str, set[str]] = {"supports": set(), "refutes": set()}
    exact = []
    for match in strong:
        direction = _rating_direction(match.get("rating", ""))
        publisher = match.get("publisher") or match.get("provider") or "unknown"
        if direction in votes:
            votes[direction].add(str(publisher).casefold())
        if (match.get("similarity") or 0) >= 0.92 and direction in {"supports", "refutes"}:
            exact.append({"direction": direction, "source": match})
    # A single near-exact fact-check is surfaced as an evidence lead; aggregate corroboration requires two publishers.
    if votes["supports"] and votes["refutes"]:
        verdict = "conflicting"
    elif len(votes["refutes"]) >= 2 or any(item["direction"] == "refutes" for item in exact):
        verdict = "refutes"
    elif len(votes["supports"]) >= 2 or any(item["direction"] == "supports" for item in exact):
        verdict = "supports"
    else:
        verdict = "insufficient"
    assessments = []
    for match in evidence.get("matches", []):
        score = float(match.get("similarity") or 0)
        alignment = match.get("context_alignment", {})
        is_strong = score >= STRONG_EVIDENCE_SCORE and not alignment.get("entity_conflict", False) and not alignment.get("quantity_conflict", False)
        assessments.append({"claim": match.get("claim", ""), "rating": match.get("rating", ""), "publisher": match.get("publisher", ""), "url": match.get("url", ""), "similarity": score, "relevance": "strong" if is_strong else "related_inconclusive", "published_rating_direction": _rating_direction(match.get("rating", "")), "context_alignment": alignment})
    return {"verdict": verdict, "strong_match_count": len(strong), "supporting_publishers": sorted(votes["supports"]), "refuting_publishers": sorted(votes["refutes"]), "match_assessments": assessments}


def _combine_claim(classifier_label: str, evidence_verdict: str) -> tuple[str, str]:
    if evidence_verdict == "conflicting":
        return "Unable to Assess", "Relevant high-similarity records publish conflicting ratings."
    if evidence_verdict == "refutes":
        if classifier_label == "Likely Genuine":
            return "Unable to Assess", "Strong fact-check evidence conflicts with the model's Likely Genuine prediction."
        return "Potentially Misleading", "A high-similarity fact-check record has a negative published rating. Open the source to review it."
    if evidence_verdict == "supports":
        if classifier_label == "Potentially Misleading":
            return "Unable to Assess", "Strong fact-check evidence conflicts with the model's Potentially Misleading prediction."
        return "Likely Genuine", "A high-similarity fact-check record has a positive published rating. Open the source to review it."
    if classifier_label == "Likely Genuine":
        return "Unable to Assess", "The classifier leans genuine, but held-out evaluation is not reliable enough to establish truth without strong supporting fact-check evidence."
    if classifier_label == "Potentially Misleading":
        return "Unable to Assess", "The classifier flags this claim, but held-out evaluation is not reliable enough to establish falsity without strong negative fact-check evidence."
    if classifier_label == "Unable to Assess":
        return classifier_label, "The classifier abstained and no sufficiently strong fact-check match was found."
    return "Unable to Assess", "The classifier and retrieved evidence do not provide enough support for an overall assessment."


def _classifier_summary(labels: list[str], assessments: list[dict[str, Any]]) -> tuple[str, float | None]:
    if labels and all(label == labels[0] for label in labels) and labels[0] in {"Not trained for Hindi", "Unavailable"}:
        return labels[0], None
    if "Potentially Misleading" in labels and "Likely Genuine" in labels:
        return "Unable to Assess (mixed claim predictions)", None
    if "Potentially Misleading" in labels:
        score = max((item["classifier_score"] for item in assessments if item["classifier_label"] == "Potentially Misleading"), default=None)
        return "Potentially Misleading", score
    if labels and all(label == "Likely Genuine" for label in labels):
        score = min((item["classifier_score"] for item in assessments if item["classifier_label"] == "Likely Genuine"), default=None)
        return "Likely Genuine", score
    scored = [item["classifier_score"] for item in assessments if item.get("classifier_score") is not None]
    return "Unable to Assess", round(sum(scored) / len(scored), 4) if scored else None


def analyze_initial(text: str, language: str = "en", model: Any | None = None) -> dict[str, Any]:
    """Return a quick classifier-only preview; this is explicitly not the final verdict."""
    total_started = time.perf_counter()
    if language not in {"en", "hi"}:
        return _unable("Choose English or Hindi.")
    normalized = _normalize(text)
    if len(normalized) < 80:
        return _unable("Add at least 80 characters of article text for screening.")

    extraction_started = time.perf_counter()
    claim_details, excluded = extract_article_claims(normalized)
    extraction_ms = round((time.perf_counter() - extraction_started) * 1000, 2)
    claims = [item["text"] for item in claim_details]
    if not claims:
        result = _unable("No sufficiently factual claim sentences were detected. Paste article text that includes verifiable statements.")
        result.update({"status": "preliminary", "preliminary_label": "Unable to Assess", "evidence_status": "Evidence search pending.", "timings_ms": {"claim_extraction": extraction_ms, "total": round((time.perf_counter() - total_started) * 1000, 2)}})
        return result

    model_started = time.perf_counter()
    models = model if model is not None else load_model()
    model_load_ms = round((time.perf_counter() - model_started) * 1000, 2)
    package = models.get(language) if isinstance(models, dict) else models
    article_estimator = package.get("article_estimator") if language == "en" and isinstance(package, dict) else None
    article_thresholds = tuple(package.get("article_thresholds", (0.45, 0.55))) if isinstance(package, dict) else (0.45, 0.55)
    if isinstance(package, dict) and "estimator" in package:
        estimator = package["estimator"]
        thresholds = tuple(package.get("thresholds", (0.35, 0.65)))
    else:
        estimator = package
        thresholds = (0.35, 0.65)
    # The current English full-article corpus contains demonstrably noisy labels
    # (including wire reports marked FAKE). Keep both English classifiers out of
    # user-facing decisions until a cleaner, independently audited corpus is used.
    english_model_paused = language == "en" and os.getenv("MISINFO_ENABLE_UNVALIDATED_ENGLISH", "").strip().lower() not in {"1", "true", "yes"}
    if english_model_paused:
        article_estimator = None
        estimator = None

    # The English article model is trained on complete articles. Do not feed it
    # isolated claim sentences or aggregate contradictory claim predictions.
    if article_estimator is not None:
        try:
            inference_started = time.perf_counter()
            row = article_estimator.predict_proba([normalized])[0]
            classes = list(article_estimator.classes_)
            p_genuine = float(row[classes.index("genuine")])
            misleading_cutoff, genuine_cutoff = article_thresholds
            label = "Likely Genuine" if p_genuine >= genuine_cutoff else "Potentially Misleading" if p_genuine <= misleading_cutoff else "Unable to Assess"
            score = round(max(p_genuine, 1 - p_genuine), 4)
            inference_ms = round((time.perf_counter() - inference_started) * 1000, 2)
            predictions = [{"claim": claim, "classifier_label": label, "classifier_score": score,
                            "class_scores": {"Likely Genuine": round(p_genuine, 4), "Potentially Misleading": round(1 - p_genuine, 4)}} for claim in claims]
            return {
                "status": "preliminary", "label": "Unable to Assess", "preliminary_label": label,
                "classifier_label": label, "confidence": None, "classifier_score": score,
                "class_scores": {"Likely Genuine": round(p_genuine, 4), "Potentially Misleading": round(1 - p_genuine, 4)},
                "language": language,
                "explanation": "Preliminary whole-article model screening only. This score is not a probability of truth; fact-check evidence is still being retrieved.",
                "evidence_status": "Evidence search pending.", "sources": [], "candidate_claims": claims,
                "claims": predictions,
                "extraction_summary": {"selected_claims": len(claims), "excluded_statements": excluded, "method": "rule-based factual-statement screening; verify extracted text and metadata"},
                "model_available": True, "model_scope": "whole_article",
                "evidence_summary": {"matching_records": 0, "strong_matches": 0, "search_errors": [], "pending": True},
                "timings_ms": {"claim_extraction": extraction_ms, "model_load": model_load_ms, "classifier_inference": inference_ms, "total": round((time.perf_counter() - total_started) * 1000, 2)},
            }
        except (AttributeError, ValueError, TypeError, IndexError):
            logger.exception("English article-model preview failed")
            return _unable("The installed English article model could not complete its prediction.")

    predictions: list[dict[str, Any]] = []
    inference_ms = 0.0
    if estimator is not None:
        try:
            inference_started = time.perf_counter()
            probabilities = estimator.predict_proba(claims)
            classes = list(estimator.classes_)
            true_index = classes.index("genuine")
            misleading_cutoff, genuine_cutoff = thresholds
            for claim_detail, row in zip(claim_details, probabilities):
                p_genuine = float(row[true_index])
                classifier_label = "Likely Genuine" if p_genuine >= genuine_cutoff else "Potentially Misleading" if p_genuine <= misleading_cutoff else "Unable to Assess"
                predictions.append({"claim": claim_detail["text"], "classifier_label": classifier_label, "classifier_score": round(max(p_genuine, 1 - p_genuine), 4), "class_scores": {"Likely Genuine": round(p_genuine, 4), "Potentially Misleading": round(1 - p_genuine, 4)}})
            inference_ms = round((time.perf_counter() - inference_started) * 1000, 2)
        except (AttributeError, ValueError, TypeError, IndexError):
            logger.exception("Model preview prediction failed")
            return _unable("The installed model could not complete its prediction.")
    else:
        predictions = [{"claim": claim, "classifier_label": "Not trained for Hindi" if language == "hi" else "Unavailable", "classifier_score": None, "class_scores": {}} for claim in claims]

    labels = [item["classifier_label"] for item in predictions]
    classifier_label, classifier_score = _classifier_summary(labels, predictions)
    preliminary_label = classifier_label if classifier_label in LABELS else "Unable to Assess"
    return {
        "status": "preliminary", "label": "Unable to Assess", "preliminary_label": preliminary_label,
        "classifier_label": classifier_label, "confidence": None, "classifier_score": classifier_score,
        "class_scores": {}, "language": language,
        "explanation": "Preliminary model signal only. Fact-check evidence is still being retrieved; this is not the final assessment.",
        "evidence_status": "Evidence search pending.", "sources": [], "candidate_claims": claims,
        "claims": predictions,
        "extraction_summary": {"selected_claims": len(claims), "excluded_statements": excluded, "method": "rule-based factual-statement screening; verify extracted text and metadata"},
        "model_available": estimator is not None,
        "model_scope": "evidence_only" if english_model_paused else "claim_level",
        "model_status": "paused_noisy_training_labels" if english_model_paused else "available",
        "evidence_summary": {"matching_records": 0, "strong_matches": 0, "search_errors": [], "pending": True},
        "timings_ms": {"claim_extraction": extraction_ms, "model_load": model_load_ms, "classifier_inference": inference_ms, "total": round((time.perf_counter() - total_started) * 1000, 2)},
    }


def analyze(text: str, language: str = "en", model: Any | None = None) -> dict[str, Any]:
    total_started = time.perf_counter()
    if language not in {"en", "hi"}:
        return _unable("Choose English or Hindi.")
    normalized = _normalize(text)
    if len(normalized) < 80:
        return _unable("Add at least 80 characters of article text for screening.")
    extraction_started = time.perf_counter()
    claim_details, excluded_statements = extract_article_claims(normalized)
    extraction_ms = round((time.perf_counter() - extraction_started) * 1000, 2)
    claims = [item["text"] for item in claim_details]
    if not claims:
        return _unable("No sufficiently factual claim sentences were detected. Paste article text that includes verifiable statements.")

    model_started = time.perf_counter()
    models = model if model is not None else load_model()
    model_load_ms = round((time.perf_counter() - model_started) * 1000, 2)
    package = models.get(language) if isinstance(models, dict) else models
    article_estimator = package.get("article_estimator") if language == "en" and isinstance(package, dict) else None
    article_thresholds = tuple(package.get("article_thresholds", (0.45, 0.55))) if isinstance(package, dict) else (0.45, 0.55)
    if isinstance(package, dict) and "estimator" in package:
        estimator = package["estimator"]
        thresholds = tuple(package.get("thresholds", (0.35, 0.65)))
    else:
        estimator = package
        thresholds = (0.35, 0.65)
    english_model_paused = language == "en" and os.getenv("MISINFO_ENABLE_UNVALIDATED_ENGLISH", "").strip().lower() not in {"1", "true", "yes"}
    if english_model_paused:
        article_estimator = None
        estimator = None

    assessments = []
    inference_ms = 0.0
    evidence_ms = 0.0
    article_prediction: dict[str, Any] | None = None
    if article_estimator is not None:
        try:
            inference_started = time.perf_counter()
            row = article_estimator.predict_proba([normalized])[0]
            classes = list(article_estimator.classes_)
            p_genuine = float(row[classes.index("genuine")])
            low, high = article_thresholds
            article_label = "Likely Genuine" if p_genuine >= high else "Potentially Misleading" if p_genuine <= low else "Unable to Assess"
            article_prediction = {"label": article_label, "score": round(max(p_genuine, 1 - p_genuine), 4),
                                  "class_scores": {"Likely Genuine": round(p_genuine, 4), "Potentially Misleading": round(1 - p_genuine, 4)}}
            inference_ms = round((time.perf_counter() - inference_started) * 1000, 2)
        except (AttributeError, ValueError, TypeError, IndexError):
            logger.exception("English article model prediction failed")
            return _unable("The installed English article model could not complete its prediction.")
    if language == "en" and article_prediction is not None:
        evidence_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(len(claims), 4), thread_name_prefix="evidence") as pool:
            evidence_results = list(pool.map(lambda claim: retrieve_evidence(claim, language), claims))
        evidence_ms = round((time.perf_counter() - evidence_started) * 1000, 2)
        for claim_detail, evidence in zip(claim_details, evidence_results):
            evidence_assessment = _strong_evidence(evidence)
            # Claim-level evidence is shown independently; the model predicts the full article.
            assessments.append({"claim": claim_detail["text"], "label": "Unable to Assess", "classifier_label": "Article-level model", "classifier_score": None, "class_scores": {}, "extraction": {"kind": "factual_candidate", "metadata": {key: claim_detail[key] for key in ("entities", "dates", "quantities")}}, "evidence_assessment": evidence_assessment, "decision_reason": "The classifier scores the full article; this individual claim is assessed only against retrieved fact-check evidence.", "evidence": evidence})
    elif language in {"en", "hi"} and estimator is not None:
        try:
            inference_started = time.perf_counter()
            probabilities = estimator.predict_proba(claims)
            inference_ms = round((time.perf_counter() - inference_started) * 1000, 2)
            classes = list(estimator.classes_)
            true_index = classes.index("genuine")
            misleading_cutoff, genuine_cutoff = thresholds
            evidence_started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=min(len(claims), 4), thread_name_prefix="evidence") as pool:
                evidence_results = list(pool.map(lambda claim: retrieve_evidence(claim, language), claims))
            evidence_ms = round((time.perf_counter() - evidence_started) * 1000, 2)
            for claim_detail, row, evidence in zip(claim_details, probabilities, evidence_results):
                claim = claim_detail["text"]
                p_genuine = float(row[true_index])
                classifier_label = "Likely Genuine" if p_genuine >= genuine_cutoff else "Potentially Misleading" if p_genuine <= misleading_cutoff else "Unable to Assess"
                evidence_assessment = _strong_evidence(evidence)
                claim_label, decision = _combine_claim(classifier_label, evidence_assessment["verdict"])
                assessments.append({"claim": claim, "label": claim_label, "classifier_label": classifier_label, "classifier_score": round(max(p_genuine, 1 - p_genuine), 4), "class_scores": {"Likely Genuine": round(p_genuine, 4), "Potentially Misleading": round(1 - p_genuine, 4)}, "extraction": {"kind": "factual_candidate", "metadata": {key: claim_detail[key] for key in ("entities", "dates", "quantities")}}, "evidence_assessment": evidence_assessment, "decision_reason": decision, "evidence": evidence})
        except (AttributeError, ValueError, TypeError, IndexError):
            logger.exception("English model prediction failed")
            return _unable("The installed model could not complete its English prediction. Check the model version and retrain it.")
    else:
        evidence_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(len(claims), 4), thread_name_prefix="evidence") as pool:
            evidence_results = list(pool.map(lambda claim: retrieve_evidence(claim, language), claims))
        evidence_ms = round((time.perf_counter() - evidence_started) * 1000, 2)
        for claim_detail, evidence in zip(claim_details, evidence_results):
            claim = claim_detail["text"]
            evidence_assessment = _strong_evidence(evidence)
            claim_label, decision = _combine_claim("Unable to Assess", evidence_assessment["verdict"])
            assessments.append({"claim": claim, "label": claim_label, "classifier_label": "Unavailable" if language == "en" else "Not trained for Hindi", "classifier_score": None, "class_scores": {}, "extraction": {"kind": "factual_candidate", "metadata": {key: claim_detail[key] for key in ("entities", "dates", "quantities")}}, "evidence_assessment": evidence_assessment, "decision_reason": decision, "evidence": evidence})

    clear_misleading = [item for item in assessments if item["label"] == "Potentially Misleading"]
    clear_genuine = [item for item in assessments if item["label"] == "Likely Genuine"]
    evidence_verdicts = [item["evidence_assessment"]["verdict"] for item in assessments]
    conflicts = [item for item in assessments if item["evidence_assessment"]["verdict"] == "conflicting"]
    if article_prediction is not None:
        article_label = article_prediction["label"]
        contradicts_genuine = article_label == "Likely Genuine" and "refutes" in evidence_verdicts
        contradicts_misleading = article_label == "Potentially Misleading" and "supports" in evidence_verdicts
        if conflicts or contradicts_genuine or contradicts_misleading:
            overall = "Unable to Assess"
            explanation = "The whole-article model and relevant fact-check evidence conflict, or the retrieved ratings conflict. Review the cited records before drawing a conclusion."
        elif article_label == "Unable to Assess":
            overall = article_label
            explanation = "The whole-article model score falls between its validation-selected decision thresholds, and retrieved evidence does not resolve the result."
        else:
            overall = article_label
            explanation = ("The whole-article English screening model leans genuine; no strong contradictory fact-check evidence was retrieved. "
                           "This is a dataset-based screening result, not confirmation that every statement is true." if overall == "Likely Genuine" else
                           "The whole-article English screening model flags this article as potentially misleading; no strong supporting fact-check evidence was retrieved. Review the individual claims and sources before concluding it is false.")
    elif clear_misleading:
        overall = "Potentially Misleading"
        explanation = "At least one claim is supported by a close negative fact-check record. Review that claim and its source; unverified claims are not treated as misleading solely from the model score."
    elif conflicts:
        overall = "Unable to Assess"
        explanation = "The classifier and relevant fact-check records conflict, or the retrieved ratings conflict. Review the sources before drawing a conclusion."
    elif clear_genuine and len(clear_genuine) > len(assessments) / 2:
        overall = "Likely Genuine"
        unresolved = len(assessments) - len(clear_genuine)
        explanation = (f"A majority ({len(clear_genuine)} of {len(assessments)}) of the extracted claims lean genuine, with no strong contradictory fact-check evidence. "
                       f"{unresolved} claim(s) remain unverified. This is a screening result, not a guarantee that every detail is accurate.")
    else:
        overall = "Unable to Assess"
        explanation = "The available model and fact-check search results do not support a cautious article-level assessment."

    classifier_labels = [item["classifier_label"] for item in assessments]
    all_matches = [match for item in assessments for match in item["evidence"].get("matches", [])]
    search_errors = [error for item in assessments for error in [item["evidence"].get("local_error"), item["evidence"].get("google_error") if item["evidence"].get("google_status") == "error" else None] if error]
    search_states = [item["evidence"]["status"] for item in assessments]
    if search_errors:
        summary_status = "One or more evidence searches failed. See claim details for errors; this is not the same as a successful search with no matches."
    elif all_matches:
        summary_status = "Fact-check records retrieved where available; relevance and verdict are displayed separately."
    elif search_states and all(state == "local_empty_live_off" for state in search_states):
        summary_status = "No sufficiently relevant local fact-check record was found; live Google search is not configured."
    else:
        summary_status = "No sufficiently relevant fact-check record was found for the detected claims. No match is not a verdict."
    if language == "hi" and estimator is None:
        overall_classifier, overall_score = "Not trained for Hindi", None
    elif article_prediction is not None:
        overall_classifier, overall_score = article_prediction["label"], article_prediction["score"]
    else:
        overall_classifier, overall_score = _classifier_summary(classifier_labels, assessments)
    return {
        "label": overall, "classifier_label": overall_classifier,
        "confidence": None, "classifier_score": overall_score, "class_scores": article_prediction["class_scores"] if article_prediction else {},
        "model_scope": "whole_article" if article_prediction else "evidence_only" if english_model_paused else "claim_level",
        "model_status": "paused_noisy_training_labels" if english_model_paused else "available",
        "language": language, "explanation": explanation,
        "evidence_status": summary_status, "sources": all_matches,
        "candidate_claims": claims, "claims": assessments,
        "extraction_summary": {"selected_claims": len(claims), "excluded_statements": excluded_statements, "method": "rule-based factual-statement screening; verify extracted text and metadata"},
        "model_available": article_prediction is not None or estimator is not None,
        "evidence_summary": {"matching_records": len(all_matches), "strong_matches": sum(item["evidence_assessment"]["strong_match_count"] for item in assessments), "search_errors": search_errors, "claim_search_statuses": search_states},
        "timings_ms": {"claim_extraction": extraction_ms, "model_load": model_load_ms, "classifier_inference": inference_ms, "evidence_retrieval": evidence_ms, "total": round((time.perf_counter() - total_started) * 1000, 2)},
    }


def _unable(reason: str) -> dict[str, Any]:
    return {"label": "Unable to Assess", "classifier_label": "Unavailable", "confidence": None, "classifier_score": None, "class_scores": {}, "language": None, "explanation": reason, "evidence_status": "No assessment", "sources": [], "candidate_claims": [], "claims": [], "extraction_summary": {"selected_claims": 0, "excluded_statements": [], "method": "rule-based factual-statement screening"}, "model_available": False, "evidence_summary": {"matching_records": 0, "strong_matches": 0, "search_errors": []}}
