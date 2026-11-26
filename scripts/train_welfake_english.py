"""Train the English article screening model on the original WELFake release.

WELFake is a benchmark, not a source of definitive truth. Its original paper
reports 35,028 real and 37,106 fake articles. This script checks that the CSV's
binary labels match those counts and preserves the official source split order
only as a documented benchmark limitation; WELFake has no article dates or
publisher field suitable for a true source/time-disjoint test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_URL = "https://zenodo.org/api/records/4561253/files/WELFake_Dataset.csv/content"
DATA_PATH = ROOT / "data" / "welfake" / "WELFake_Dataset.csv"
MODEL_PATH = ROOT / "models" / "misinformation_tfidf.joblib"
REPORT_PATH = ROOT / "outputs" / "welfake_english_evaluation.md"
LABELS = ("genuine", "misleading")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", text)
    return " ".join(text.split())


def canonical(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", normalize(text).casefold()).split())


def download_data(path: Path) -> None:
    if path.exists() and path.stat().st_size > 200_000_000:
        return
    from urllib.request import Request, urlopen

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    request = Request(DATA_URL, headers={"User-Agent": "MisinformationShield-education/1.0"})
    with urlopen(request, timeout=120) as response, temp.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    if temp.stat().st_size < 200_000_000:
        temp.unlink(missing_ok=True)
        raise RuntimeError("Downloaded WELFake file is incomplete or unexpectedly small")
    temp.replace(path)


def load_dataset(path: Path) -> tuple[list[str], list[str], dict[str, int]]:
    csv.field_size_limit(10_000_000)
    texts: list[str] = []
    labels: list[str] = []
    counts = {"0": 0, "1": 0}
    by_text: dict[str, tuple[str, str]] = {}
    conflicting: set[str] = set()
    rows_seen = 0
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_seen += 1
            raw_label = str(row.get("label", "")).strip()
            if raw_label not in counts:
                continue
            counts[raw_label] += 1
            title = normalize(row.get("title", ""))
            body = normalize(row.get("text", ""))
            combined = normalize(f"{title}. {body}" if title else body)
            if len(canonical(combined)) < 60:
                continue
            key = hashlib.sha256(canonical(combined).encode("utf-8")).hexdigest()
            # Original release distribution: 0=real (35,028), 1=fake (37,106).
            target = "genuine" if raw_label == "0" else "misleading"
            if key in by_text and by_text[key][0] != target:
                conflicting.add(key)
            else:
                by_text.setdefault(key, (target, combined))
            if rows_seen % 10000 == 0:
                print(f"Audited {rows_seen:,} WELFake rows", flush=True)
    if counts != {"0": 35028, "1": 37106}:
        raise ValueError(f"Label counts do not match the original release; refusing training: {counts}")
    for key, (target, text) in by_text.items():
        if key not in conflicting:
            texts.append(text)
            labels.append(target)
    if not texts:
        raise ValueError("No usable labeled WELFake articles were found")
    print(f"Source rows verified: real(label 0)={counts['0']:,}; fake(label 1)={counts['1']:,}")
    print(f"Usable unique articles={len(texts):,}; conflicting duplicate texts excluded={len(conflicting):,}")
    return texts, labels, counts


def vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        n_features=2**17, ngram_range=(1, 2), alternate_sign=False,
        norm="l2", dtype=np.float32, strip_accents="unicode",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    download_data(args.data)
    texts, labels, source_counts = load_dataset(args.data)

    # Deduplication precedes splitting so exact articles cannot leak between sets.
    indices = np.arange(len(texts))
    train_idx, held_idx = train_test_split(indices, test_size=0.20, random_state=2026, stratify=labels)
    val_idx, test_idx = train_test_split(held_idx, test_size=0.50, random_state=2026, stratify=[labels[i] for i in held_idx])
    vec = vectorizer()
    train_x = vec.transform([texts[i] for i in train_idx])
    val_x = vec.transform([texts[i] for i in val_idx])
    test_x = vec.transform([texts[i] for i in test_idx])
    train_y = [labels[i] for i in train_idx]
    val_y = [labels[i] for i in val_idx]
    test_y = [labels[i] for i in test_idx]

    # Select C by validation macro-F1; keep the test split untouched until final evaluation.
    candidates = []
    for c in (0.25, 0.5, 1.0, 2.0, 4.0):
        model = LogisticRegression(C=c, max_iter=250, solver="liblinear")
        model.fit(train_x, train_y)
        pred = model.predict(val_x)
        macro = f1_score(val_y, pred, labels=LABELS, average="macro", zero_division=0)
        acc = accuracy_score(val_y, pred)
        candidates.append((float(macro), float(acc), c))
        print(f"validation C={c:g}: accuracy={acc:.4f}; macro_f1={macro:.4f}", flush=True)
    _, _, best_c = max(candidates)

    # Report an all-input binary decision. The app can return Unable to Assess
    # only when evidence conflicts or the classifier's decision is exactly tied.
    final_vec = vectorizer()
    all_train_x = final_vec.transform([texts[i] for i in np.concatenate([train_idx, val_idx])])
    all_train_y = [labels[i] for i in np.concatenate([train_idx, val_idx])]
    final_model = LogisticRegression(C=best_c, max_iter=250, solver="liblinear")
    final_model.fit(all_train_x, all_train_y)
    test_pred = final_model.predict(test_x)
    test_prob = final_model.predict_proba(test_x)
    accuracy = accuracy_score(test_y, test_pred)
    macro = f1_score(test_y, test_pred, labels=LABELS, average="macro", zero_division=0)
    print(f"heldout accuracy={accuracy:.4f}; macro_f1={macro:.4f}; rows={len(test_y)}")
    print(classification_report(test_y, test_pred, labels=LABELS, zero_division=0))
    print("confusion matrix true rows genuine,misleading; predicted columns genuine,misleading:")
    print(confusion_matrix(test_y, test_pred, labels=LABELS))

    estimator = Pipeline([("features", final_vec), ("classifier", final_model)])
    package = joblib.load(args.model)
    if not isinstance(package, dict) or not isinstance(package.get("en"), dict):
        raise ValueError(f"Unexpected model package format: {args.model}")
    package["en"].update({
        "article_estimator": estimator,
        "article_thresholds": (0.5, 0.5),
        "article_model_name": f"WELFake HashingVectorizer word 1-2 grams + LogisticRegression C={best_c:g}",
        "article_model_enabled": True,
        "article_training_dataset": "WELFake original Zenodo release 4561253",
    })
    package.setdefault("metadata", {})["english_article_model"] = {
        "dataset": "WELFake original Zenodo release 4561253",
        "license": "CC BY 4.0 (per Zenodo record metadata)",
        "label_mapping": {"0": "genuine", "1": "misleading"},
        "verified_source_counts": source_counts,
        "unique_articles": len(texts),
        "validation_candidates": candidates,
        "selected_c": best_c,
        "heldout_accuracy": float(accuracy),
        "heldout_macro_f1": float(macro),
        "heldout_rows": len(test_y),
        "decision_threshold": 0.5,
        "evaluation_limit": "random, deduplicated split from same source-combined benchmark; publisher/time generalization is not established",
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    temp_model = args.model.with_suffix(args.model.suffix + ".tmp")
    joblib.dump(package, temp_model, compress=3)
    temp_model.replace(args.model)

    report = [
        "# English article classifier: WELFake evaluation", "",
        "Label mapping checked from the original WELFake release counts: label 0 is real (35,028 rows) and label 1 is fake (37,106 rows). The random test split is deduplicated but shares the benchmark's mixed-source construction; these results do not establish publisher/time-disjoint performance or truth on current news.", "",
        f"- Usable unique articles: {len(texts):,}", f"- Train+validation rows: {len(all_train_y):,}", f"- Held-out test rows: {len(test_y):,}",
        f"- Selected LogisticRegression C: {best_c:g}", f"- Held-out accuracy: {accuracy:.4f}", f"- Held-out macro-F1: {macro:.4f}",
        f"- Confusion matrix (true rows genuine,misleading; predicted columns genuine,misleading): `{confusion_matrix(test_y, test_pred, labels=LABELS).tolist()}`", "",
        "```text", classification_report(test_y, test_pred, labels=LABELS, zero_division=0), "```", "",
        "The application uses the article model for a binary screening label and retains evidence conflicts as Unable to Assess. The model score is not the probability that an article is true. WELFake was assembled from multiple older datasets and has source and topic limitations; claims require current evidence review.", "",
        "Source record: https://zenodo.org/records/4561253. Original paper: https://doi.org/10.1109/TCSS.2021.3068519.", "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved trained package: {args.model}")
    print(f"Saved evaluation: {args.report}")


if __name__ == "__main__":
    main()
