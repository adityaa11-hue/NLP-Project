"""Train and audit the Hindi article baseline from Zenodo CC BY 4.0 data.

This is a source-folder-labeled benchmark. It is useful as an experimental
baseline but does not establish cross-publisher or real-world accuracy.
"""
from __future__ import annotations

import hashlib
import re
import random
import unicodedata
import zipfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "hindi_fr_news.zip"
MODEL = ROOT / "models" / "hindi_misinfo_experimental.joblib"
REPORT = ROOT / "outputs" / "hindi_model_evaluation.md"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def load_records() -> tuple[list[str], list[str]]:
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Download the CC BY 4.0 Hindi archive to {ARCHIVE}")
    rows: dict[str, str] = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in archive.namelist():
            if not name.endswith(".txt") or "/._" in name:
                continue
            if name.startswith("Hindi_real_news/"):
                label = "genuine"
            elif name.startswith("Hindi_fake_news/"):
                label = "misleading"
            else:
                continue
            try:
                text = normalize(archive.read(name).decode("utf-8", errors="replace"))
            except (KeyError, OSError):
                continue
            # Stable content deduplication prevents exact copies crossing splits.
            digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
            if len(text) >= 40:
                rows.setdefault(digest, (text, label))
    records = list(rows.values())
    # Keep the classroom-sized CPU run tractable while preserving both classes.
    rng = random.Random(73)
    sampled = []
    for label in ("genuine", "misleading"):
        group = [row for row in records if row[1] == label]
        rng.shuffle(group)
        sampled.extend(group[:1200])
    records = sampled
    texts = [row[0] for row in records]
    labels = [row[1] for row in records]
    if len(set(labels)) != 2:
        raise ValueError("Hindi archive must contain both real and fake classes.")
    return texts, labels


def main() -> None:
    texts, labels = load_records()
    x_train, x_temp, y_train, y_temp = train_test_split(texts, labels, test_size=.30, random_state=42, stratify=labels)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=.50, random_state=43, stratify=y_temp)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=40_000, sublinear_tf=True, dtype=np.float32)),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=300, C=0.8)),
    ])
    model.fit(x_train, y_train)
    misleading_index = list(model.classes_).index("misleading")
    val_p = model.predict_proba(x_val)[:, misleading_index]
    # Conservative selective cutoffs are selected on validation only.
    selected = None
    grid = np.arange(.50, .951, .025)
    for low in grid:
        for high in grid:
            if high <= low:
                continue
            preds = ["Potentially Misleading" if p >= high else "Likely Genuine" if p <= low else "Unable to Assess" for p in val_p]
            truth = ["Potentially Misleading" if y == "misleading" else "Likely Genuine" for y in y_val]
            macro = f1_score(truth, preds, labels=["Likely Genuine", "Potentially Misleading", "Unable to Assess"], average="macro", zero_division=0)
            coverage = sum(p != "Unable to Assess" for p in preds) / len(preds)
            # Promote trustworthy fake detection: require >=.80 validation precision on classified fake items.
            selected_preds = [(p, y) for p, y in zip(preds, truth) if p == "Potentially Misleading"]
            precision = sum(p == y for p, y in selected_preds) / max(len(selected_preds), 1)
            if precision < .80:
                continue
            score = (macro, coverage, -(high - low))
            if selected is None or score > selected[0]:
                selected = (score, (float(low), float(high)), preds, precision)
    if selected is None:
        # No dependable threshold found: abstain for both classes until reviewed.
        thresholds = (.0, 1.0)
        val_preds = ["Unable to Assess"] * len(y_val)
        val_fake_precision = None
    else:
        _, thresholds, val_preds, val_fake_precision = selected

    test_p = model.predict_proba(x_test)[:, misleading_index]
    lo, hi = thresholds
    test_preds = ["Potentially Misleading" if p >= hi else "Likely Genuine" if p <= lo else "Unable to Assess" for p in test_p]
    test_truth = ["Potentially Misleading" if y == "misleading" else "Likely Genuine" for y in y_test]
    test_fake = [(p, y) for p, y in zip(test_preds, test_truth) if p == "Potentially Misleading"]
    test_precision = sum(p == y for p, y in test_fake) / max(len(test_fake), 1)
    test_coverage = sum(p != "Unable to Assess" for p in test_preds) / len(test_preds)

    # Refit on train + validation after selecting the confidence band.
    final_model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=40_000, sublinear_tf=True, dtype=np.float32)),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=300, C=0.8)),
    ])
    final_model.fit(x_train + x_val, y_train + y_val)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"estimator": final_model, "thresholds": thresholds, "thresholds_for": "misleading", "name": "Hindi character TF-IDF + Logistic Regression", "dataset": "Zenodo record 11408513", "experimental": True}, MODEL)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join([
        "# Hindi baseline evaluation", "",
        f"Source: Zenodo record 11408513 (CC BY 4.0); {len(texts)} exact-deduplicated text files.",
        f"Split sizes: train {len(x_train)}, validation {len(x_val)}, test {len(x_test)}.",
        f"Validation-selected thresholds on P(misleading): genuine <= {thresholds[0]:.3f}; misleading >= {thresholds[1]:.3f}; otherwise abstain.",
        f"Validation fake precision at selected threshold: {val_fake_precision if val_fake_precision is not None else 'no threshold met 0.80 precision'}.",
        f"Held-out fake precision: {test_precision:.4f}; coverage: {test_coverage:.4f}.", "",
        "## Held-out test report", "", "```text", classification_report(test_truth, test_preds, labels=["Likely Genuine", "Potentially Misleading", "Unable to Assess"], zero_division=0), "```", "",
        "## Limitations", "",
        "This dataset is organized into real-news and fake-news folders and the sources may differ by class. Files are randomly split after exact deduplication; source and near-duplicate leakage may remain. The report is an in-dataset baseline, not a real-world accuracy estimate. The model must be shown as experimental and uses a validation-selected abstain band on the misleading-class probability.",
    ]), encoding="utf-8")
    print(f"Rows={len(texts)} split train/val/test={len(x_train)}/{len(x_val)}/{len(x_test)}")
    print(f"Thresholds={thresholds}; held-out fake precision={test_precision:.3f}; coverage={test_coverage:.3f}")
    print(f"Saved {MODEL} and {REPORT}")


if __name__ == "__main__":
    main()
