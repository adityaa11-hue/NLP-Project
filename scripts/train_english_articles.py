"""Train and evaluate the English full-article screening model."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
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
LABELS = ("genuine", "misleading")
CONFIDENCE_GRID = tuple(np.round(np.arange(0.50, 0.751, 0.025), 3))
TARGET_SELECTIVE_ACCURACY = 0.92


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", text)
    return " ".join(text.split())


def canonical_key(text: str) -> bytes:
    text = re.sub(r"[^\w\s]", " ", normalize_text(text).casefold())
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).digest()


def read_split(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("language", "").strip().lower() != "en":
                continue
            text = normalize_text(row.get("text", ""))
            label = row.get("label", "").strip().lower()
            if text and label in LABELS:
                texts.append(text)
                labels.append(label)
    if not texts:
        raise ValueError(f"No labeled English articles found in {path}")
    return texts, labels


def make_vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        n_features=2**14, ngram_range=(1, 2), alternate_sign=False,
        norm="l2", dtype=np.float32, strip_accents="unicode",
    )


def probs_genuine(classifier: LogisticRegression, matrix) -> np.ndarray:
    index = list(classifier.classes_).index("genuine")
    return classifier.predict_proba(matrix)[:, index]


def choose_cutoff(probabilities: np.ndarray, truth: list[str]) -> tuple[float, float, float]:
    chosen: tuple[float, float, float] | None = None
    for cutoff in CONFIDENCE_GRID:
        accepted = np.maximum(probabilities, 1.0 - probabilities) >= cutoff
        if not accepted.any():
            continue
        predicted = np.where(probabilities[accepted] >= 0.5, "genuine", "misleading")
        accuracy = float(accuracy_score(np.asarray(truth)[accepted], predicted))
        coverage = float(accepted.mean())
        if accuracy >= TARGET_SELECTIVE_ACCURACY and (chosen is None or coverage > chosen[2]):
            chosen = (float(cutoff), accuracy, coverage)
    if chosen is None:
        predicted = np.where(probabilities >= 0.5, "genuine", "misleading")
        return 0.50, float(accuracy_score(truth, predicted)), 1.0
    return chosen


def predict_in_batches(vectorizer, classifier, texts: list[str], batch_size: int = 128) -> np.ndarray:
    values: list[float] = []
    for start in range(0, len(texts), batch_size):
        matrix = vectorizer.transform(texts[start:start + batch_size])
        values.extend(probs_genuine(classifier, matrix).tolist())
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "misinformation_tfidf.joblib")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "english_article_model_evaluation.md")
    args = parser.parse_args()

    train_texts, train_labels = read_split(args.train)
    test_texts, test_labels = read_split(args.test)
    test_keys = {canonical_key(text) for text in test_texts}
    keep = [canonical_key(text) not in test_keys for text in train_texts]
    duplicate_count = len(keep) - sum(keep)
    train_texts = [text for text, retain in zip(train_texts, keep) if retain]
    train_labels = [label for label, retain in zip(train_labels, keep) if retain]

    train_indices, validation_indices = train_test_split(
        np.arange(len(train_texts)), test_size=0.20, random_state=2026, stratify=train_labels,
    )
    vectorizer = make_vectorizer()
    train_matrix = vectorizer.transform([train_texts[i] for i in train_indices])
    validation_matrix = vectorizer.transform([train_texts[i] for i in validation_indices])
    validation_labels = [train_labels[i] for i in validation_indices]

    best: tuple[float, float, float, float] | None = None
    candidate_rows: list[str] = []
    for c_value in (0.25, 0.5, 1.0, 2.0):
        classifier = LogisticRegression(C=c_value, max_iter=150, solver="liblinear")
        classifier.fit(train_matrix, [train_labels[i] for i in train_indices])
        probabilities = probs_genuine(classifier, validation_matrix)
        binary = np.where(probabilities >= 0.5, "genuine", "misleading")
        macro = float(f1_score(validation_labels, binary, labels=LABELS, average="macro", zero_division=0))
        cutoff, selective_accuracy, coverage = choose_cutoff(probabilities, validation_labels)
        candidate_rows.append(f"| {c_value:g} | {macro:.4f} | {cutoff:.3f} | {selective_accuracy:.4f} | {coverage:.4f} |")
        print(f"Validation C={c_value:g}: accuracy={accuracy_score(validation_labels, binary):.4f}; macro-F1={macro:.4f}; cutoff={cutoff:.3f}; selective accuracy={selective_accuracy:.4f}; coverage={coverage:.4f}", flush=True)
        if best is None or macro > best[1]:
            best = (c_value, macro, cutoff, coverage)
        del classifier, probabilities, binary
        gc.collect()
    if best is None:
        raise RuntimeError("No classifier candidate completed")

    best_c, validation_macro, cutoff, validation_coverage = best
    selected_classifier = LogisticRegression(C=best_c, max_iter=150, solver="liblinear")
    selected_classifier.fit(train_matrix, [train_labels[i] for i in train_indices])
    validation_probabilities = probs_genuine(selected_classifier, validation_matrix)
    _, validation_selective_accuracy, _ = choose_cutoff(validation_probabilities, validation_labels)
    del train_matrix, validation_matrix, selected_classifier, validation_probabilities
    gc.collect()

    final_vectorizer = make_vectorizer()
    full_train_matrix = final_vectorizer.transform(train_texts)
    final_classifier = LogisticRegression(C=best_c, max_iter=150, solver="liblinear")
    final_classifier.fit(full_train_matrix, train_labels)
    del full_train_matrix
    gc.collect()

    test_probabilities = predict_in_batches(final_vectorizer, final_classifier, test_texts)
    binary_predictions = np.where(test_probabilities >= 0.5, "genuine", "misleading")
    low, high = 1.0 - cutoff, cutoff
    three_way = ["Potentially Misleading" if p <= low else "Likely Genuine" if p >= high else "Unable to Assess" for p in test_probabilities]
    accepted = np.asarray([label != "Unable to Assess" for label in three_way])
    selective_accuracy = float(accuracy_score(np.asarray(test_labels)[accepted], np.asarray(binary_predictions)[accepted]))

    pipeline = Pipeline([("features", final_vectorizer), ("classifier", final_classifier)])
    package = joblib.load(args.out)
    if not isinstance(package, dict) or not isinstance(package.get("en"), dict):
        raise ValueError(f"Expected the existing English claim-model package at {args.out}")
    package["en"].update({
        "article_estimator": pipeline,
        "article_thresholds": (low, high),
        "article_model_name": f"HashingVectorizer word 1-2 grams + LogisticRegression C={best_c:g}",
    })
    package.setdefault("metadata", {})["english_article_model"] = {
        "training_rows": len(train_texts),
        "excluded_exact_train_test_duplicates": duplicate_count,
        "validation_macro_f1": validation_macro,
        "validation_selective_accuracy": validation_selective_accuracy,
        "validation_coverage": validation_coverage,
        "confidence_cutoff": cutoff,
        "test_accuracy_binary": float(accuracy_score(test_labels, binary_predictions)),
        "test_selective_accuracy": selective_accuracy,
        "test_coverage": float(accepted.mean()),
        "test_rows": len(test_texts),
        "dataset": "fauxNeuz/RealFakeNews (educational English article corpus)",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, args.out)

    report = [
        "# English full-article model evaluation", "",
        "The article-level model is trained on full article text from the fauxNeuz RealFakeNews educational corpus. This is a binary real/fake screening model, not a claim verifier. The application's X-FACT evidence retrieval remains an independent signal.", "",
        "Dataset card: https://huggingface.co/datasets/fauxNeuz/RealFakeNews. The card lists CC BY-NC-SA 4.0, educational/non-commercial scope, and possible topic/source/label bias.", "",
        "## Split and model selection", "",
        f"- English train rows: {len(train_texts):,} (genuine={train_labels.count('genuine'):,}, misleading={train_labels.count('misleading'):,}).",
        f"- Exact normalized train/test duplicates excluded from training: {duplicate_count:,}.",
        f"- English test rows: {len(test_texts):,}; exact canonical duplicates within test: none.",
        "- Candidate C was selected on a deterministic stratified 80/20 split of training data by validation macro-F1. Test labels were not used for candidate or threshold selection.",
        f"- Selected C={best_c:g}; validation macro-F1={validation_macro:.4f}; validation selective accuracy={validation_selective_accuracy:.4f}; validation coverage={validation_coverage:.4f}; confidence cutoff={cutoff:.3f}.", "",
        "## Held-out test", "", "### Binary decision at 0.50", "", "```text",
        f"Accuracy: {accuracy_score(test_labels, binary_predictions):.4f}",
        f"Macro F1: {f1_score(test_labels, binary_predictions, labels=LABELS, average='macro', zero_division=0):.4f}",
        classification_report(test_labels, binary_predictions, labels=LABELS, zero_division=0),
        "Confusion matrix (true rows; genuine, misleading columns):",
        str(confusion_matrix(test_labels, binary_predictions, labels=LABELS)), "```", "",
        f"### Three-way output with confidence cutoff {cutoff:.3f}", "",
        f"- Classified coverage: {accepted.mean():.4f} ({int(accepted.sum()):,}/{len(test_texts):,}).",
        f"- Accuracy among classified articles: {selective_accuracy:.4f}.",
        f"- Unable to Assess abstentions: {len(test_texts) - int(accepted.sum()):,}.", "",
        "## Limitations", "",
        "These metrics apply to this held-out benchmark split only. They do not establish accuracy across publishers, topics, dates, or current news. The source labels may be noisy, and the test set comes from the same dataset lineage as training. Treat the result as a model screening signal, not truth probability or verified fact-check. Independent publisher/time-disjoint evaluation is still needed.", "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report), encoding="utf-8")
    print(f"Held-out binary accuracy={accuracy_score(test_labels, binary_predictions):.4f}; macro-F1={f1_score(test_labels, binary_predictions, labels=LABELS, average='macro', zero_division=0):.4f}")
    print(f"Held-out classified coverage={accepted.mean():.4f}; selective accuracy={selective_accuracy:.4f}; cutoff={cutoff:.3f}")
    print(f"Saved model: {args.out}\nSaved report: {args.report}")


if __name__ == "__main__":
    main()
