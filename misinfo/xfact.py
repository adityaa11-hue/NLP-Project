"""Robust reader for X-FACT TSVs with unescaped tabs in some text fields."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

LANGUAGES = {"en", "hi"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:T|$)")
URL_RE = re.compile(r"^https?://", re.I)
XFACT_FIELDS = ["language", "site", "claim", "label", "claimant", "claimDate", "reviewDate", "link_1", "evidence_1"]


def _decode_for_language(value: str, language: str) -> str:
    if language == "hi":
        try:
            return value.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return value
    try:
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def read_xfact(path: str | Path, languages: set[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    allowed = languages or LANGUAGES
    rows: list[dict] = []
    skipped = 0
    with path.open("r", encoding="latin-1", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quotechar='"')
        header = next(reader, [])
        if header[:2] != ["language", "site"]:
            raise ValueError(f"Unexpected X-FACT header in {path.name}: {header[:3]}")
        for row in reader:
            if len(row) < len(header):
                skipped += 1
                continue
            language = row[0].strip().lower()
            if language not in allowed:
                continue
            date_positions = [i for i, part in enumerate(row) if DATE_RE.match(part.strip())]
            if len(date_positions) < 2:
                skipped += 1
                continue
            claim_date_idx, review_date_idx = date_positions[-2:]
            if review_date_idx + 2 >= len(row):
                skipped += 1
                continue
            claim = " ".join(part.strip() for part in row[review_date_idx + 2:-1] if part.strip())
            label = row[-1].strip()
            claimant = row[review_date_idx + 1].strip() if review_date_idx + 1 < len(row) else ""
            segments = row[2:claim_date_idx]
            urls = [part.strip() for part in segments if URL_RE.match(part.strip())]
            evidence_parts = [part.strip() for part in segments if part.strip() and not URL_RE.match(part.strip()) and part.strip() not in {"none", "NO_EVIDENCE", "<DUMMY_EVIDENCE>"}]
            claim = _decode_for_language(claim, language)
            rows.append({
                "language": language,
                "site": _decode_for_language(row[1].strip(), language),
                "claim": " ".join(claim.split()),
                "label": label,
                "claimant": _decode_for_language(claimant, language),
                "claimDate": row[claim_date_idx].strip(),
                "reviewDate": row[review_date_idx].strip(),
                "link_1": urls[0] if urls else "",
                "links": urls,
                "evidence_1": _decode_for_language(" ".join(evidence_parts), language),
            })
    frame = pd.DataFrame(rows, columns=[*XFACT_FIELDS, "links"])
    frame.attrs["malformed_rows_skipped"] = skipped
    frame.attrs["source_path"] = str(path)
    return frame
