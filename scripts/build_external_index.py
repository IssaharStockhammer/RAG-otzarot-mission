from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "otzrot_home_assignment_candidate"
OUTPUT_DIR = ROOT / "data" / "external"
SOURCE_TEXT_PATH = OUTPUT_DIR / "sefaria_noda_biyahudah_i_orach_chaim.json"
ALIGNED_INDEX_PATH = OUTPUT_DIR / "aligned_passages.json"
API_REF = "Noda BiYehudah I, Orach Chaim 1-41"
API_BASE = "https://www.sefaria.org/api/texts/"
USER_AGENT = "OtzarotRAGPOC/0.1 (local take-home assignment)"

HEBREW_DIACRITICS_RE = re.compile(r"[\u0591-\u05C7]")
NON_WORD_RE = re.compile(r"[^\w\u05d0-\u05ea\s]")
MULTISPACE_RE = re.compile(r"\s+")
FINAL_LETTER_MAP = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})


def normalize(text: str, *, compact: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = HEBREW_DIACRITICS_RE.sub("", text)
    text = NON_WORD_RE.sub(" ", text)
    text = text.translate(FINAL_LETTER_MAP)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text.replace(" ", "") if compact else text


def character_ngrams(text: str, size: int = 3) -> set[str]:
    if len(text) < size:
        return {text} if text else set()
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [text for item in value for text in flatten_strings(item)]
    return []


def fetch_source_text() -> dict[str, Any]:
    encoded_ref = quote(API_REF, safe=",")
    url = f"{API_BASE}{encoded_ref.replace(' ', '_')}?context=0"
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not download the Sefaria text ({type(error).__name__}: {error})") from error

    if not isinstance(data.get("he"), list) or not data["he"]:
        raise RuntimeError("Sefaria returned no Hebrew text for the requested reference.")

    chapters: list[dict[str, Any]] = []
    for siman, paragraphs in enumerate(data["he"], start=1):
        texts = flatten_strings(paragraphs)
        if texts:
            chapters.append(
                {
                    "siman": siman,
                    "ref": f"Noda BiYehudah I, Orach Chaim {siman}",
                    "paragraphs": texts,
                }
            )

    if not chapters:
        raise RuntimeError("The downloaded Sefaria response contained no usable paragraphs.")

    return {
        "source": "Sefaria",
        "api_ref": data.get("ref", API_REF),
        "he_ref": data.get("heRef"),
        "version_title": data.get("heVersionTitle"),
        "version_source": data.get("heVersionSource"),
        "license": data.get("heLicense"),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "attribution": (
            "Noda BeYehuda Warsaw 1880, Hebrew text via Sefaria; "
            "Sefaria API reports the Hebrew text license as Public Domain."
        ),
        "chapters": chapters,
    }


def load_ocr_pages() -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for source, folder_name in (
        ("hebrewbooks", "ocr_hebrewbooks"),
        ("tesseract", "ocr_tesseract"),
    ):
        folder = DATASET_ROOT / folder_name
        if not folder.exists():
            raise RuntimeError(f"Missing OCR folder: {folder}")
        for path in sorted(folder.glob("page-*.txt")):
            file_lines = path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            lines = [
                (line_number, line.strip())
                for line_number, line in enumerate(
                    file_lines,
                    start=1,
                )
                if line.strip()
            ]
            pages.append(
                {
                    "page": path.stem,
                    "source": source,
                    "lines": lines,
                    "line_count": len(file_lines),
                    "text": " ".join(line for _, line in lines),
                    "normalized": normalize(" ".join(line for _, line in lines), compact=True),
                    "line_windows": [
                        {
                            "line_from": lines[start][0],
                            "line_to": lines[end][0],
                            "text": normalize(
                                " ".join(lines[index][1] for index in range(start, end + 1)),
                                compact=True,
                            ),
                        }
                        for start in range(len(lines))
                        for end in range(start, min(len(lines), start + 4))
                    ],
                }
            )
    if not pages:
        raise RuntimeError(f"No OCR page files found under {DATASET_ROOT}")
    return pages


def make_idf(documents: list[str], ngram_size: int = 3) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for text in documents:
        document_frequency.update(character_ngrams(text, ngram_size))
    count = len(documents)
    return {
        gram: math.log((count + 1) / (frequency + 1)) + 1
        for gram, frequency in document_frequency.items()
    }


def weighted_coverage(query: str, document: str, idf: dict[str, float]) -> float:
    query_grams = character_ngrams(query)
    if not query_grams or not document:
        return 0.0
    document_grams = character_ngrams(document)
    total_weight = sum(idf.get(gram, 1.0) for gram in query_grams)
    matched_weight = sum(idf.get(gram, 1.0) for gram in query_grams & document_grams)
    return matched_weight / total_weight if total_weight else 0.0


def best_line_anchor(
    paragraph: str, page: dict[str, Any], idf: dict[str, float]
) -> dict[str, Any] | None:
    compact_paragraph = normalize(paragraph, compact=True)
    if len(compact_paragraph) < 30:
        return None

    anchor_length = min(140, max(55, len(compact_paragraph) // 5))
    stride = max(anchor_length, len(compact_paragraph) // 5)
    start_positions = {
        0,
        max(0, (len(compact_paragraph) - anchor_length) // 2),
        max(0, len(compact_paragraph) - anchor_length),
    }
    start_positions.update(range(0, max(1, len(compact_paragraph) - anchor_length + 1), stride))

    windows: list[dict[str, Any]] = page["line_windows"]
    best: tuple[float, dict[str, Any]] | None = None
    for start in sorted(start_positions):
        anchor = compact_paragraph[start : start + anchor_length]
        if len(anchor) < 30:
            continue
        for window in windows:
            if len(window["text"]) > anchor_length * 2:
                continue
            score = weighted_coverage(anchor, window["text"], idf)
            if best is None or score > best[0]:
                best = (score, window)

    if best is None:
        return None
    score, window = best
    if page["line_count"]:
        center = (window["line_from"] + window["line_to"]) / (2 * page["line_count"])
        column_hint = "right" if center <= 0.5 else "left"
    else:
        column_hint = "unknown"
    return {
        "score": round(score, 4),
        "line_from": window["line_from"],
        "line_to": window["line_to"],
        "column_hint": column_hint,
        "top_pct": round(
            max(0.0, min(100.0, (window["line_from"] - 1) / page["line_count"] * 100)),
            2,
        )
        if page["line_count"]
        else 0.0,
        "height_pct": round(
            max(
                2.5,
                (window["line_to"] - window["line_from"] + 1)
                / page["line_count"]
                * 100,
            ),
            2,
        )
        if page["line_count"]
        else 0.0,
    }


def alignment_status(score: float, margin: float, paragraph_length: int) -> str:
    if paragraph_length < 45:
        return "unmapped"
    if score >= 0.72 and margin >= 0.04:
        return "high"
    if score >= 0.58 and margin >= 0.05:
        return "medium"
    return "ambiguous"


def align_source(source: dict[str, Any], pages: list[dict[str, Any]]) -> dict[str, Any]:
    idf = make_idf([page["normalized"] for page in pages])
    aligned: list[dict[str, Any]] = []

    for chapter in source["chapters"]:
        for paragraph_number, text in enumerate(chapter["paragraphs"], start=1):
            normalized = normalize(text, compact=True)
            if len(normalized) < 30:
                status = "unmapped"
                candidates: list[dict[str, Any]] = []
            else:
                ranked: list[tuple[float, dict[str, Any]]] = []
                query_grams = character_ngrams(normalized)
                for page in pages:
                    score = weighted_coverage(normalized, page["normalized"], idf)
                    ranked.append((score, page))
                ranked.sort(key=lambda item: item[0], reverse=True)

                best_score = ranked[0][0] if ranked else 0.0
                top_page = ranked[0][1]["page"] if ranked else ""
                second_page_score = max(
                    (
                        score
                        for score, page in ranked
                        if page["page"] != top_page
                    ),
                    default=0.0,
                ) if ranked else 0.0
                margin = best_score - second_page_score
                status = alignment_status(best_score, margin, len(normalized))
                candidates = []

                page_rankings: dict[str, tuple[float, dict[str, Any]]] = {}
                for score, page in ranked:
                    if page["page"] not in page_rankings or score > page_rankings[page["page"]][0]:
                        page_rankings[page["page"]] = (score, page)

                for page_name, (score, best_source) in sorted(
                    page_rankings.items(), key=lambda item: item[1][0], reverse=True
                )[:3]:
                    line_match = best_line_anchor(text, best_source, idf)
                    candidates.append(
                        {
                            "page": page_name,
                            "ocr_source": best_source["source"],
                            "score": round(score, 4),
                            "lines": line_match,
                        }
                    )

                # Short passages often share boilerplate; only retain a page mapping
                # when the top result is strong and clearly separates from alternatives.
                if status in {"high", "medium"}:
                    candidates = [
                        candidate
                        for candidate in candidates
                        if candidate["score"] >= max(0.68, best_score * 0.88)
                    ]
                else:
                    candidates = []

            aligned.append(
                {
                    "id": f"siman-{chapter['siman']:02d}-paragraph-{paragraph_number:03d}",
                    "ref": chapter["ref"],
                    "siman": chapter["siman"],
                    "paragraph": paragraph_number,
                    "text": text.strip(),
                    "normalized": normalize(text),
                    "alignment_status": status,
                    "mappings": candidates,
                }
            )

    return {
        "source": source["source"],
        "version_title": source["version_title"],
        "version_source": source["version_source"],
        "license": source["license"],
        "attribution": source["attribution"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mapping_method": (
            "IDF-weighted character trigram coverage against both supplied OCR layers; "
            "line ranges are approximate OCR line anchors, not printed coordinates."
        ),
        "passages": aligned,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {API_REF} from Sefaria...")
    source = fetch_source_text()
    SOURCE_TEXT_PATH.write_text(
        json.dumps(source, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Saved {sum(len(chapter['paragraphs']) for chapter in source['chapters'])} "
        f"paragraphs across {len(source['chapters'])} simanim."
    )

    print("Aligning the digital text to both supplied OCR layers...")
    pages = load_ocr_pages()
    aligned = align_source(source, pages)
    ALIGNED_INDEX_PATH.write_text(
        json.dumps(aligned, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    statuses = Counter(passage["alignment_status"] for passage in aligned["passages"])
    mapped_count = sum(bool(passage["mappings"]) for passage in aligned["passages"])
    print(f"Saved mapped index: {ALIGNED_INDEX_PATH}")
    print(
        f"Passages: {len(aligned['passages'])}; mapped: {mapped_count}; "
        f"high: {statuses['high']}; medium: {statuses['medium']}; "
        f"ambiguous/unmapped: {statuses['ambiguous'] + statuses['unmapped']}."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Index build failed: {error}", file=sys.stderr)
        raise
