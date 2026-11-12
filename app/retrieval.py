from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.deepinfra import api_key, embedding_model, embeddings


HEBREW_DIACRITICS_RE = re.compile(r"[\u0591-\u05C7]")
NON_WORD_RE = re.compile(r"[^\w\u05d0-\u05ea\s]")
MULTISPACE_RE = re.compile(r"\s+")
FINAL_LETTER_MAP = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})
MIN_DENSE_SIMILARITY = 0.35
MIN_LEXICAL_SCORE_FOR_ABSTENTION = 0.08

STOP_WORDS = {
    "איך", "האם", "מה", "מי", "למה", "מתי", "איפה", "אפשר", "מותר", "צריך",
    "של", "על", "עם", "את", "או", "גם", "אם", "בין", "הוא", "היא", "זה",
    "זו", "יש", "אין", "להיות", "יכול", "יכולה", "כאשר", "שיש", "שישנה",
}

# Small, transparent vocabulary bridges for common modern/rabbinic variants.
TERM_GROUPS = (
    {"גוי", "גוים", "נכרי", "נכרים", "עכו״ם"},
    {"תפלה", "תפילה", "להתפלל", "מתפלל", "תפלת"},
    {"מנין", "מניין", "צבור", "ציבור", "בצבור", "בציבור"},
    {"מוסף", "מוספין", "מוספים"},
    {"תמיד", "קרבן", "קרבנות", "הקרבן"},
    {"ספר", "ספרים", "ספרי", "ספר תורה", "ספרי תורה"},
    {"תורה", "תורָה"},
    {"ארון", "היכל", "ארון הקודש", "קודש"},
    {"ירושה", "יורש", "יורשים", "נחלה", "נחלת"},
    {"בת", "בתו", "בנות"},
    {"בן", "בנו", "בנים"},
    {"סכין", "סכינים", "מאכלת"},
    {"פגימה", "פגום", "פגומה", "פגימות"},
    {"שחיטה", "לשחוט", "שוחט", "שחט"},
    {"כשר", "כשרה", "כשרים", "כשרות"},
    {"תפילין", "תפלין", "בתים", "בתי"},
    {"עור", "עורות"},
    {"שחור", "שחורה", "שחורות", "השחרה", "צבע"},
    {"כהן", "כהנים", "כהנים", "דוכן", "נשיאת כפים", "ברכת כהנים"},
    {"זקן", "זקנים", "רעד", "רועדות"},
    {"טלית", "מטפחת", "מפה", "ידיים", "ידים"},
    {"שבת", "בשבת"},
    {"חול המועד", "מועד", "חול"},
    {"קבלנות", "קבלן", "קבלנים"},
    {"בניה", "בונה", "לבנות", "בנין"},
    {"כיסא", "כסא", "נישא", "נושא"},
    {"קיץ", "קיץ"},
    {"קריאת שמע", "ק״ש", "שמע"},
    {"שם ה", "השם", "הויה", "ה׳", "אדנות"},
)


def normalize_hebrew(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = HEBREW_DIACRITICS_RE.sub("", text)
    text = NON_WORD_RE.sub(" ", text)
    text = text.translate(FINAL_LETTER_MAP)
    return MULTISPACE_RE.sub(" ", text).strip()


def _query_terms(query: str) -> list[str]:
    return [token for token in normalize_hebrew(query).split() if token not in STOP_WORDS and len(token) > 1]


def _expanded_query(query: str) -> str:
    terms = _query_terms(query)
    normalized_groups = [
        {token for term in group for token in normalize_hebrew(term).split()}
        for group in TERM_GROUPS
    ]
    expanded = set(terms)
    for group in normalized_groups:
        if expanded.intersection(group):
            expanded.update(group)
    return " ".join(sorted(expanded))


def _weighted_relevance(
    lexical_relevance: np.ndarray,
    dense_relevance: np.ndarray,
    lexical_weight: float,
    dense_weight: float,
) -> np.ndarray:
    total_weight = lexical_weight + dense_weight
    if total_weight <= 0:
        raise ValueError("At least one retrieval weight must be positive.")
    return (
        lexical_weight / total_weight * lexical_relevance
        + dense_weight / total_weight * dense_relevance
    )


@dataclass
class Passage:
    page: str
    source: str
    line_from: int
    line_to: int
    total_lines: int
    text: str
    normalized_text: str


@dataclass
class SearchResult:
    page: str
    snippet_or_location: str
    full_text: str
    source: str
    line_from: int | None
    line_to: int | None
    location_hint: str
    confidence: float
    score: float
    text_source: str = "OCR"
    external_ref: str | None = None
    alignment_status: str | None = None
    alignment_score: float | None = None
    highlight: dict[str, float | str] | None = None
    score_details: dict[str, float] | None = None


class RetrievalEngine:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root
        self.passages = self._load_passages(dataset_root)
        if not self.passages:
            raise ValueError(f"No OCR passages found under {dataset_root}")

        corpus = [passage.normalized_text for passage in self.passages]
        self._char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        self._word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)
        self._char_matrix = self._char_vectorizer.fit_transform(corpus)
        self._word_matrix = self._word_vectorizer.fit_transform(corpus)
        self.external_index_path = (
            dataset_root.parent / "data" / "external" / "aligned_passages.json"
        )
        self.embeddings_path = (
            dataset_root.parent / "data" / "external" / "embeddings_bge_m3.json"
        )
        self.external_passages: list[dict[str, object]] = []
        self._external_char_vectorizer: TfidfVectorizer | None = None
        self._external_word_vectorizer: TfidfVectorizer | None = None
        self._external_char_matrix = None
        self._external_word_matrix = None
        self._external_embeddings: np.ndarray | None = None
        if self.external_index_path.exists():
            self._load_external_index()

    def _load_external_index(self) -> None:
        with self.external_index_path.open(encoding="utf-8") as index_file:
            data = json.load(index_file)
        self.external_passages = [
            passage
            for passage in data.get("passages", [])
            if passage.get("alignment_status") in {"high", "medium"}
            and passage.get("mappings")
        ]
        if not self.external_passages:
            return

        corpus = [normalize_hebrew(str(passage["text"])) for passage in self.external_passages]
        self._external_char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True
        )
        self._external_word_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True
        )
        self._external_char_matrix = self._external_char_vectorizer.fit_transform(corpus)
        self._external_word_matrix = self._external_word_vectorizer.fit_transform(corpus)
        if self.embeddings_path.exists():
            cache = json.loads(self.embeddings_path.read_text(encoding="utf-8"))
            if cache.get("model") == embedding_model():
                passage_ids = cache.get("passage_ids", {})
                vectors = cache.get("embeddings", {})
                keyed_vectors = [
                    vectors[passage_ids[str(passage["id"])]]
                    for passage in self.external_passages
                    if str(passage["id"]) in passage_ids
                    and passage_ids[str(passage["id"])] in vectors
                ]
                if len(keyed_vectors) == len(self.external_passages) and keyed_vectors:
                    matrix = np.asarray(keyed_vectors, dtype=np.float32)
                    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    self._external_embeddings = matrix / norms

    @property
    def external_index_status(self) -> dict[str, object]:
        if not self.external_passages:
            return {
                "enabled": False,
                "source": None,
                "mapped_passages": 0,
                "path": str(self.external_index_path),
            }
        with self.external_index_path.open(encoding="utf-8") as index_file:
            data = json.load(index_file)
        return {
            "enabled": True,
            "source": data.get("source"),
            "version_title": data.get("version_title"),
            "version_source": data.get("version_source"),
            "license": data.get("license"),
            "mapped_passages": len(self.external_passages),
            "embedding_model": embedding_model(),
            "embedding_cache_ready": self._external_embeddings is not None,
            "deepinfra_query_embeddings_enabled": (
                self._external_embeddings is not None and api_key() is not None
            ),
            "path": str(self.external_index_path),
        }

    @property
    def retrieval_mode(self) -> str:
        if self._external_embeddings is not None and api_key() is not None:
            return "hybrid (DeepInfra embeddings + lexical)"
        if self.external_passages:
            return "lexical (external text)"
        return "lexical (supplied OCR)"

    @staticmethod
    def _load_passages(dataset_root: Path) -> list[Passage]:
        passages: list[Passage] = []
        for source, folder in (
            ("hebrewbooks", dataset_root / "ocr_hebrewbooks"),
            ("tesseract", dataset_root / "ocr_tesseract"),
        ):
            for text_file in sorted(folder.glob("page-*.txt")):
                file_text = text_file.read_text(encoding="utf-8", errors="replace")
                total_lines = len(file_text.splitlines())
                nonempty_lines = [
                    (line_number, line.strip())
                    for line_number, line in enumerate(
                        file_text.splitlines(),
                        start=1,
                    )
                    if line.strip()
                ]
                window_size = 3
                stride = 2
                for start in range(0, len(nonempty_lines), stride):
                    window = nonempty_lines[start : start + window_size]
                    if not window:
                        continue
                    text = " ".join(line for _, line in window)
                    passages.append(
                        Passage(
                            page=text_file.stem,
                            source=source,
                            line_from=window[0][0],
                            line_to=window[-1][0],
                            total_lines=total_lines,
                            text=text,
                            normalized_text=normalize_hebrew(text),
                        )
                    )
        return passages

    def _score_query(self, query: str) -> np.ndarray:
        normalized_query = " ".join(_query_terms(query))
        expanded_query = _expanded_query(query)
        if not normalized_query:
            return np.zeros(len(self.passages))

        char_query = self._char_vectorizer.transform([normalized_query])
        word_query = self._word_vectorizer.transform([normalized_query])
        direct_scores = (
            0.62 * (self._char_matrix @ char_query.T).toarray().ravel()
            + 0.38 * (self._word_matrix @ word_query.T).toarray().ravel()
        )

        if expanded_query == normalized_query:
            return direct_scores

        expanded_char_query = self._char_vectorizer.transform([expanded_query])
        expanded_word_query = self._word_vectorizer.transform([expanded_query])
        expanded_scores = (
            0.62 * (self._char_matrix @ expanded_char_query.T).toarray().ravel()
            + 0.38 * (self._word_matrix @ expanded_word_query.T).toarray().ravel()
        )
        return 0.82 * direct_scores + 0.18 * expanded_scores

    def _score_external_query(self, query: str) -> np.ndarray:
        if (
            not self.external_passages
            or self._external_char_vectorizer is None
            or self._external_word_vectorizer is None
            or self._external_char_matrix is None
            or self._external_word_matrix is None
        ):
            return np.zeros(0)

        normalized_query = " ".join(_query_terms(query))
        expanded_query = _expanded_query(query)
        if not normalized_query:
            return np.zeros(len(self.external_passages))

        def score(text: str) -> np.ndarray:
            char_query = self._external_char_vectorizer.transform([text])
            word_query = self._external_word_vectorizer.transform([text])
            return (
                0.62 * (self._external_char_matrix @ char_query.T).toarray().ravel()
                + 0.38 * (self._external_word_matrix @ word_query.T).toarray().ravel()
            )

        direct_scores = score(normalized_query)
        if expanded_query == normalized_query:
            return direct_scores
        expanded_scores = score(expanded_query)
        return 0.82 * direct_scores + 0.18 * expanded_scores

    def _search_external(
        self,
        query: str,
        top_k: int,
        lexical_weight: float,
        dense_weight: float,
    ) -> list[SearchResult]:
        lexical_scores = self._score_external_query(query)
        if not len(lexical_scores):
            return []

        dense_scores: np.ndarray | None = None
        if self._external_embeddings is not None and api_key() is not None:
            query_vector = np.asarray(embeddings([query])[0], dtype=np.float32)
            query_norm = np.linalg.norm(query_vector)
            if query_norm:
                query_vector /= query_norm
            dense_scores = self._external_embeddings @ query_vector

        if dense_scores is not None:
            if (
                float(np.max(dense_scores)) < MIN_DENSE_SIMILARITY
                and float(np.max(lexical_scores)) < MIN_LEXICAL_SCORE_FOR_ABSTENTION
            ):
                return []
        elif float(np.max(lexical_scores)) < 0.035:
            return []

        lexical_relevance = np.clip(lexical_scores / 0.2, 0.0, 1.0)
        if dense_scores is not None:
            dense_relevance = np.clip(
                (dense_scores - MIN_DENSE_SIMILARITY)
                / (1.0 - MIN_DENSE_SIMILARITY),
                0.0,
                1.0,
            )
            retrieval_scores = _weighted_relevance(
                lexical_relevance,
                dense_relevance,
                lexical_weight,
                dense_weight,
            )
            total_weight = lexical_weight + dense_weight
            effective_lexical_weight = lexical_weight / total_weight
            effective_dense_weight = dense_weight / total_weight
        else:
            dense_relevance = np.zeros(len(lexical_scores), dtype=np.float32)
            effective_lexical_weight = 1.0
            effective_dense_weight = 0.0
            retrieval_scores = lexical_relevance
        best_by_page: dict[str, tuple[float, dict[str, object], dict[str, object]]] = {}
        for index, passage in enumerate(self.external_passages):
            lexical_score = float(lexical_scores[index])
            dense_score = float(dense_scores[index]) if dense_scores is not None else None
            retrieval_score = float(retrieval_scores[index])
            for mapping in passage["mappings"]:
                page = str(mapping["page"])
                current = best_by_page.get(page)
                if current is None or retrieval_score > current[0]:
                    best_by_page[page] = (
                        retrieval_score,
                        passage,
                        {
                            **mapping,
                            "lexical_score": lexical_score,
                            "dense_score": dense_score,
                            "lexical_relevance": float(lexical_relevance[index]),
                            "dense_relevance": float(dense_relevance[index]),
                            "effective_lexical_weight": effective_lexical_weight,
                            "effective_dense_weight": effective_dense_weight,
                        },
                    )

        ranked = sorted(best_by_page.values(), key=lambda item: item[0], reverse=True)
        if not ranked:
            return []

        best_retrieval_score = float(ranked[0][0])
        second_retrieval_score = (
            float(ranked[1][0]) if len(ranked) > 1 else best_retrieval_score * 0.92
        )
        top_margin = max(
            0.0,
            min(
                1.0,
                (best_retrieval_score - second_retrieval_score)
                / max(best_retrieval_score, 1e-9),
            ),
        )
        results: list[SearchResult] = []
        for score, passage, mapping in ranked[:top_k]:
            siman = int(passage["siman"])
            paragraph_number = int(passage["paragraph"])
            lines = mapping.get("lines")
            line_from = int(lines["line_from"]) if lines else None
            line_to = int(lines["line_to"]) if lines else None
            ocr_source = str(mapping["ocr_source"])
            location_parts = [
                f"נודע ביהודה, קמא, אורח חיים, סימן {siman}, פסקה {paragraph_number}",
                f"{mapping['page']} לפי {ocr_source} (יישור {passage['alignment_status']}, {mapping['score']:.2f})",
            ]
            if lines:
                location_parts.append(
                    f"שורות OCR משוערות {line_from}-{line_to}"
                )
            snippet = re.sub(r"\s+", " ", str(passage["text"])).strip()
            if len(snippet) > 280:
                snippet = snippet[:277].rstrip() + "..."
            confidence = max(0.0, min(1.0, float(score)))
            alignment_relevance = float(mapping["score"])
            results.append(
                SearchResult(
                    page=str(mapping["page"]),
                    snippet_or_location=snippet,
                    full_text=str(passage["text"]),
                    source=ocr_source,
                    line_from=line_from,
                    line_to=line_to,
                    location_hint=" · ".join(location_parts),
                    confidence=round(max(0.0, min(1.0, confidence)), 3),
                    score=score,
                    text_source="Sefaria",
                    external_ref=str(passage["ref"]),
                    alignment_status=str(passage["alignment_status"]),
                    alignment_score=round(float(mapping["score"]), 4),
                    highlight=(
                        self._box_highlight(
                            top_pct=float(lines["top_pct"]),
                            height_pct=float(lines["height_pct"]),
                            column_hint=str(lines.get("column_hint", "unknown")),
                        )
                        if lines and "top_pct" in lines and "height_pct" in lines
                        else None
                    ),
                    score_details={
                        "alignment": round(alignment_relevance, 4),
                        "lexical": round(float(mapping["lexical_relevance"]), 4),
                        "dense": round(float(mapping["dense_relevance"]), 4),
                        "lexical_weight": round(
                            float(mapping["effective_lexical_weight"]), 4
                        ),
                        "dense_weight": round(
                            float(mapping["effective_dense_weight"]), 4
                        ),
                        "weighted_relevance": round(confidence, 4),
                        "top_margin": round(top_margin, 4),
                    },
                )
            )
        return results

    @staticmethod
    def _confidence(score: float, best_score: float, margin: float) -> float:
        # A ranking score, not a statistically calibrated probability.
        relative = max(0.0, min(1.0, score / max(best_score, 1e-9)))
        return float(max(0.0, min(1.0, 0.84 * relative + 0.16 * margin)))

    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        lexical_weight: float = 0.45,
        dense_weight: float = 0.55,
    ) -> list[SearchResult]:
        if not _query_terms(query):
            return []
        if lexical_weight < 0 or dense_weight < 0 or lexical_weight + dense_weight <= 0:
            raise ValueError("At least one non-negative retrieval weight must be positive.")

        external_results = self._search_external(
            query,
            top_k,
            lexical_weight,
            dense_weight,
        )
        if self.external_passages:
            return external_results

        passage_scores = self._score_query(query)
        best_by_page: dict[str, tuple[float, Passage]] = {}
        source_reliability = {"hebrewbooks": 1.0, "tesseract": 0.88}

        for passage, score in zip(self.passages, passage_scores):
            adjusted_score = float(score) * source_reliability[passage.source]
            current = best_by_page.get(passage.page)
            if current is None or adjusted_score > current[0]:
                best_by_page[passage.page] = (adjusted_score, passage)

        ranked = sorted(best_by_page.values(), key=lambda item: item[0], reverse=True)
        if not ranked or ranked[0][0] < 0.035:
            return []

        best_score = float(ranked[0][0])
        second_score = float(ranked[1][0]) if len(ranked) > 1 else best_score * 0.92
        top_margin = max(0.0, min(1.0, (best_score - second_score) / max(best_score, 1e-9)))
        results: list[SearchResult] = []
        for rank_index, (score, passage) in enumerate(ranked[:top_k], start=1):
            snippet = re.sub(r"\s+", " ", passage.text).strip()
            if len(snippet) > 280:
                snippet = snippet[:277].rstrip() + "..."
            relative = max(0.0, min(1.0, float(score) / max(best_score, 1e-9)))
            confidence = self._confidence(float(score), best_score, top_margin if rank_index == 1 else 0.0)
            results.append(
                SearchResult(
                    page=passage.page,
                    snippet_or_location=snippet,
                    full_text=passage.text,
                    source=passage.source,
                    line_from=passage.line_from,
                    line_to=passage.line_to,
                    location_hint=f"{passage.source} lines {passage.line_from}-{passage.line_to}",
                    confidence=round(confidence, 3),
                    score=score,
                    highlight=self._line_highlight(
                        passage.line_from, passage.line_to, passage.total_lines
                    ),
                    score_details={
                        "alignment": 0.0,
                        "lexical": round(
                            max(0.0, min(1.0, float(score) / 0.22)),
                            4,
                        ),
                        "dense": 0.0,
                        "rrf_relative": round(relative, 4),
                        "top_margin": round(top_margin, 4),
                    },
                )
            )
        return results

    @staticmethod
    def _line_highlight(
        line_from: int, line_to: int, total_lines: int
    ) -> dict[str, float | str] | None:
        if total_lines <= 0:
            return None
        top_pct = max(0.0, min(100.0, (line_from - 1) / total_lines * 100))
        bottom_pct = max(
            top_pct,
            min(100.0, line_to / total_lines * 100),
        )
        center = (line_from + line_to) / (2 * total_lines)
        column_hint = "right" if center <= 0.5 else "left"
        return RetrievalEngine._box_highlight(
            top_pct=top_pct,
            height_pct=max(2.5, bottom_pct - top_pct),
            column_hint=column_hint,
        )

    @staticmethod
    def _box_highlight(
        *, top_pct: float, height_pct: float, column_hint: str
    ) -> dict[str, float | str]:
        normalized_hint = column_hint if column_hint in {"left", "right"} else "full"
        if normalized_hint == "right":
            left_pct = 52.0
            width_pct = 44.0
        elif normalized_hint == "left":
            left_pct = 4.0
            width_pct = 44.0
        else:
            left_pct = 4.0
            width_pct = 92.0
        return {
            "top_pct": round(max(0.0, min(100.0, top_pct)), 2),
            "height_pct": round(max(2.5, min(100.0, height_pct)), 2),
            "left_pct": left_pct,
            "width_pct": width_pct,
            "label": "טווח משוער",
            "column_hint": normalized_hint,
        }
