from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openpyxl import load_workbook
from pydantic import BaseModel, Field

from app.deepinfra import DeepInfraError
from app.retrieval import RetrievalEngine


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "otzrot_home_assignment_candidate"
PAGES_ROOT = DATASET_ROOT / "pages"
QUERIES_XLSX = DATASET_ROOT / "queries.xlsx"
FEEDBACK_DIR = ROOT / "data" / "feedback"
FEEDBACK_JSONL = FEEDBACK_DIR / "dev_feedback.jsonl"

app = FastAPI(title="Otzarot RAG POC", version="0.1.0")
engine = RetrievalEngine(DATASET_ROOT)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=3)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    lexical_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.55, ge=0.0, le=1.0)


class QueryItem(BaseModel):
    id: str
    query: str


class DevFeedbackRequest(BaseModel):
    query_id: str | None = None
    query_text: str = Field(min_length=1)
    retrieval_mode: str | None = None
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rank: int = Field(ge=1, le=3)
    page: str = Field(min_length=1)
    text_source: str | None = None
    ocr_source: str | None = None
    text_relevance: int | None = Field(default=None, ge=0, le=5)
    location_accuracy: int | None = Field(default=None, ge=0, le=5)
    page_correct: bool | None = None
    location_band_correct: str | None = None
    comment: str | None = Field(default=None, max_length=500)
    ranking_weights: dict[str, float] = Field(default_factory=dict)
    score_snapshot: dict[str, Any] = Field(default_factory=dict)


def _feedback_entries() -> list[dict[str, Any]]:
    if not FEEDBACK_JSONL.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in FEEDBACK_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entries.append(json.loads(line))
    return entries


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/index-status")
def index_status() -> dict[str, object]:
    return engine.external_index_status


@app.get("/api/queries", response_model=list[QueryItem])
def get_queries() -> list[QueryItem]:
    workbook = load_workbook(QUERIES_XLSX, read_only=True, data_only=True)
    worksheet = workbook["queries"]
    rows = list(worksheet.iter_rows(values_only=True))
    header = rows[0]
    id_idx = header.index("id")
    query_idx = header.index("query")
    items: list[QueryItem] = []
    for row in rows[1:]:
        q_id = row[id_idx]
        q_text = row[query_idx]
        if q_id and q_text:
            items.append(QueryItem(id=str(q_id), query=str(q_text)))
    return items


@app.post("/api/search")
def search(request: SearchRequest) -> dict[str, object]:
    requested_weight_total = request.lexical_weight + request.dense_weight
    if requested_weight_total <= 0:
        raise HTTPException(
            status_code=422,
            detail="At least one retrieval weight must be greater than zero.",
        )
    if engine.external_passages and engine.external_index_status.get(
        "deepinfra_query_embeddings_enabled"
    ):
        applied_weights = {
            "lexical": request.lexical_weight / requested_weight_total,
            "semantic": request.dense_weight / requested_weight_total,
        }
    else:
        applied_weights = {"lexical": 1.0, "semantic": 0.0}
    try:
        results = engine.search(
            request.query,
            request.top_k,
            lexical_weight=request.lexical_weight,
            dense_weight=request.dense_weight,
        )
    except DeepInfraError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    candidate_count = len(results)
    filtered_results = [
        result for result in results if result.confidence >= request.min_confidence
    ]
    if not results:
        return {
            "not_found": True,
            "not_found_reason": "no_candidates",
            "retrieval_mode": engine.retrieval_mode,
            "min_confidence": request.min_confidence,
            "ranking_weights": applied_weights,
            "candidate_count": 0,
            "results": [],
        }
    if not filtered_results:
        return {
            "not_found": True,
            "not_found_reason": "below_min_confidence",
            "retrieval_mode": engine.retrieval_mode,
            "min_confidence": request.min_confidence,
            "ranking_weights": applied_weights,
            "candidate_count": candidate_count,
            "best_confidence": max(result.confidence for result in results),
            "results": [],
        }
    return {
        "not_found": False,
        "retrieval_mode": engine.retrieval_mode,
        "min_confidence": request.min_confidence,
        "ranking_weights": applied_weights,
        "candidate_count": candidate_count,
        "results": [
            {
                "rank": i + 1,
                "page": result.page,
                "snippet_or_location": result.snippet_or_location,
                "full_text": result.full_text,
                "source": result.source,
                "line_from": result.line_from,
                "line_to": result.line_to,
                "location_hint": result.location_hint,
                "confidence": result.confidence,
                "score": round(result.score, 6),
                "text_source": result.text_source,
                "external_ref": result.external_ref,
                "alignment_status": result.alignment_status,
                "alignment_score": result.alignment_score,
                "score_details": result.score_details,
                "highlight": result.highlight,
                "image_url": f"/pages/{result.page}.png",
            }
            for i, result in enumerate(filtered_results)
        ],
    }


@app.post("/api/results-csv")
def build_results_csv() -> dict[str, str]:
    queries = get_queries()
    output_path = ROOT / "results.csv"
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["id", "rank", "page", "snippet_or_location", "confidence"],
        )
        writer.writeheader()
        for query in queries:
            matches = engine.search(query.query, top_k=3)
            if not matches:
                writer.writerow(
                    {
                        "id": query.id,
                        "rank": 1,
                        "page": "",
                        "snippet_or_location": "",
                        "confidence": 0.0,
                    }
                )
                continue
            for i, match in enumerate(matches, start=1):
                writer.writerow(
                    {
                        "id": query.id,
                        "rank": i,
                        "page": match.page,
                        "snippet_or_location": f"{match.location_hint} :: {match.snippet_or_location}",
                        "confidence": match.confidence,
                    }
                )
    return {"status": "ok", "path": str(output_path)}


@app.post("/api/dev-feedback")
def save_dev_feedback(request: DevFeedbackRequest) -> dict[str, object]:
    if request.text_relevance is None and request.location_accuracy is None:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one independent feedback rating.",
        )
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    payload = request.model_dump()
    payload["submitted_at"] = datetime.now(timezone.utc).isoformat()
    with FEEDBACK_JSONL.open("a", encoding="utf-8") as feedback_file:
        feedback_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {
        "status": "ok",
        "path": str(FEEDBACK_JSONL),
    }


@app.get("/api/dev-feedback/summary")
def dev_feedback_summary() -> dict[str, object]:
    entries = _feedback_entries()
    text_scores = [
        int(entry["text_relevance"])
        for entry in entries
        if entry.get("text_relevance") is not None
    ]
    location_scores = [
        int(entry["location_accuracy"])
        for entry in entries
        if entry.get("location_accuracy") is not None
    ]
    text_hits = sum(score >= 4 for score in text_scores)
    location_hits = sum(score >= 4 for score in location_scores)
    return {
        "entries": len(entries),
        "rating_counts": {
            "text_relevance": len(text_scores),
            "location_accuracy": len(location_scores),
        },
        "averages": {
            "text_relevance": round(sum(text_scores) / len(text_scores), 3)
            if text_scores
            else None,
            "location_accuracy": round(sum(location_scores) / len(location_scores), 3)
            if location_scores
            else None,
        },
        "kpis": {
            "text_hit_rate_ge4": round(text_hits / len(text_scores), 3)
            if text_scores
            else None,
            "location_hit_rate_ge4": round(location_hits / len(location_scores), 3)
            if location_scores
            else None,
        },
    }


@app.get("/")
def root() -> FileResponse:
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> None:
    raise HTTPException(status_code=404)


app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
app.mount("/pages", StaticFiles(directory=PAGES_ROOT), name="pages")
