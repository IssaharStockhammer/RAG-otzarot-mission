import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app import main as app_main
from app.main import app, engine
from app.retrieval import SearchResult, _weighted_relevance


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_bank_is_available() -> None:
    client = TestClient(app)
    response = client.get("/api/queries")

    assert response.status_code == 200
    queries = response.json()
    assert len(queries) == 12
    assert queries[0]["id"] == "Q01"
    assert queries[0]["query"]


def test_ui_includes_saved_query_selector() -> None:
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert 'id="saved-query"' in response.text
    assert 'id="previous-query"' in response.text
    assert 'id="next-query"' in response.text
    assert 'id="min-confidence"' in response.text
    assert 'id="sticky-query-text"' in response.text
    assert 'id="lexical-weight"' in response.text
    assert 'id="dense-weight"' in response.text
    assert 'id="apply-weights"' in response.text
    assert 'data-mode="user"' in response.text
    assert 'data-mode="text"' in response.text
    assert 'data-mode="location"' in response.text


def test_external_index_status_reports_provenance() -> None:
    client = TestClient(app)
    response = client.get("/api/index-status")

    assert response.status_code == 200
    status = response.json()
    assert status["enabled"] is True
    assert status["source"] == "Sefaria"
    assert status["license"] == "Public Domain"
    assert status["mapped_passages"] > 0


def test_search_returns_traceable_external_record(monkeypatch) -> None:
    client = TestClient(app)
    index_path = Path(__file__).resolve().parents[1] / "data" / "external" / "aligned_passages.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    sample_text = next(
        passage["text"][:100]
        for passage in index["passages"]
        if passage["alignment_status"] in {"high", "medium"} and passage["mappings"]
    )
    if engine._external_embeddings is not None:
        monkeypatch.setattr(
            "app.retrieval.api_key",
            lambda: "test-key",
        )
        monkeypatch.setattr(
            "app.retrieval.embeddings",
            lambda texts: [engine._external_embeddings[0].tolist()],
        )
    response = client.post(
        "/api/search",
        json={"query": sample_text, "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"]
    result = payload["results"][0]
    assert result["text_source"] == "Sefaria"
    assert result["external_ref"]
    assert result["alignment_status"] in {"high", "medium"}
    assert result["page"].startswith("page-")
    assert result["image_url"].endswith(".png")
    assert result["full_text"]
    assert result["highlight"] is not None
    assert (
        0 <= result["highlight"]["top_pct"] <= 100
        and 0 < result["highlight"]["height_pct"] <= 100
    )


def test_search_abstains_when_both_retrieval_signals_are_empty(monkeypatch) -> None:
    if engine._external_embeddings is not None:
        monkeypatch.setattr("app.retrieval.api_key", lambda: "test-key")
        monkeypatch.setattr(
            "app.retrieval.embeddings",
            lambda texts: [[0.0] * engine._external_embeddings.shape[1]],
        )

    client = TestClient(app)
    response = client.post(
        "/api/search",
        json={"query": "zxqvplmbrt", "top_k": 3},
    )

    assert response.status_code == 200
    assert response.json()["not_found"] is True
    assert response.json()["results"] == []


def test_search_respects_min_confidence_threshold(monkeypatch) -> None:
    result = SearchResult(
        page="page-01",
        snippet_or_location="sample",
        full_text="sample",
        source="hebrewbooks",
        line_from=1,
        line_to=3,
        location_hint="sample location",
        confidence=0.8,
        score=0.8,
    )
    monkeypatch.setattr(
        app_main.engine,
        "search",
        lambda query, top_k, **kwargs: [result],
    )
    client = TestClient(app)
    baseline = client.post(
        "/api/search",
        json={"query": "sample query", "top_k": 3, "min_confidence": 0.0},
    )
    assert baseline.status_code == 200
    baseline_payload = baseline.json()
    assert baseline_payload["not_found"] is False
    best_confidence = baseline_payload["results"][0]["confidence"]
    threshold = float(best_confidence) + 0.01

    response = client.post(
        "/api/search",
        json={"query": "sample query", "top_k": 3, "min_confidence": threshold},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["not_found"] is True
    assert payload["not_found_reason"] == "below_min_confidence"
    assert payload["results"] == []


def test_manual_weights_change_retrieval_order() -> None:
    lexical = np.array([0.9, 0.1])
    semantic = np.array([0.1, 0.9])
    lexical_first = _weighted_relevance(lexical, semantic, 0.9, 0.1)
    semantic_first = _weighted_relevance(lexical, semantic, 0.1, 0.9)

    assert int(np.argmax(lexical_first)) == 0
    assert int(np.argmax(semantic_first)) == 1


def test_search_accepts_and_reports_manual_weights(monkeypatch) -> None:
    monkeypatch.setattr("app.retrieval.api_key", lambda: None)
    client = TestClient(app)
    response = client.post(
        "/api/search",
        json={
            "query": "נודע ביהודה תפלה",
            "top_k": 3,
            "min_confidence": 0.0,
            "lexical_weight": 0.8,
            "dense_weight": 0.2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ranking_weights"] == {"lexical": 1.0, "semantic": 0.0}
    if payload["results"]:
        details = payload["results"][0]["score_details"]
        assert details["lexical_weight"] == 1.0
        assert details["dense_weight"] == 0.0


def test_search_rejects_all_zero_weights() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/search",
        json={
            "query": "בדיקה",
            "lexical_weight": 0,
            "dense_weight": 0,
        },
    )

    assert response.status_code == 422


def test_dev_feedback_endpoints(tmp_path, monkeypatch) -> None:
    feedback_dir = tmp_path / "feedback"
    feedback_jsonl = feedback_dir / "dev_feedback.jsonl"
    monkeypatch.setattr(app_main, "FEEDBACK_DIR", feedback_dir)
    monkeypatch.setattr(app_main, "FEEDBACK_JSONL", feedback_jsonl)
    client = TestClient(app)

    payload = {
        "query_id": "Q01",
        "query_text": "בדיקת משוב",
        "retrieval_mode": "חיפוש מילולי",
        "min_confidence": 0.35,
        "rank": 1,
        "page": "page-01",
        "text_source": "Sefaria",
        "ocr_source": "hebrewbooks",
        "text_relevance": 4,
        "location_accuracy": 2,
        "page_correct": True,
        "location_band_correct": "partial",
        "comment": "בדיקה",
        "ranking_weights": {"lexical": 0.8, "semantic": 0.2},
        "score_snapshot": {"confidence": 0.77, "lexical": 0.65},
    }
    save_response = client.post("/api/dev-feedback", json=payload)
    assert save_response.status_code == 200
    assert feedback_jsonl.exists()

    summary_response = client.get("/api/dev-feedback/summary")
    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert summary["entries"] == 1
    assert summary["averages"]["text_relevance"] == 4.0
    assert summary["averages"]["location_accuracy"] == 2.0


def test_dev_feedback_can_rate_one_independent_axis(tmp_path, monkeypatch) -> None:
    feedback_dir = tmp_path / "feedback"
    feedback_jsonl = feedback_dir / "dev_feedback.jsonl"
    monkeypatch.setattr(app_main, "FEEDBACK_DIR", feedback_dir)
    monkeypatch.setattr(app_main, "FEEDBACK_JSONL", feedback_jsonl)
    client = TestClient(app)
    payload = {
        "query_id": "Q02",
        "query_text": "בדיקת טקסט בלבד",
        "rank": 2,
        "page": "page-02",
        "text_relevance": 5,
        "ranking_weights": {"lexical": 0.3, "semantic": 0.7},
    }

    response = client.post("/api/dev-feedback", json=payload)

    assert response.status_code == 200
    saved = json.loads(feedback_jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert saved["text_relevance"] == 5
    assert saved["location_accuracy"] is None
    assert saved["ranking_weights"] == {"lexical": 0.3, "semantic": 0.7}
    summary = client.get("/api/dev-feedback/summary").json()
    assert summary["rating_counts"] == {"text_relevance": 1, "location_accuracy": 0}
    assert summary["averages"]["location_accuracy"] is None
