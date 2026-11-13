# RAG-otzarot-mission

POC for Hebrew retrieval over the Otzrot dataset.

> Resuming work? Read [`PROJECT_STATUS_he.md`](PROJECT_STATUS_he.md) first: current state, findings from the feedback, and open tasks.

> Working with multiple AI agents? Read [`AGENTS.md`](AGENTS.md) and
> [`AGENT_CHAT_he.md`](AGENT_CHAT_he.md), the shared discussion and handoff journal.
> Append findings and replies there; keep the status file as the current-state summary.

## Quick start

### Local (Python)

```powershell
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open: `http://localhost:8000`

### Docker Compose

```powershell
docker compose up --build
```

Open: `http://localhost:8000`

## Generate required CSV

```powershell
python scripts/generate_results.py
```

This writes `results.csv` in the repository root with:
`id, rank, page, snippet_or_location, confidence`.

## Build the external-text index

The POC can use Sefaria's Hebrew transcription as its searchable text and align
each paragraph back to the supplied page scans using both provided OCR layers.
No API key is required:

```powershell
python scripts/build_external_index.py
```

The command downloads the Hebrew Noda BiYehuda I, Orach Chaim text for
simanim 1-41 from the Sefaria API, then creates:

- `data/external/sefaria_noda_biyahudah_i_orach_chaim.json` — source text and
  provenance metadata.
- `data/external/aligned_passages.json` — searchable paragraphs, mapped source
  pages, OCR layer, approximate OCR line anchors, and alignment confidence.

The downloaded version identifies itself as **Noda BeYehuda Warsaw 1880** and
the Sefaria API reports its Hebrew text license as **Public Domain**. The
printed-page mapping is inferred from OCR similarity; OCR line numbers are
approximate anchors, not page-image coordinates. Always inspect the linked
scan when validating a result. The app falls back to OCR-only retrieval if the
external index has not been built or has no mapped candidate.

## DeepInfra hybrid retrieval

The app combines lexical TF-IDF retrieval with dense semantic retrieval using
DeepInfra's OpenAI-compatible embeddings API and `BAAI/bge-m3` (multilingual,
including Hebrew). Set `DEEPINFRA_API_KEY` in `.env`; do not commit `.env` or
paste the key into source files. `.env.example` lists the supported settings.
If the key is absent, search remains available in lexical-only mode.

Precompute and cache embeddings for mapped passages once:

```powershell
python scripts/build_embeddings.py
```

The script skips vectors already cached for the exact text/model pair. Rebuild
only after changing the source index or model; use `--force` to recompute all
vectors. The cache is local and ignored by git. At query time the app sends
only the query text to DeepInfra and fuses dense and lexical rankings with
normalized, configurable lexical and semantic relevance weights. In developer
mode (`?dev=1` or the UI toggle), adjust these weights and rerun the same query
to compare candidate ordering and scores. Alignment/location quality is shown
separately and does not participate in text ranking. If DeepInfra is configured but a query request
fails, the UI surfaces the error rather than silently pretending to use
semantic search.

The page marker is an estimated OCR band with a **column hint** ("טווח משוער")
so you can focus on the likely column first. It is still approximate (not
pixel-accurate bounding boxes), so verify against the scan.

## Tests

```powershell
pytest -q
pytest tests/test_health.py -q
```

## Evaluation modes

The UI has three modes: regular user mode, text-relevance evaluation, and
page-location evaluation. Text evaluation displays the complete retrieved
passage and lets you tune lexical/semantic weights. Location evaluation shows
the estimated source column and accepts location-only feedback. Both evaluation
metrics are recorded independently.

Each saved feedback record includes the ranking weights actually applied to
that search in its top-level `ranking_weights` field, as well as a score
snapshot. Feedback is appended to `data/feedback/dev_feedback.jsonl`; separate
averages and hit rates are available via `/api/dev-feedback/summary`.

## Current POC scope

- Uses Sefaria's clean text for retrieval when the generated index is available.
- Aligns external paragraphs to source pages through IDF-weighted character
  trigram matching against both OCR layers.
- Falls back to hybrid lexical TF-IDF over the supplied OCR when needed.
- Returns top 3 page candidates or "not found".
- Minimal Hebrew RTL UI with source page preview image, adjustable minimum
  confidence threshold, and score-breakdown metadata for debugging.
