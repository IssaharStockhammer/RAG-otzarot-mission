# Repository context

## Shared agent journal

Before starting work, read `AGENTS.md`, `PROJECT_STATUS_he.md`, and
`AGENT_CHAT_he.md`. Use the journal for attributed findings, proposals, replies,
and handoffs between Copilot, Codex, and other agents. Append new entries with
unique IDs and references to earlier entries; preserve other agents' messages.
Distinguish verified findings and user decisions from unimplemented proposals.
At handoff, record actual changes and validation, and update the concise status
summary when relevant. Git commits/pushes/pulls synchronize this journal; it is
not a live chat. The starter descriptions below predate the implementation;
use the current code and status file to establish what exists now.

This repository is the starter dataset for a Hebrew RAG retrieval assignment. There is no application source, dependency manifest, build script, test suite, or linter yet. The assignment specification is `otzrot_home_assignment_candidate/תרגיל בית – מפתח_ת אוצרות התורה (v3).pdf`; treat it as the authoritative delivery and evaluation contract.

## Dataset architecture

- `otzrot_home_assignment_candidate/pages/page-NN.png` contains the 30 scanned source pages. Use these images as the ground truth for page references and source snippets.
- `ocr_hebrewbooks/page-NN.txt` and `ocr_tesseract/page-NN.txt` are two OCR transcriptions of the same corresponding page image. Preserve the `page-NN` mapping when chunking, indexing, evaluating, or returning results.
- `queries.xlsx` contains the `queries` worksheet with query IDs (`Q01`–`Q12`) and Hebrew queries. Its additional `page`, `siman_approx`, and `answerable ?` columns are reserved for evaluation data.

## Retrieval and output conventions

- The corpus and user queries are Hebrew, including rabbinic/Halachic terminology. Keep Hebrew text intact through ingestion, normalization, embeddings, retrieval, and result rendering.
- Compare or combine the two supplied OCR sources deliberately; do not assume either transcription is fully accurate. When a returned snippet is unclear, validate it against the matching image in `pages/`.
- Retrieval output is expected as `results.csv` with `id`, `rank`, `page`, `snippet_or_location`, and `confidence` fields. `page` must use the source-page identity, and `snippet_or_location` must be traceable to that page.
- If adding the requested service, document its local startup command in `README.md`; the assignment calls for a Docker Compose startup flow (`docker compose up`) and configurable provider credentials through an `.env` file rather than committed secrets.

## Validation

No build, lint, or automated test commands exist in the starter repository. When implementation is introduced, add and document the project-specific commands, including a focused command for one test.
