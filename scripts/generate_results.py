from __future__ import annotations

import csv
from pathlib import Path
import sys

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.retrieval import RetrievalEngine

DATASET_ROOT = ROOT / "otzrot_home_assignment_candidate"
QUERIES_XLSX = DATASET_ROOT / "queries.xlsx"


def read_queries() -> list[tuple[str, str]]:
    workbook = load_workbook(QUERIES_XLSX, read_only=True, data_only=True)
    worksheet = workbook["queries"]
    rows = list(worksheet.iter_rows(values_only=True))
    header = rows[0]
    id_idx = header.index("id")
    query_idx = header.index("query")
    queries: list[tuple[str, str]] = []
    for row in rows[1:]:
        if row[id_idx] and row[query_idx]:
            queries.append((str(row[id_idx]), str(row[query_idx])))
    return queries


def main() -> None:
    engine = RetrievalEngine(DATASET_ROOT)
    out_file = ROOT / "results.csv"
    queries = read_queries()
    with out_file.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["id", "rank", "page", "snippet_or_location", "confidence"],
        )
        writer.writeheader()
        for q_id, query in queries:
            matches = engine.search(query, top_k=3)
            if not matches:
                writer.writerow(
                    {
                        "id": q_id,
                        "rank": 1,
                        "page": "",
                        "snippet_or_location": "",
                        "confidence": 0.0,
                    }
                )
                continue
            for rank, match in enumerate(matches, start=1):
                writer.writerow(
                    {
                        "id": q_id,
                        "rank": rank,
                        "page": match.page,
                        "snippet_or_location": f"{match.location_hint} :: {match.snippet_or_location}",
                        "confidence": match.confidence,
                    }
                )
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
