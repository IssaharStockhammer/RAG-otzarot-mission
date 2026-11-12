from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.deepinfra import embedding_model, embeddings


INDEX_PATH = ROOT / "data" / "external" / "aligned_passages.json"
EMBEDDINGS_PATH = ROOT / "data" / "external" / "embeddings_bge_m3.json"
BATCH_SIZE = 16


def passage_hash(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}\0{text}".encode("utf-8")).hexdigest()


def load_cache() -> dict[str, Any]:
    if not EMBEDDINGS_PATH.exists():
        return {"model": embedding_model(), "embeddings": {}}
    cache = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))
    if cache.get("model") != embedding_model():
        return {"model": embedding_model(), "embeddings": {}}
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute embeddings for high/medium aligned Sefaria passages."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all passage embeddings, including cached passages.",
    )
    args = parser.parse_args()

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INDEX_PATH}. Run scripts/build_external_index.py first."
        )

    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    passages = [
        passage
        for passage in index.get("passages", [])
        if passage.get("alignment_status") in {"high", "medium"}
        and passage.get("mappings")
        and passage.get("text", "").strip()
    ]
    if not passages:
        raise RuntimeError("The aligned index contains no mapped passages to embed.")

    model = embedding_model()
    cache = load_cache()
    vectors: dict[str, Any] = cache["embeddings"]
    pending = [
        passage
        for passage in passages
        if args.force
        or passage_hash(passage["text"], model) not in vectors
    ]
    print(
        f"Model: {model}; mapped passages: {len(passages)}; "
        f"cached: {len(passages) - len(pending)}; to embed: {len(pending)}."
    )

    for offset in range(0, len(pending), BATCH_SIZE):
        batch = pending[offset : offset + BATCH_SIZE]
        batch_vectors = embeddings([passage["text"] for passage in batch])
        for passage, vector in zip(batch, batch_vectors):
            vectors[passage_hash(passage["text"], model)] = vector
        cache["model"] = model
        cache["dimension"] = len(batch_vectors[0])
        cache["embeddings"] = vectors
        EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        EMBEDDINGS_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        print(f"Embedded {min(offset + len(batch), len(pending))}/{len(pending)} passages.")

    cache["passage_ids"] = {
        passage["id"]: passage_hash(passage["text"], model)
        for passage in passages
    }
    cache["dimension"] = len(next(iter(vectors.values()))) if vectors else 0
    EMBEDDINGS_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Saved embeddings cache: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
