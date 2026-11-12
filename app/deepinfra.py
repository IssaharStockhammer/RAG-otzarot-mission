from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


class DeepInfraError(RuntimeError):
    pass


def api_key() -> str | None:
    value = os.getenv("DEEPINFRA_API_KEY", "").strip()
    return value or None


def base_url() -> str:
    return os.getenv("DEEPINFRA_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")


def embedding_model() -> str:
    return os.getenv("DEEPINFRA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()


def embeddings(texts: list[str]) -> list[list[float]]:
    key = api_key()
    if not key:
        raise DeepInfraError(
            "DEEPINFRA_API_KEY is not configured. Set it in the local .env file "
            "or as an environment variable."
        )
    if not texts:
        return []

    payload = json.dumps(
        {"model": embedding_model(), "input": texts},
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        f"{base_url()}/embeddings",
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:1000]
        raise DeepInfraError(
            f"DeepInfra embeddings request failed with HTTP {error.code}: {detail}"
        ) from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise DeepInfraError(
            f"DeepInfra embeddings request failed ({type(error).__name__}: {error})"
        ) from error

    rows = data.get("data")
    if not isinstance(rows, list) or len(rows) != len(texts):
        raise DeepInfraError("DeepInfra returned an unexpected embeddings response.")

    ordered: list[list[float] | None] = [None] * len(texts)
    for fallback_index, row in enumerate(rows):
        index = row.get("index", fallback_index)
        vector = row.get("embedding")
        if not isinstance(index, int) or not 0 <= index < len(texts):
            raise DeepInfraError("DeepInfra returned an invalid embedding index.")
        if not isinstance(vector, list) or not vector:
            raise DeepInfraError("DeepInfra returned an empty or invalid embedding.")
        ordered[index] = [float(value) for value in vector]

    if any(vector is None for vector in ordered):
        raise DeepInfraError("DeepInfra omitted one or more requested embeddings.")
    return [vector for vector in ordered if vector is not None]
