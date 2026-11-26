"""opencodeGO の OpenAI 互換 API への薄いクライアント。

承認済みのクラウドモデル利用のための生成専用クライアント。
埋め込みモデルは opencodeGO カタログに存在しないため、本モジュールは
chat completions のみを提供する（埋め込みはローカル Ollama を維持）。

認証情報は環境変数名のみを参照し、値はログ・例外・成果物に一切出力しない。
既定では無効で、明示的なオプトイン（RAG_CLOUD_ENABLED=1）が必要。
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import requests

from config import mask_secrets

DEFAULT_BASE_URL = "https://opencode.ai/inference/go/openai/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"
ENV_BASE_URL = "RAG_CLOUD_BASE_URL"
ENV_MODEL = "RAG_CLOUD_MODEL"
ENV_ENABLED = "RAG_CLOUD_ENABLED"
ENV_TIMEOUT = "RAG_CLOUD_TIMEOUT"
KEY_NAMES = ("OPENCODE_API_KEY", "OPENCODE_GO_API_KEY")


class CloudUnavailable(Exception):
    """クラウド API が使えない場合の例外（理由はマスク済み）。"""


def _fail(reason: object) -> CloudUnavailable:
    return CloudUnavailable("cloud unavailable: %s" % mask_secrets(reason))


def enabled() -> bool:
    return os.environ.get(ENV_ENABLED) == "1"


def api_key_name() -> str | None:
    """設定されているキーの「環境変数名」のみを返す（値は返さない）。"""
    for name in KEY_NAMES:
        if os.environ.get(name):
            return name
    return None


def base_url() -> str:
    return os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL


def model() -> str:
    return os.environ.get(ENV_MODEL) or DEFAULT_MODEL


def timeout() -> float:
    try:
        return max(1.0, float(os.environ.get(ENV_TIMEOUT, "60")))
    except (TypeError, ValueError):
        return 60.0


def chat(messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
    """OpenAI 互換 chat completions を呼び、本文文字列を返す。

    呼び出し前に enabled() とキー環境変数の存在を確認する（値は読まない）。
    x-opencode-session ヘッダには会話単位の UUID を付与する。
    """
    if not enabled():
        raise _fail("cloud disabled (set %s=1)" % ENV_ENABLED)
    key_name = api_key_name()
    if key_name is None:
        raise _fail("no API key environment variable set")
    url = base_url().rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model(),
        "messages": messages,
        "temperature": temperature,
    }
    headers = {
        "Authorization": "Bearer %s" % os.environ.get(key_name, ""),
        "Content-Type": "application/json",
        "x-opencode-session": str(uuid.uuid4()),
    }
    try:
        response = requests.post(
            url, json=payload, headers=headers, timeout=timeout()
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])
    except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
        raise _fail(exc) from exc


# opencodeGO の Go プランで利用可能な埋め込みモデル（1M 入力トークン単価）。
# 出典: https://opencode.ai/docs/console/go/ （2026-11-26 取得）
EMBEDDING_MODELS: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "qwen3-embedding-8b": 0.01,
    "gemini-embedding-2": 0.20,
}
ENV_EMBED_MODEL = "RAG_CLOUD_EMBED_MODEL"


def embedding_models() -> list[str]:
    """Go プランで利用可能な埋め込みモデル ID を返す。"""
    return list(EMBEDDING_MODELS)


def embed_model() -> str:
    configured = os.environ.get(ENV_EMBED_MODEL)
    if configured:
        return configured
    return "text-embedding-3-small"


def _embed_batch(
    texts: list[str], *, dimensions: int | None = None
) -> list[list[float]]:
    """1 リクエスト分の埋め込み呼び出し（内部用）。"""
    key_name = api_key_name()
    if key_name is None:
        raise _fail("no API key environment variable set")
    url = base_url().rstrip("/") + "/embeddings"
    payload: dict[str, Any] = {
        "model": embed_model(),
        "input": texts,
    }
    if dimensions is not None:
        payload["dimensions"] = dimensions
    headers = {
        "Authorization": "Bearer %s" % os.environ.get(key_name, ""),
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            url, json=payload, headers=headers, timeout=timeout()
        )
        response.raise_for_status()
        data = response.json()
        vectors = [item["embedding"] for item in data["data"]]
        if len(vectors) != len(texts):
            raise ValueError("embedding count mismatch")
        return vectors
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        raise _fail(exc) from exc


# 1 リクエストあたりの入力上限（リクエストサイズとトークン制限を避ける）。
MAX_BATCH_SIZE = 64


def embed(
    texts: list[str],
    *,
    dimensions: int | None = None,
    batch_size: int | None = None,
) -> list[list[float]]:
    """OpenAI 互換 Embeddings API でベクトルを取得する。

    埋め込みリクエストには x-opencode-session が不要（公式ドキュメント）。
    大量入力は batch_size（既定 64）で分割し、入力順序どおりに結合する。
    認証情報の値はログへ出さない。
    """
    if not enabled():
        raise _fail("cloud disabled (set %s=1)" % ENV_ENABLED)
    size = batch_size or MAX_BATCH_SIZE
    if size < 1:
        raise _fail("batch_size must be positive")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), size):
        vectors.extend(
            _embed_batch(texts[start : start + size], dimensions=dimensions)
        )
    return vectors
