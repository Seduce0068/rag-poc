"""opencodeGO の OpenAI 互換 API への薄いクライアント。

chat completions と OpenAI 互換 /embeddings の両方を提供する。埋め込みは
ローカル Ollama の代替として使えるが、既定では無効で明示的なオプトイン
（RAG_CLOUD_ENABLED=1）が必要。オプトインしない限り実通信は行わない。

認証情報は環境変数名のみを参照し、値はログ・例外・成果物に一切出力しない。
キー名の優先順は OPENCODE_GO_API_KEY（正規）→ OPENCODE_API_KEY（互換）。
利用可能な埋め込みモデルは EMBEDDING_MODELS 定数を参照する（任意で
fetch_models() が /v1/models を取得する）。API レスポンスの usage
（トークン数、あれば cost）は last_usage() から参照できる。
"""

from __future__ import annotations

import os
import uuid
import warnings
from typing import Any

import requests

import config

DEFAULT_BASE_URL = "https://opencode.ai/inference/go/openai/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"
ENV_BASE_URL = "RAG_CLOUD_BASE_URL"
ENV_MODEL = "RAG_CLOUD_MODEL"
ENV_ENABLED = "RAG_CLOUD_ENABLED"
ENV_TIMEOUT = "RAG_CLOUD_TIMEOUT"

# 正規（OPENCODE_GO_API_KEY）を優先し、OPENCODE_API_KEY は後方互換として残す。
KEY_NAMES = ("OPENCODE_GO_API_KEY", "OPENCODE_API_KEY")

# base_url() は既定で .../v1 を含むため、"/models" で実質 /v1/models を呼ぶ。
MODELS_PATH = "/models"


class CloudUnavailable(Exception):
    """クラウド API が使えない場合の例外（理由はマスク済み）。"""


def _fail(reason: object) -> CloudUnavailable:
    return CloudUnavailable("cloud unavailable: %s" % config.mask_secrets(reason))


def enabled() -> bool:
    return os.environ.get(ENV_ENABLED) == "1"


def api_key_name() -> str | None:
    """設定されているキーの「環境変数名」のみを返す（値は返さない）。

    優先順は OPENCODE_GO_API_KEY（正規）→ OPENCODE_API_KEY（後方互換）。
    両方が設定されている場合は警告を出すが、警告文には変数名だけを含め、
    キーの値は一切出力しない。
    """
    primary, compat = KEY_NAMES
    if os.environ.get(primary) and os.environ.get(compat):
        warnings.warn(
            "両方の API キー環境変数が設定されています: %s を優先し、%s は無視します"
            % (primary, compat),
            RuntimeWarning,
            stacklevel=2,
        )
    if os.environ.get(primary):
        return primary
    if os.environ.get(compat):
        return compat
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


# ---------------------------------------------------------------------------
# 共通リクエスト（ヘッダ方針・リトライ方針・usage 保持を 1 箇所へ集約）
# ---------------------------------------------------------------------------

_SESSION: Any | None = None


def _shared_session() -> Any:
    """config.RETRY_* 準拠の Retry をマウントした Session を遅延生成する。

    リトライ対象には config.RETRY_STATUS_FORCELIST の 429 を含む。
    import 時には生成せず、最初の実通信時だけ作る（テストの stub にも耐える）。
    """
    global _SESSION
    if _SESSION is None:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retry = Retry(
            total=config.RETRY_TOTAL,
            backoff_factor=config.RETRY_BACKOFF,
            status_forcelist=list(config.RETRY_STATUS_FORCELIST),
            allowed_methods=frozenset(config.RETRY_ALLOWED_METHODS),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _SESSION = session
    return _SESSION


# 直近の chat() / embed() が受け取った usage の数値項目（呼び出し単位で加算）。
_LAST_USAGE: dict[str, Any] | None = None


def _reset_usage() -> None:
    global _LAST_USAGE
    _LAST_USAGE = None


def _accumulate_usage(usage: object) -> None:
    """API の usage から数値項目だけを取り出し、呼び出し単位で加算保持する。

    文字列値や長いキー名（認証情報になり得る値）は保持しない。cost が返る
    実装では cost / cost_usd 等の数値項目もそのまま加算される。
    nested な辞書（例: cost: {"usd": 0.01}）は "cost.usd" として平坦化する。
    """
    if not isinstance(usage, dict):
        return
    additions: dict[str, Any] = {}
    for key, value in usage.items():
        if not isinstance(key, str) or len(key) > 64:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat = "%s.%s" % (key, sub_key)
                if (isinstance(sub_key, str) and len(flat) <= 128
                        and isinstance(sub_value, (int, float))
                        and not isinstance(sub_value, bool)):
                    additions[flat] = sub_value
        elif isinstance(value, (int, float)):
            additions[key] = value
    if not additions:
        return
    global _LAST_USAGE
    if _LAST_USAGE is None:
        _LAST_USAGE = {}
    for key, value in additions.items():
        _LAST_USAGE[key] = _LAST_USAGE.get(key, 0) + value


def last_usage() -> dict[str, Any] | None:
    """直近の chat() / embed() が保持した usage のコピーを返す（通信なし）。

    戻り値は API の usage に含まれていた数値項目だけ（例: prompt_tokens,
    completion_tokens, total_tokens, cost / cost_usd）。embed() は複数バッチの
    値を合計する。usage が無いレスポンスなら None。認証情報の値は含まない。

    cloud_bench 等の呼び出し側は chat() / embed() の直後に本関数を呼び、
    record / summary へ格納すれば良い（戻り値の str / list は従来どおり）。
    """
    return dict(_LAST_USAGE) if _LAST_USAGE is not None else None


def _request(
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    session_id: str | None = None,
    method: str = "POST",
) -> dict[str, Any]:
    """chat / embed 共通の HTTP リクエスト。

    ヘッダ方針:
      Authorization / Content-Type は共通。x-opencode-session は session_id が
      渡されたとき（chat completions）だけ付与する。Embeddings API には公式
      仕様上不要なため付けない。
    リトライ方針:
      config.RETRY_TOTAL / RETRY_BACKOFF / RETRY_STATUS_FORCELIST（429 を含む）
      / RETRY_ALLOWED_METHODS を唯一の情報源とする。
    usage:
      レスポンスの usage を _accumulate_usage() で保持する。
    秘密情報:
      例外は config.mask_secrets を通してから CloudUnavailable にする。
    """
    if not enabled():
        raise _fail("cloud disabled (set %s=1)" % ENV_ENABLED)
    key_name = api_key_name()
    if key_name is None:
        raise _fail("no API key environment variable set")
    url = base_url().rstrip("/") + path
    headers: dict[str, str] = {
        "Authorization": "Bearer %s" % os.environ.get(key_name, ""),
        "Content-Type": "application/json",
    }
    if session_id is not None:
        headers["x-opencode-session"] = str(session_id)
    try:
        response = _shared_session().request(
            method, url, json=payload, headers=headers, timeout=timeout()
        )
        response.raise_for_status()
        data: Any = response.json()
    except (requests.RequestException, ValueError, TypeError) as exc:
        raise _fail(exc) from exc
    if not isinstance(data, dict):
        raise _fail("unexpected API response type: %s" % type(data).__name__)
    _accumulate_usage(data.get("usage"))
    return data


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    session_id: str | None = None,
) -> str:
    """OpenAI 互換 chat completions を呼び、本文文字列を返す（後方互換）。

    x-opencode-session には呼び出し単位の UUID を付与する。session_id を明示
    すればその値を使う（呼び出し側で相関 ID を渡せる）。省略時は uuid4。
    戻り値は従前どおり str。usage は last_usage() から参照する。
    """
    _reset_usage()
    sid = str(uuid.uuid4()) if session_id is None else str(session_id)
    data = _request(
        "/chat/completions",
        {"model": model(), "messages": messages, "temperature": temperature},
        session_id=sid,
    )
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
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
    """Go プランで利用可能な埋め込みモデル ID を定数から返す（通信なし）。"""
    return list(EMBEDDING_MODELS)


def fetch_models() -> list[str]:
    """任意: OpenAI 互換のモデル一覧から ID を取得する（実通信あり）。

    base_url() は既定で .../v1 を含むため MODELS_PATH("/models") を連結し、
    実質 /v1/models を呼ぶ。呼び出し側が明示したときだけ通信し、
    embedding_models() は従来どおり定数を返して自動取得しない。
    失敗は CloudUnavailable（理由はマスク済み）。
    """
    data = _request(MODELS_PATH, None, method="GET")
    items = data.get("data")
    if not isinstance(items, list):
        raise _fail("models response has no data list")
    return [
        str(item["id"])
        for item in items
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


def embed_model() -> str:
    configured = os.environ.get(ENV_EMBED_MODEL)
    if configured:
        return configured
    return "text-embedding-3-small"


def _embed_batch(
    texts: list[str], *, dimensions: int | None = None
) -> list[list[float]]:
    """1 リクエスト分の埋め込み呼び出し（内部用）。"""
    payload: dict[str, Any] = {
        "model": embed_model(),
        "input": texts,
    }
    if dimensions is not None:
        payload["dimensions"] = dimensions
    # session_id を渡さない＝ x-opencode-session を付けない（公式仕様）。
    data = _request("/embeddings", payload)
    try:
        vectors = [item["embedding"] for item in data["data"]]
    except (KeyError, TypeError) as exc:
        raise _fail(exc) from exc
    if len(vectors) != len(texts):
        raise _fail("embedding count mismatch")
    return vectors


# 1 リクエストあたりの入力上限（リクエストサイズとトークン制限を避ける）。
MAX_BATCH_SIZE = 64


def embed(
    texts: list[str],
    *,
    dimensions: int | None = None,
    batch_size: int | None = None,
) -> list[list[float]]:
    """OpenAI 互換 Embeddings API でベクトルを取得する（後方互換）。

    埋め込みリクエストには x-opencode-session を付けない（公式仕様）。
    大量入力は batch_size（既定 MAX_BATCH_SIZE）で分割し、入力順序どおりに
    結合する。batch_size=0 は明示的に拒否する。
    戻り値は従前どおり list[list[float]]。usage は全バッチの合計が last_usage()
    から参照できる（呼び出し直後に読み取ること）。認証情報の値は出さない。
    """
    size = MAX_BATCH_SIZE if batch_size is None else batch_size
    if size < 1:
        raise _fail("batch_size must be positive")
    _reset_usage()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), size):
        vectors.extend(
            _embed_batch(texts[start : start + size], dimensions=dimensions)
        )
    return vectors
