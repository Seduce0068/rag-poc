"""opencodeGO の OpenAI 互換 API への薄いクライアント。

chat completions と OpenAI 互換 /embeddings の両方を提供する。埋め込みは
ローカル Ollama の代替として使えるが、既定では無効で明示的なオプトイン
（RAG_CLOUD_ENABLED=1）が必要。オプトインしない限り実通信は行わない。

認証情報は環境変数名のみを参照し、値はログ・例外・成果物に一切出力しない。
キー名の優先順は OPENCODE_GO_API_KEY（正規）→ OPENCODE_API_KEY（互換）。
利用可能な埋め込みモデルは EMBEDDING_MODELS 定数を参照する（任意で
fetch_models() が /v1/models を取得する）。API レスポンスの usage
（トークン数、あれば cost）は last_usage() から参照できる。

埋め込み次元は明示の dimensions 引数 → RAG_CLOUD_EMBED_DIMENSION（互換
RAG_CLOUD_DIMENSIONS）→ モデル既定（dimensions を送らない）の順で解決し、
不正値は CloudUnavailable で拒否する。
"""

from __future__ import annotations

import math
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
# 正規は単数形。複数形は移行期の互換として config の共通リゾルバが読む。
ENV_DIMENSIONS = config.ENV_CLOUD_DIMENSIONS  # 後方互換用の別名
ENV_EMBED_DIMENSION = config.ENV_CLOUD_EMBED_DIMENSION

# 正規（OPENCODE_GO_API_KEY）を優先し、OPENCODE_API_KEY は後方互換として残す。
KEY_NAMES = ("OPENCODE_GO_API_KEY", "OPENCODE_API_KEY")

# base_url() は既定で .../v1 を含むため、"/models" で実質 /v1/models を呼ぶ。
MODELS_PATH = "/models"


class CloudUnavailable(Exception):
    """クラウド API が使えない場合の例外（理由はマスク済み）。

    ``completed_batches`` は部分バッチ進捗の構造化情報（未発生なら None）。
    メッセージ末尾の ``(completed_batches=N)`` は後方互換のため維持する。
    """

    completed_batches: int | None = None

    def __init__(self, message: str, *, completed_batches: int | None = None) -> None:
        super().__init__(message)
        self.completed_batches = completed_batches


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


def _normalize_dimensions(value: object, label: str = "dimensions") -> int:
    """dimensions を正の int へ正規化する（不正値は CloudUnavailable）。

    値そのものは例外文に含めない（認証情報の誤設定などを漏らさないため）。
    bool は int の部分クラスだが、次元としては不正な値として扱う。
    """
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise _fail("%s must be a positive integer" % label)
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        raise _fail("%s must be a positive integer" % label) from None
    if normalized <= 0:
        raise _fail("%s must be a positive integer" % label)
    return normalized


def resolve_dimensions(dimensions: int | None = None) -> int | None:
    """embed() が使う埋め込み次元を解決する。

    優先順: 明示の dimensions 引数 > config.resolve_cloud_embed_dimension()
    （RAG_CLOUD_EMBED_DIMENSION 正規、RAG_CLOUD_DIMENSIONS 互換）> None。
    環境変数の読み出し・正規化は config の共通リゾルバに委譲する。不正値は
    CloudUnavailable を送出する（値自体は例外文に出さない）。
    """
    if dimensions is not None:
        return _normalize_dimensions(dimensions, "dimensions")
    try:
        return config.resolve_cloud_embed_dimension()
    except ValueError as exc:
        raise _fail(exc) from exc


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


# opencodeGO が usage.opencode.billing_source で報告する課金ソースの既知列挙値。
# この列挙以外の文字列は秘密値になり得るため保持しない（ホワイトリスト）。
# 出典: https://opencode.ai/docs/console/api/inference/#cost
BILLING_SOURCE_VALUES: tuple[str, ...] = (
    "free",
    "balance",
    "go",
    "go-plus",
    "byok",
)

# 直近の chat() / embed() が受け取った usage の数値項目（呼び出し単位で加算）。
_LAST_USAGE: dict[str, Any] | None = None
# 直近の usage.opencode.billing_source（加算しない「最後の値」・リセットで None）。
_LAST_BILLING_SOURCE: str | None = None
# 直近 1 リクエストの生 usage ホワイトリスト（last_raw_usage() の正本）。
_LAST_RAW_USAGE: dict[str, Any] | None = None


def _reset_usage() -> None:
    global _LAST_USAGE, _LAST_BILLING_SOURCE, _LAST_RAW_USAGE
    _LAST_USAGE = None
    _LAST_BILLING_SOURCE = None
    _LAST_RAW_USAGE = None


def _accumulate_usage(usage: object) -> None:
    """API の usage から数値項目だけを取り出し、呼び出し単位で加算保持する。

    文字列値は opencode.billing_source の既知列挙値のみ保持する
    （BILLING_SOURCE_VALUES のホワイトリスト）。cost が返る実装では
    cost / cost_usd 等の数値項目もそのまま加算される。nested な辞書
    （例: cost: {"usd": 0.01}）は "cost.usd" として平坦化する。

    opencode.allowance は opencodeGO の公式仕様で「要請単位の貢献ドル」
    （公式原文: "how much the request counted toward your usage limits"）
    と定義されるため、per-request 値と解釈して加算する。
    出典: https://opencode.ai/docs/console/api/inference/#cost
    """
    global _LAST_BILLING_SOURCE
    if not isinstance(usage, dict):
        return
    opencode = usage.get("opencode")
    if isinstance(opencode, dict):
        # billing_source は数値ではないため加算せず「最後の値」を保持する。
        # 列挙外・非文字列・長さ 32 超は無視し、保持は _reset_usage で None へ
        # クリアされる。
        source = opencode.get("billing_source")
        if (isinstance(source, str) and len(source) <= 32
                and source in BILLING_SOURCE_VALUES):
            _LAST_BILLING_SOURCE = source
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
                        and not isinstance(sub_value, bool)
                        and math.isfinite(sub_value)):
                    additions[flat] = sub_value
        elif isinstance(value, (int, float)) and math.isfinite(value):
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
    completion_tokens, total_tokens, cost / cost_usd）。opencodeGO が
    usage.opencode.billing_source を報告した場合のみ、既知列挙値を
    "opencode.billing_source" として含める（文字列値はこのホワイトリストのみ
    保持）。embed() は複数バッチの値を合計する。usage が無いレスポンスなら
    None。認証情報の値は含まない。

    cloud_bench 等の呼び出し側は chat() / embed() の直後に本関数を呼び、
    record / summary へ格納すれば良い（戻り値の str / list は従来どおり）。
    """
    if _LAST_USAGE is None and _LAST_BILLING_SOURCE is None:
        return None
    result = dict(_LAST_USAGE or {})
    if _LAST_BILLING_SOURCE is not None:
        result["opencode.billing_source"] = _LAST_BILLING_SOURCE
    return result


# last_raw_usage() の抽出ホワイトリスト（秘密値になり得る項目を通さない）。
_RAW_USAGE_TOP_KEYS: tuple[str, ...] = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost",
    "cost_usd",
)
_RAW_USAGE_OPENCODE_NUMERIC_KEYS: tuple[str, ...] = ("allowance", "list_cost")


def _extract_raw_usage(usage: object) -> dict[str, Any] | None:
    """直近 1 リクエストの usage からホワイトリスト項目のみを抽出する。

    ホワイトリスト: トップレベル数値の prompt_tokens / completion_tokens /
    total_tokens / cost / cost_usd、ネスト opencode の allowance / list_cost
    （数値）と billing_source（既知列挙値）。それ以外のキー・値は破棄する
    ため秘密値を含まない。usage が辞書でない場合・該当項目が無い場合は None。
    """
    if not isinstance(usage, dict):
        return None
    extracted: dict[str, Any] = {}
    for key in _RAW_USAGE_TOP_KEYS:
        value = usage.get(key)
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value)):
            extracted[key] = value
    opencode = usage.get("opencode")
    if isinstance(opencode, dict):
        for key in _RAW_USAGE_OPENCODE_NUMERIC_KEYS:
            value = opencode.get(key)
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value)):
                extracted["opencode.%s" % key] = value
        source = opencode.get("billing_source")
        if (isinstance(source, str) and len(source) <= 32
                and source in BILLING_SOURCE_VALUES):
            extracted["opencode.billing_source"] = source
    return extracted or None


def _record_raw_usage(usage: object) -> None:
    """直近 1 リクエストの usage ホワイトリストを保持する（内部用）。

    _request が _accumulate_usage(data.get("usage")) を呼ぶのと同じタイミ
    ングで呼ぶ。usage が辞書でなければ保持を None へ戻す（直近リクエストに
    usage が無いことを反映する）。保持は _reset_usage でクリアされる。
    """
    global _LAST_RAW_USAGE
    _LAST_RAW_USAGE = _extract_raw_usage(usage)


def last_raw_usage() -> dict[str, Any] | None:
    """直近 1 リクエストの生 usage ホワイトリストを返す（通信なし）。

    _request が usage を受け取ったタイミングで保持し、_reset_usage で
    クリアする。embed() で複数バッチに分割した場合は直近 1 リクエスト
    （最後のバッチ）の値を返す。ホワイトリスト以外のキーは含まないため
    秘密値を含まない。usage が無ければ None。
    """
    return dict(_LAST_RAW_USAGE) if _LAST_RAW_USAGE is not None else None


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
    # 直近 1 リクエストの生 usage ホワイトリストを同じタイミングで保持する
    # （last_raw_usage() の正本。allowance の実測値と per-request / 累積の判別
    # 材料として成果物側へ記録する）。
    _record_raw_usage(data.get("usage"))
    return data


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    session_id: str | None = None,
    isolated: bool = False,
) -> str:
    """OpenAI 互換 chat completions を呼び、本文文字列を返す（後方互換）。

    x-opencode-session には呼び出し単位の UUID を付与する。session_id を明示
    すればその値を使う（呼び出し側で相関 ID を渡せる）。省略時は uuid4。
    戻り値は従前どおり str。usage は last_usage() から参照する。

    1 呼び出しごとに usage をリセットし、セッション ID も生成するため、ある
    呼び出しの失敗は別の呼び出しの状態へ波及しない。``isolated=True`` を指定
    すると、想定外の例外まで CloudUnavailable に変換して個別失敗を呼び出し
    単位へ閉じ込める（既定 False は従来どおり例外をそのまま再送出）。
    """
    try:
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
    except CloudUnavailable:
        # 既にマスク済みの CloudUnavailable はそのまま再送出（二重変換しない）。
        raise
    except Exception as exc:  # noqa: BLE001 - isolated 指定時のみ閉じ込める
        if not isolated:
            raise
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


def expected_dim(model: str) -> int | None:
    """config.CLOUD_EMBED_DIMENSIONS からモデルの既定次元を返す（通信なし）。

    次元テーブルは config が唯一の情報源。未知モデルは None を返し、
    呼び出し側が RAG_CLOUD_EMBED_DIMENSION の明示を要求する。
    """
    return config.CLOUD_EMBED_DIMENSIONS.get(model)


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
    結合する。batch_size=0 / 負値 / bool は明示的に拒否する（bool は int の
    部分クラスだが、バッチ件数としては不正な値として扱う）。
    dimensions は明示引数 > RAG_CLOUD_EMBED_DIMENSION（互換
    RAG_CLOUD_DIMENSIONS）> モデル既定（送らない）の順で解決する。バッチ途中で
    失敗した場合は、成功済みバッチ数をマスク済みの CloudUnavailable メッセージ
    末尾に (completed_batches=N) として付与し、構造化属性にも保持する。
    戻り値は従前どおり list[list[float]]。usage は全バッチの合計が last_usage()
    から参照できる（呼び出し直後に読み取ること）。認証情報の値は出さない。
    """
    # bool は int の部分クラスなので True が size=1 として通るのを明示的に防ぐ。
    # 0 / 負値の拒否は従来どおり size < 1 で行う。
    if isinstance(batch_size, bool):
        raise _fail("batch_size must be a positive integer")
    size = MAX_BATCH_SIZE if batch_size is None else batch_size
    if size < 1:
        raise _fail("batch_size must be positive")
    dims = resolve_dimensions(dimensions)
    _reset_usage()
    vectors: list[list[float]] = []
    completed_batches = 0
    for start in range(0, len(texts), size):
        try:
            vectors.extend(
                _embed_batch(texts[start : start + size], dimensions=dims)
            )
        except Exception as exc:
            reason = (
                str(exc)
                if isinstance(exc, CloudUnavailable)
                else "embed failed: %s" % exc
            )
            raise CloudUnavailable(
                "%s (completed_batches=%d)"
                % (config.mask_secrets(reason), completed_batches),
                completed_batches=completed_batches,
            ) from exc
        completed_batches += 1
    return vectors
