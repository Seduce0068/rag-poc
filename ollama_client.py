"""Ollama 埋め込み API の共通クライアント。

ingest / query / compare_embeddings などが同じ手順で Ollama の埋め込みを
呼ぶための薄いラッパー。設定値は config 由来、HTTP は共有 Session（Retry 付き）を使う。

通信前の接続先検証は呼び出し側（各 CLI の _check_endpoints）の責務。
このモジュールはエラー文言へ config.mask_secrets を適用し、URL / 認証情報 /
IP / パスを出力しない。
"""
from __future__ import annotations

import math
import sys
from typing import Any, Sequence

try:
    import requests
except ImportError:  # pragma: no cover - requests 未導入でも import は通す
    requests = None  # type: ignore[assignment]

from config import (
    EMBED_BATCH_SIZE,
    EMBED_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    RETRY_ALLOWED_METHODS,
    RETRY_BACKOFF,
    RETRY_STATUS_FORCELIST,
    RETRY_TOTAL,
    mask_secrets,
)

__all__ = ["embed"]

# /api/embed が件数不正を返したとき、/api/embeddings へ再送信する 1 バッチあたりの
# 上限。共有 Session の Retry（RETRY_TOTAL=3）が乗じると 4+4N リクエストまで
# 膨らむため、成功済みベクトルを再利用したうえで再送信件数を明示的に制限する。
_MAX_FALLBACK_RESENDS = 64


def _warn(message: str) -> None:
    """マスク済みの警告を stderr へ出す（JSON 出力を汚さない）。"""
    print(f"[ollama] {mask_secrets(message)}", file=sys.stderr, flush=True)


# 全呼び出しで共有する Session。query.py の _shared_session() と同じく、
# 最初の HTTP 呼び出しまで生成を遅延する（requests が stub された環境でも
# module import を通すため）。
_SESSION: Any | None = None


def _make_session() -> Any:
    """urllib3 Retry をマウントした共有 Session を作る（query.py と同等）。

    一時的障害 (429 / 500 / 502 / 503 / 504) では GET / POST / PUT を
    RETRY_TOTAL 回までリトライし、RETRY_BACKOFF で待つ。
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=list(RETRY_STATUS_FORCELIST),
        allowed_methods=frozenset(RETRY_ALLOWED_METHODS),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _shared_session() -> Any:
    """共有 Session を遅延生成して返す。"""
    global _SESSION
    if _SESSION is None:
        _SESSION = _make_session()
    return _SESSION


def _post(
    session: Any, url: str, payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """JSON を POST し、dict レスポンスを返す（例外文言はマスク済み）。"""
    try:
        response = session.post(url, json=payload, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - 外部サービス障害は上位で判断する
        raise RuntimeError(f"Ollama への接続に失敗: {mask_secrets(exc)}") from exc

    status = getattr(response, "status_code", None)
    if isinstance(status, int) and status >= 400:
        body = getattr(response, "text", "")
        raise RuntimeError(
            f"Ollama へのリクエストに失敗: HTTP {status} "
            f"{mask_secrets(str(body)[:300])}"
        )
    try:
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Ollama 応答の JSON 解析に失敗: {mask_secrets(exc)}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Ollama 応答が dict ではありません")
    return data


def _vector(value: Any) -> list[float] | None:
    """数値シーケンスを float ベクトルへ正規化する（不正/非有限なら None）。

    NaN / ±inf は下流（Qdrant への upsert や cos 類似度）で誤った値になるため、
    ``math.isfinite`` で明示的に拒否する。bool や非数値も従来どおり拒否する。
    """
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    try:
        numbers = [float(item) for item in value]
    except (OverflowError, ValueError):
        return None
    if not all(math.isfinite(number) for number in numbers):
        return None
    return numbers


def _batch_vectors(data: dict[str, Any], expected: int) -> list[list[float]] | None:
    """/api/embed の embeddings を期待件数分のベクトルへ変換する。"""
    raw = data.get("embeddings")
    if not isinstance(raw, list) or not raw:
        return None
    if all(isinstance(item, (list, tuple)) for item in raw):
        parsed = [_vector(item) for item in raw]
        if any(vector is None for vector in parsed):
            return None
        vectors = [vector for vector in parsed if vector is not None]
        return vectors if len(vectors) == expected else None
    # 旧実装互換: input 1 件のとき数値フラット配列を 1 ベクトルとして扱う。
    # _vector 側の math.isfinite チェックにより NaN / ±inf を含む配列も拒否される。
    if expected == 1:
        flat = _vector(raw)
        if flat is not None:
            return [flat]
    return None


def _partial_vectors(data: dict[str, Any], expected: int) -> list[list[float]]:
    """/api/embed の応答から先頭へ連続する有効ベクトルだけを取り出す。

    件数が期待より少ない／多い場合に成功済み分を再利用し、未取得分だけを
    /api/embeddings で補うために使う。不正値が出現した時点で打ち切ることで
    入力テキストとの位置対応を保持する（後ろの有効ベクトルは再利用しない）。
    """
    raw = data.get("embeddings")
    if not isinstance(raw, list) or not raw:
        return []
    if all(isinstance(item, (list, tuple)) for item in raw):
        vectors: list[list[float]] = []
        for item in raw:
            vector = _vector(item)
            if vector is None:
                break
            vectors.append(vector)
        return vectors[:expected]
    # フラット配列は input 1 件のときだけ 1 ベクトルとして再利用する。
    if expected == 1:
        flat = _vector(raw)
        return [flat] if flat is not None else []
    return []


def _check_dim(
    vectors: list[list[float]], validate_dim: int | None
) -> list[list[float]]:
    """validate_dim 指定時だけ次元を検証する（未指定なら何もしない）。"""
    if validate_dim is None:
        return vectors
    for vector in vectors:
        if len(vector) != validate_dim:
            raise RuntimeError(
                f"埋め込み次元が {len(vector)} です（期待: {validate_dim}）"
            )
    return vectors


def _embed_batch(
    session: Any,
    inputs: list[str],
    *,
    model: str,
    base_url: str,
    timeout: float,
    validate_dim: int | None,
) -> list[list[float]]:
    """1 バッチ分を /api/embed（失敗時 /api/embeddings）で埋め込む。"""
    primary_error: BaseException | None = None
    primary_data: dict[str, Any] | None = None
    vectors: list[list[float]] | None = None
    try:
        data = _post(
            session,
            f"{base_url}/api/embed",
            {"model": model, "input": inputs},
            timeout,
        )
        primary_data = data
        vectors = _batch_vectors(data, len(inputs))
    except RuntimeError as exc:
        primary_error = exc

    if vectors is not None:
        # 次元不一致はフォールバックでも解消しないため、ここで明示的に失敗させる。
        return _check_dim(vectors, validate_dim)

    # 件数不正でも先頭から連続する有効ベクトルは再利用し、未取得分だけを送る。
    # 全文本を再送信して RETRY_TOTAL を乗じる（4+4N）リクエスト膨張を避ける。
    reused = (
        _partial_vectors(primary_data, len(inputs))
        if primary_data is not None
        else []
    )
    missing = inputs[len(reused) :]
    if len(missing) > _MAX_FALLBACK_RESENDS:
        _warn(
            "埋め込みフォールバックの再送信上限を超過: "
            f"未取得 {len(missing)} 件 / 上限 {_MAX_FALLBACK_RESENDS} 件"
        )
        raise RuntimeError(
            "埋め込みフォールバックの再送信上限を超過しました"
        )

    if primary_error is None:
        primary_error = RuntimeError(
            "/api/embed から期待件数の embeddings を取得できませんでした"
        )
    _warn(
        "Ollama embed フォールバック: /api/embeddings へ "
        f"{len(missing)} 件を再送信（{len(inputs)} 件中 {len(reused)} 件を再利用）"
    )

    # 旧 API は 1 プロンプトずつ受け取るため、未取得分だけを 1 件ずつ呼ぶ。
    legacy: list[list[float]] = []
    for text in missing:
        try:
            data = _post(
                session,
                f"{base_url}/api/embeddings",
                {"model": model, "prompt": text},
                timeout,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Ollama embed に失敗: {mask_secrets(primary_error)}; "
                f"フォールバック /api/embeddings も失敗: {mask_secrets(exc)}"
            ) from exc
        vector = _vector(data.get("embedding"))
        if vector is None:
            raise RuntimeError(
                "/api/embeddings から embedding を取得できませんでした"
            )
        legacy.append(vector)

    # 再利用分 + 補完分で入力順を復元する。
    result = reused + legacy
    if len(result) != len(inputs):
        raise RuntimeError(
            f"埋め込み件数が不一致: 入力 {len(inputs)} / 応答 {len(result)}"
        )
    return _check_dim(result, validate_dim)


def embed(
    texts: Sequence[str],
    *,
    model: str | None = None,
    prefix: str = "",
    batch_size: int | None = None,
    validate_dim: int | None = None,
    url: str | None = None,
    timeout: float | None = None,
) -> list[list[float]]:
    """テキスト列を Ollama で埋め込み、全件分のベクトルを同じ順序で返す。

    * prefix は送信直前に各テキストへ付与し、入力の texts は変更しない。
      モデル別 prefix は ``config.embedding_prefixes(model)`` から取得して
      呼び出し側が渡す（既定は空文字＝既存モデルの挙動を変えない）。
    * batch_size が None / 不正なときは config.EMBED_BATCH_SIZE を使う。
    * validate_dim を指定したときだけ、各ベクトルの次元を検証する。
    * 空入力のときは HTTP 呼び出しをせず空リストを返す。
    * /api/embed が使えないときは /api/embeddings へフォールバックし、
      戻り値の件数が入力と一致することを保証する。
    """
    items = list(texts)
    if not items:
        return []

    resolved_model = model if model is not None else EMBED_MODEL
    base_url = (url if url is not None else OLLAMA_URL).rstrip("/")
    resolved_timeout = timeout if timeout is not None else OLLAMA_TIMEOUT
    size = batch_size if batch_size is not None and batch_size > 0 else EMBED_BATCH_SIZE

    session = _shared_session()
    vectors: list[list[float]] = []
    for offset in range(0, len(items), size):
        batch = items[offset : offset + size]
        inputs = [f"{prefix}{text}" if prefix else text for text in batch]
        vectors.extend(
            _embed_batch(
                session,
                inputs,
                model=resolved_model,
                base_url=base_url,
                timeout=resolved_timeout,
                validate_dim=validate_dim,
            )
        )
    return vectors
