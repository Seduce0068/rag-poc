"""コーパス (.md) をチャンク化し、Ollama で埋め込み、Qdrant へ投入する。

起動例（リポジトリルート、rag_poc 内のどちらからでも可）::

    python3 rag_poc/ingest.py
    python3 rag_poc/ingest.py --help
    python3 rag_poc/ingest.py --corpus-dir rag_poc/corpus-subset --collection rag_poc_docs
    python3 rag_poc/ingest.py --recreate --yes --corpus-dir rag_poc/corpus-subset

`--recreate` はコレクションを DELETE → PUT（再作成）→ 全件再投入する。
DELETE の実実行には `--yes` の同時指定が必須。`--yes` が無い場合は削除を行わず、
終了コード 2 で停止する。

`--recreate --yes` の実行時は .recreate-audit.json（チェックポイントとは別）
へ承認・実効 vector_size をマスク済みで原子的に監査記録する。DELETE 以降の
途中失敗では「コレクション <name> は削除済みです。再投入が必要です」を
マスク済み stderr で案内し、RuntimeError は終了コード 1 で扱う。

設定は config.py（環境変数で上書き可）を参照する。
    OLLAMA_URL / EMBED_MODEL / QDRANT_URL / COLLECTION /
    CHUNK_SIZE / CHUNK_OVERLAP / CORPUS_DIR / RAG_INGEST_RESUME

CLI の --corpus-dir / --collection を明示すると、対応する環境変数より
優先される（未指定なら従来どおり env 設定を使う）。

RAG_INGEST_RESUME=1 のときは .ingest-checkpoint.json を読み、
完了済みバッチの upsert をスキップする（既定 0 = 最初から）。

注: point ID を blake2b 64bit に変更したため、旧 ID で投入済みの
コレクションは再作成（または既存ポイント削除）が必要。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence

try:
    import requests
except ImportError:  # requests 未導入でも --help / 設定確認は動かす
    requests = None  # type: ignore[assignment]

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION,
    CORPUS_DIR,
    EMBED_MODEL,
    embedding_prefixes,
    expected_vector_size,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    PAYLOAD_CHUNK_INDEX_KEY,
    PAYLOAD_SOURCE_KEY,
    PAYLOAD_TEXT_KEY,
    PAYLOAD_TITLE_KEY,
    QDRANT_TIMEOUT,
    QDRANT_URL,
    REPO_ROOT,
    VECTOR_SIZE,
    mask_secrets,
    is_fatal_endpoint_problem,
    qdrant_collection_path,
    resolve_corpus_dir as config_resolve_corpus_dir,
    validate_endpoint,
    validate_settings,
)

try:
    # Ollama 埋め込みの共通 API。モデル・prefix・次元検証・バッチ・リトライを
    # 一箇所に集約するため、ここではembed 経由で呼び出す。
    from ollama_client import embed as _ollama_embed
except ImportError:  # 共通クライアント未導入でも --help / 設定確認は動かす
    _ollama_embed = None  # type: ignore[assignment]

# 1 リクエストあたりの件数。
EMBED_BATCH_SIZE = 16
UPSERT_BATCH_SIZE = 64

# 旧 CRC32 ID 残存チェック用の scroll ページング設定。
# 先頭ページだけでは後半バッチに混在した旧 ID を見逃すため、
# MAX_SCROLL_PAGES を上限にコレクション全体を走査する。
SCROLL_PAGE_SIZE = 256
MAX_SCROLL_PAGES = 100

# 再開用チェックポイント（このファイルと同じディレクトリ）。
CHECKPOINT_PATH = Path(__file__).resolve().parent / ".ingest-checkpoint.json"

# --recreate --yes の監査証跡（チェックポイントとは別ファイル）。
# 破壊操作の承認・実行パラメータを原子的に記録する。
AUDIT_PATH = Path(__file__).resolve().parent / ".recreate-audit.json"

# CLI --corpus-dir が明示されたときだけ使う。None のときは config 委譲
# （= 既存の env 動作）をそのまま維持する。
_CLI_CORPUS_DIR: str | None = None

_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*$")


# ---------------------------------------------------------------------------
# 埋め込み設定（passage prefix / モデル別次元）
# ---------------------------------------------------------------------------


def _passage_prefix() -> str:
    """config.embedding_prefixes(EMBED_MODEL) から passage 側 prefix を返す。

    config 側の戻り値が dict (passage/document/doc) でも文字列でも扱えるようにし、
    prefix 未定義のモデルでは空文字を返す（= 既存挙動を維持）。
    """
    prefixes = embedding_prefixes(EMBED_MODEL)
    # NamedTuple (EmbeddingPrefixes) は tuple 継承だが str/dict ではないため
    # 属性アクセスを最優先で確認する（Ruri の「文章: 」が失われる問題を防止）。
    passage = getattr(prefixes, "passage", None)
    if isinstance(passage, str):
        return passage
    if isinstance(prefixes, str):
        return prefixes
    if isinstance(prefixes, dict):
        for key in ("passage", "document", "doc"):
            value = prefixes.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def resolved_vector_size() -> int:
    """埋め込み検証・コレクション作成で使う次元を解決する。

    expected_vector_size(EMBED_MODEL) を権威として優先する。既存の
    RAG_VECTOR_SIZE (config.VECTOR_SIZE) 上書きも尊重し、両者が不一致の
    場合は警告を出して expected を採用する（モデルの実出力次元に合わせ、
    モデル切り替え時に古い既定値でブロックされないようにするため）。
    """
    expected = expected_vector_size(EMBED_MODEL)
    if VECTOR_SIZE != expected:
        print(
            f"[ingest] 警告: RAG_VECTOR_SIZE={VECTOR_SIZE} は "
            f"EMBED_MODEL={EMBED_MODEL!r} の期待次元 {expected} と不一致です。"
            f"期待次元 {expected} を採用します。",
            file=sys.stderr,
            flush=True,
        )
    return expected


# ---------------------------------------------------------------------------
# パス・チャンク
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """このファイルのある rag_poc の親ディレクトリ。"""
    return Path(__file__).resolve().parent.parent


def _resolve_cli_corpus_dir(value: str) -> Path:
    """CLI 指定の CORPUS_DIR を config と同じ規則で絶対パス化する。"""
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    root = REPO_ROOT.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            "CORPUS_DIR はリポジトリ配下である必要があります: "
            f"{mask_secrets(str(resolved))}"
        )
    return resolved


def resolve_corpus_dir() -> Path:
    """コーパスディレクトリを解決する（cwd 非依存・リポジトリ配下のみ）。

    既定は config.resolve_corpus_dir() に委譲し、evaluate と解決順を統一する。
    CLI --corpus-dir が指定された場合のみ、同じ規則で CLI 値を解決する。
    """
    if _CLI_CORPUS_DIR is None:
        resolved = config_resolve_corpus_dir()
        label = CORPUS_DIR
    else:
        resolved = _resolve_cli_corpus_dir(_CLI_CORPUS_DIR)
        label = _CLI_CORPUS_DIR
    if not resolved.is_dir():
        raise FileNotFoundError(f"CORPUS_DIR が見つかりません: {label!r} ({resolved})")
    return resolved


def iter_source_files(corpus_dir: Path) -> Iterator[Path]:
    """コーパス配下の .md を決定的な順序で列挙する。"""
    for path in sorted(corpus_dir.rglob("*.md")):
        if path.is_file():
            yield path


def _heading_text(line: str) -> str:
    match = _HEADING_RE.match(line)
    return match.group(2).strip() if match else line.strip()


def _split_long_text(text: str) -> list[str]:
    """CHUNK_SIZE / CHUNK_OVERLAP で文字単位のウィンドウに分割する。"""
    if len(text) <= CHUNK_SIZE:
        return [text] if text else []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    if step <= 0:
        raise ValueError(
            f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) は CHUNK_SIZE ({CHUNK_SIZE}) より小さく必要です"
        )
    pieces: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + CHUNK_SIZE, length)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        start += step
    return pieces


def _point_id(source: str, chunk_index: int) -> int:
    """再実行しても同じになる安定 ID（blake2b 64bit、Qdrant の uint64 に収まる）。"""
    digest = hashlib.blake2b(
        f"{source}\0{chunk_index}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, byteorder="big")


def iter_chunks(path: Path, root: Path) -> Iterator[dict]:
    """1 ファイルを見出し区切り＋サイズ分割してポイント素片を返す。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        source = str(path.resolve().relative_to(root))
    except ValueError:
        source = str(path.resolve())

    # 見出し行でセクションを切る。見出し行はセクション先頭に残す。
    sections: list[tuple[str, str]] = []
    heading: str | None = None
    first_heading: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if _HEADING_RE.match(line):
            if first_heading is None:
                first_heading = _heading_text(line)
            if buf:
                body = "\n".join(buf).strip()
                if body:
                    sections.append((heading or path.stem, body))
            heading = _heading_text(line)
            buf = [line]
        else:
            buf.append(line)
    if buf:
        body = "\n".join(buf).strip()
        if body:
            sections.append((heading or path.stem, body))

    # payload[PAYLOAD_TITLE_KEY] はドキュメントの先頭見出し（無ければファイル名）。
    doc_title = first_heading or path.stem

    chunk_index = 0
    for _section_heading, body in sections:
        first_line, separator, rest = body.partition("\n")
        if _HEADING_RE.match(first_line):
            # 見出しを全チャンクに付けて文脈を保つ。
            remainder = rest.strip() if separator else ""
            pieces = _split_long_text(remainder)
            prefix: str | None = first_line
            if not pieces:
                pieces = [""]  # 見出しのみのセクション
        else:
            # フロントマター等、見出しの無い先頭部分はそのまま分割する。
            pieces = _split_long_text(body)
            prefix = None
        for piece in pieces:
            chunk_text = f"{prefix}\n{piece}" if prefix and piece else (piece or prefix or body)
            yield {
                "id": _point_id(source, chunk_index),
                "vector": None,  # embed_texts() で埋める
                "payload": {
                    PAYLOAD_SOURCE_KEY: source,
                    PAYLOAD_TITLE_KEY: doc_title,
                    PAYLOAD_TEXT_KEY: chunk_text,
                    PAYLOAD_CHUNK_INDEX_KEY: chunk_index,
                },
            }
            chunk_index += 1


# ---------------------------------------------------------------------------
# HTTP 処理
# ---------------------------------------------------------------------------


def _url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


def _raise_for_status(response: requests.Response, context: str) -> None:
    if response.status_code >= 400:
        # サーバ本文には URL / ホスト / 認証情報が含まれ得るため共通マスクを適用。
        raise RuntimeError(
            f"{context} に失敗: HTTP {response.status_code} {mask_secrets(response.text[:300])}"
        )


def _make_session() -> requests.Session:
    """全 HTTP 呼び出しで使う共通 Session（リトライ付き）を作る。

    requests が stub されたテスト環境でも module import を通すため、
    adapters / Retry は関数内で読み込む。
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        # embed は冪等なので POST もリトライ対象に含める。
        allowed_methods=frozenset({"GET", "PUT", "POST", "HEAD", "OPTIONS", "DELETE"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _check_legacy_ids(session: requests.Session) -> None:
    """旧 CRC32 ID の残存を検出したら中止して再作成手順を案内する。

    Qdrant の scroll API でコレクションを MAX_SCROLL_PAGES ページまで走査し、
    1 件でも旧 CRC32 ID があれば即座に RuntimeError を送出する。先頭 10 件
    だけを確認する従来方式では、後半バッチに旧 ID が混在していても
    ``all(...)`` が真になり見逃されるため、ページングで全件を確認する。

    bool は int のサブクラスなので ``isinstance(..., int)`` では ``id=True``
    を旧 ID と誤判定する。``type(...) is int`` で厳密に判定する。
    """
    name = qdrant_collection_path(COLLECTION)
    url = _url(QDRANT_URL, f"/collections/{name}/points/scroll")
    offset = None
    for _ in range(MAX_SCROLL_PAGES):
        body: dict[str, object] = {"limit": SCROLL_PAGE_SIZE, "with_payload": False}
        if offset is not None:
            body["offset"] = offset
        response = session.post(url, json=body, timeout=QDRANT_TIMEOUT)
        if response.status_code != 200:
            raise RuntimeError(
                "旧 ID の確認のため Qdrant scroll に失敗しました "
                "(HTTP %s)。コレクション状態を確認してください。" % response.status_code
            )
        try:
            result = response.json().get("result", {}) or {}
            points = result.get("points", []) or []
        except (ValueError, AttributeError):
            raise RuntimeError(
                "旧 ID の確認のため Qdrant scroll 応答を解析できませんでした。"
            )
        # CRC32 は 32bit、blake2b 64bit は通常 2**32 を超える。
        # 1 件でも旧 ID があれば混在として検出する（all ではなく any）。
        if any(
            type(point.get("id")) is int and 0 <= point["id"] < 2**32
            for point in points
        ):
            raise RuntimeError(
                "旧 CRC32 ID のポイントが残存しています。コレクションを削除してから再投入してください。"
            )
        next_offset = result.get("next_page_offset")
        if next_offset is None:
            break
        offset = next_offset
    else:
        # MAX_SCROLL_PAGES を使い切った = 末尾まで確認できていない。
        _warn("旧 ID 確認がページ上限に到達しました。末尾の旧 ID は未確認です。")


def ensure_collection(session: requests.Session, vector_size: int | None = None) -> bool:
    """存在しなければモデル期待次元の Cosine コレクションを作る。既存は触らない。"""
    size = vector_size if vector_size is not None else resolved_vector_size()
    name = qdrant_collection_path(COLLECTION)
    url = _url(QDRANT_URL, f"/collections/{name}")

    response = session.get(url, timeout=QDRANT_TIMEOUT)
    if response.status_code == 200:
        _check_legacy_ids(session)
        return False
    if response.status_code != 404:
        _raise_for_status(response, "Qdrant コレクション確認")

    response = session.put(
        url,
        json={"vectors": {"size": size, "distance": "Cosine"}},
        timeout=QDRANT_TIMEOUT,
    )
    _raise_for_status(response, "Qdrant コレクション作成")
    return True


def _validate_recreate_dimensions(vector_size: int) -> None:
    """DELETE 前に埋め込み次元が有効であることを検証する。

    resolved_vector_size() と EMBED_MODEL の整合は main() の
    validate_settings() で既に担保されているため、ここでは破壊操作の前に
    実効値が正の int であることを二重に確認する（fail-closed）。
    """
    if isinstance(vector_size, bool) or not isinstance(vector_size, int) or vector_size <= 0:
        raise RuntimeError(
            f"埋め込み次元が不正です: {vector_size!r}（正の整数が必要です）"
        )


def recreate_collection(
    session: requests.Session,
    vector_size: int | None = None,
    on_deleted: Callable[[], None] | None = None,
) -> None:
    """--recreate --yes 専用: DELETE → PUT でコレクションを作り直す。

    DELETE は呼び出し側の ``--yes`` ガードで既に承認済みであることが前提。
    既存コレクションが無い場合（HTTP 404）は削除済みとして扱い、PUT で新規作成する。
    PUT のボディは ensure_collection() と同じ（Cosine / モデル期待次元）。

    ``on_deleted`` は DELETE 成功（または 404）直後に呼ばれ、呼び出し側が
    「削除済み」状態を記録するために使う。PUT 以降の失敗でも同じ経路で
    利用者へ案内できるようにするためである。
    """
    size = vector_size if vector_size is not None else resolved_vector_size()
    # DELETE の前に次元を検証する（不正なら何も削除しない）。
    _validate_recreate_dimensions(size)
    name = qdrant_collection_path(COLLECTION)
    url = _url(QDRANT_URL, f"/collections/{name}")

    response = session.delete(url, timeout=QDRANT_TIMEOUT)
    if response.status_code != 404:
        _raise_for_status(response, "Qdrant コレクション削除")
    if on_deleted is not None:
        on_deleted()

    response = session.put(
        url,
        json={"vectors": {"size": size, "distance": "Cosine"}},
        timeout=QDRANT_TIMEOUT,
    )
    _raise_for_status(response, "Qdrant コレクション再作成")


def embed_texts(
    session: requests.Session,
    texts: Sequence[str],
    vector_size: int | None = None,
) -> list[list[float]]:
    """Ollama 埋め込みを ollama_client.embed 経由で取得する。

    passage prefix は共通クライアント側で各テキストへ適用するため、元の text
    (payload に保存される値) は変更しない。次元はモデル期待値で検証する。
    ``session`` は既存 call site / テスト互換のため受け取るが、HTTP セッション
    とリトライは共通クライアント側に委譲したため、ここでは使用しない。
    """
    if _ollama_embed is None:
        raise RuntimeError("ollama_client.embed を import できませんでした")
    expected_size = vector_size if vector_size is not None else resolved_vector_size()
    prefix = _passage_prefix()
    vectors = _ollama_embed(
        list(texts),
        model=EMBED_MODEL,
        prefix=prefix,
        batch_size=EMBED_BATCH_SIZE,
        validate_dim=expected_size,
        url=OLLAMA_URL,
        timeout=OLLAMA_TIMEOUT,
    )
    # 既存の進捗ログを維持する（共通クライアントがバッチ単位で送信する）。
    total_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    for batch_index, offset in enumerate(range(0, len(texts), EMBED_BATCH_SIZE)):
        count = min(EMBED_BATCH_SIZE, len(texts) - offset)
        print(
            f"[ingest] embed batch={batch_index + 1}/{total_batches} texts={count}",
            flush=True,
        )
    return vectors


def upsert_points(
    session: requests.Session,
    points: Sequence[dict],
    completed_batches: int = 0,
    on_batch_done: Callable[[int], None] | None = None,
) -> int:
    """Qdrant /collections/{name}/points へバッチ投入する。

    completed_batches 件目までのバッチはチェックポイント済みとしてスキップする。
    実際に投入した点数を返す。
    """
    name = qdrant_collection_path(COLLECTION)
    url = _url(QDRANT_URL, f"/collections/{name}/points") + "?wait=true"
    total_batches = (len(points) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE
    upserted_points = 0
    for batch_index, offset in enumerate(range(0, len(points), UPSERT_BATCH_SIZE)):
        batch_no = batch_index + 1
        if batch_index < completed_batches:
            print(
                f"[ingest] upsert batch={batch_no}/{total_batches} skipped (checkpoint)",
                flush=True,
            )
            continue
        batch = list(points[offset : offset + UPSERT_BATCH_SIZE])
        response = session.put(
            url,
            json={"points": batch},
            timeout=QDRANT_TIMEOUT,
        )
        _raise_for_status(response, "Qdrant points upsert")
        upserted_points += len(batch)
        print(
            f"[ingest] upsert batch={batch_no}/{total_batches} points={len(batch)}",
            flush=True,
        )
        if on_batch_done is not None:
            on_batch_done(batch_no)
    return upserted_points


# ---------------------------------------------------------------------------
# チェックポイント（RAG_INGEST_RESUME=1 のときだけ有効）
# ---------------------------------------------------------------------------


def _resume_enabled() -> bool:
    """RAG_INGEST_RESUME=1 のときだけチェックポイントを使う。"""
    return os.environ.get("RAG_INGEST_RESUME", "0") == "1"


def _checkpoint_signature(points: Sequence[dict]) -> str:
    """ID と本文の fingerprint。コーパス変更時は再開を無効化する。"""
    digest = hashlib.blake2b(digest_size=8)
    for point in points:
        digest.update(str(point["id"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(point["payload"][PAYLOAD_TEXT_KEY]).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _warn(message: str) -> None:
    """マスク済みの警告を stderr へ出す。"""
    print(f"[ingest] {mask_secrets(message)}", file=sys.stderr, flush=True)


def _warn_collection_deleted() -> None:
    """再作成途中の失敗時に「削除済み」をマスク済み stderr で明示する。"""
    _warn(f"コレクション {COLLECTION} は削除済みです。再投入が必要です")


def _warn_checkpoint_invalid(reason: str) -> None:
    """チェックポイント無効化の理由をマスクして stderr へ警告する。"""
    print(
        f"[ingest] {mask_secrets('checkpoint invalidated: ' + reason)}",
        file=sys.stderr,
        flush=True,
    )


def _verify_qdrant_state(
    session: requests.Session,
    points: Sequence[dict],
    completed_batches: int,
    vector_size: int,
) -> bool:
    """チェックポイントと Qdrant の実状態が整合するか検証する。

    確認項目:
      1. /collections/{name}/points/count の実件数が、先头 completed_batches
         バッチのポイント数合計と一致すること。
      2. /collections/{name} の vectors.size / distance が期待値
         (vector_size, Cosine) と一致すること。

    不整合・応答不正・取得失敗の場合はマスク済み警告を stderr に出し False を
    返す。呼び出し側はチェックポイントを破棄して全件再投入する。
    """
    name = qdrant_collection_path(COLLECTION)

    # 期待件数 = 先头 completed_batches バッチの len 合計（末尾バッチは短い）。
    total_batches = (len(points) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE
    batch_count = max(0, min(completed_batches, total_batches))
    expected_count = sum(
        len(points[offset : offset + UPSERT_BATCH_SIZE])
        for offset in range(0, batch_count * UPSERT_BATCH_SIZE, UPSERT_BATCH_SIZE)
    )

    # 実件数: Qdrant の count API は POST /points/count {"exact": true}。
    try:
        count_response = session.post(
            _url(QDRANT_URL, f"/collections/{name}/points/count"),
            json={"exact": True},
            timeout=QDRANT_TIMEOUT,
        )
        if count_response.status_code != 200:
            _warn_checkpoint_invalid(
                f"Qdrant point count request failed "
                f"(HTTP {count_response.status_code})"
            )
            return False
        actual_count = int(count_response.json()["result"]["count"])
    except Exception as exc:  # network / JSON / キー欠落はすべて無効扱い
        _warn_checkpoint_invalid(f"Qdrant point count check failed: {exc}")
        return False

    if actual_count != expected_count:
        _warn_checkpoint_invalid(
            f"Qdrant point count mismatch "
            f"(expected={expected_count}, actual={actual_count})"
        )
        return False

    # コレクション設定: size / distance が期待値か。
    try:
        info_response = session.get(
            _url(QDRANT_URL, f"/collections/{name}"),
            timeout=QDRANT_TIMEOUT,
        )
        if info_response.status_code != 200:
            _warn_checkpoint_invalid(
                f"Qdrant collection info request failed "
                f"(HTTP {info_response.status_code})"
            )
            return False
        vectors = info_response.json()["result"]["config"]["params"]["vectors"]
    except Exception as exc:  # network / JSON / キー欠落はすべて無効扱い
        _warn_checkpoint_invalid(f"Qdrant collection info check failed: {exc}")
        return False

    # vectors は単純 {"size": .., "distance": ..} と named vectors 辞書があり得る。
    if isinstance(vectors, dict):
        if "size" in vectors:
            config = vectors
        else:
            config = next(
                (value for value in vectors.values() if isinstance(value, dict)),
                {},
            )
    else:
        config = {}
    actual_size = config.get("size")
    actual_distance = config.get("distance")

    if actual_size != vector_size:
        _warn_checkpoint_invalid(
            f"Qdrant vector size mismatch "
            f"(expected={vector_size}, actual={actual_size})"
        )
        return False
    if actual_distance != "Cosine":
        _warn_checkpoint_invalid(
            f"Qdrant distance mismatch (expected=Cosine, actual={actual_distance})"
        )
        return False
    return True


def _load_checkpoint(
    points: Sequence[dict],
    session: requests.Session | None = None,
    vector_size: int | None = None,
) -> int:
    """有効なチェックポイントから完了済みバッチ数を返す（無効なら 0）。

    session が渡された場合は、値を返す直前に _verify_qdrant_state() で
    Qdrant の実状態を検証し、不一致なら 0（＝全件再投入）を返す。
    session 省略時はローカル検証のみ（単体テスト等の直接呼び出し用）。
    """
    if not _resume_enabled():
        return 0
    try:
        state = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not isinstance(state, dict):
        return 0
    expected_size = vector_size if vector_size is not None else resolved_vector_size()
    if (
        state.get("collection") != COLLECTION
        or state.get("vector_size") != expected_size
        or state.get("embed_model") != EMBED_MODEL
        or state.get("signature") != _checkpoint_signature(points)
    ):
        return 0
    completed = state.get("completed_batches", 0)
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
        return 0
    total_batches = (len(points) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE
    completed = min(completed, total_batches)
    if completed and session is not None:
        size = vector_size if vector_size is not None else resolved_vector_size()
        if not _verify_qdrant_state(session, points, completed, size):
            return 0
    return completed


def _save_checkpoint(
    points: Sequence[dict], completed_batches: int, vector_size: int | None = None
) -> None:
    """バッチ完了時にチェックポイントを原子的に更新する。

    vector_size には実効値（resolved_vector_size()）を渡す。省略時は
    resolved_vector_size() を参照し、生設定の RAG_VECTOR_SIZE は記録しない。
    """
    if not _resume_enabled():
        return
    state = {
        "collection": COLLECTION,
        "vector_size": (
            vector_size if vector_size is not None else resolved_vector_size()
        ),
        "embed_model": EMBED_MODEL,
        "signature": _checkpoint_signature(points),
        "completed_batches": completed_batches,
    }
    tmp_path = CHECKPOINT_PATH.with_name(CHECKPOINT_PATH.name + ".tmp")
    tmp_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(CHECKPOINT_PATH)


def _record_recreate_audit(vector_size: int) -> None:
    """--recreate --yes の承認・実行を監査証跡として原子的に記録する。

    CHECKPOINT_PATH とは別の AUDIT_PATH（rag_poc/.recreate-audit.json）へ
    最新1件をトップレベルに書き、過去分を ``history`` へ追記する。
    RFC3339 の UTC オフセット付きタイムスタンプ・コレクション名・実効
    vector_size・``--yes`` 承認済みフラグをすべてマスクして保存する。
    書き込みは tmp → replace の原子的更新で、途中失敗時に半端なファイルが
    残らないようにする。記録に失敗したら fail-closed で RuntimeError とし、
    監査なしに DELETE へ進まないようにする。
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "collection": mask_secrets(COLLECTION),
        "vector_size": vector_size,
        "yes_approved": True,
    }
    history: list[dict] = []
    try:
        existing = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            past = existing.get("history")
            if isinstance(past, list):
                history = [item for item in past if isinstance(item, dict)]
            elif all(
                key in existing for key in ("timestamp", "collection", "vector_size")
            ):
                # 旧形式（トップレベル1件のみ）はそのまま履歴へ繰り入れる。
                history = [existing]
    except (OSError, ValueError, TypeError):
        pass

    state = dict(record)
    state["history"] = history
    tmp_path = AUDIT_PATH.with_name(AUDIT_PATH.name + ".tmp")
    try:
        tmp_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(AUDIT_PATH)
    except OSError as exc:
        raise RuntimeError(f"監査証跡の記録に失敗しました: {exc}") from exc


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------


def _check_endpoints() -> None:
    """Ollama / Qdrant の接続先を検証する（既定はループバックのみ許可）。

    ``RAG_ALLOW_REMOTE_ENDPOINTS=1`` は allowlist + https が揃ったリモートだけを
    緩和する。allowlist 不一致・平文 HTTP・不正 scheme/port は
    ``config.is_fatal_endpoint_problem()`` が真を返すため、オプトインに関係なく
    致命として停止する（文字列一致による格下げはしない）。警告は mask_secrets
    を通して stderr へ出し、URL / 認証情報 / IP / パスをマスクする。
    """
    problems = [
        *validate_endpoint(OLLAMA_URL, "OLLAMA_URL"),
        *validate_endpoint(QDRANT_URL, "QDRANT_URL"),
    ]
    if not problems:
        return

    allow_remote = os.environ.get("RAG_ALLOW_REMOTE_ENDPOINTS") == "1"
    fatal = [p for p in problems if is_fatal_endpoint_problem(p)]
    warnings = [p for p in problems if not is_fatal_endpoint_problem(p)]

    for problem in warnings:
        message = f"接続先検証: {problem}"
        if allow_remote:
            message += " (RAG_ALLOW_REMOTE_ENDPOINTS=1 のため警告のみで継続)"
        print(f"[ingest] {mask_secrets(message)}", file=sys.stderr, flush=True)
    for problem in fatal:
        print(
            f"[ingest] {mask_secrets('接続先検証: ' + problem)} (致命)",
            file=sys.stderr,
            flush=True,
        )

    if fatal or not allow_remote:
        detail = "; ".join(str(p) for p in (fatal or problems))
        raise RuntimeError(
            "リモート endpoint は拒否されました。致命的な問題: "
            + mask_secrets(detail)
            + "（scheme / port / 許可リストを確認してください）"
        )


# 接続失敗時に stderr へ出す固定メッセージ（秘密情報を含まない）。
_CONNECTION_ERROR_MESSAGE = (
    "OLLAMA_URL/QDRANT_URL への接続に失敗。"
    "README の curl 生存確認を実行してください"
)


def _connection_exceptions() -> tuple[type[BaseException], ...]:
    """requests の ConnectionError / Timeout を取得する。

    requests が stub / 未導入でも import と --help が通るよう、属性が無い
    場合は空タプル（＝どの例外も捕捉しない）を返す。
    """
    module_exceptions = getattr(requests, "exceptions", None)
    if module_exceptions is None:
        return ()
    classes: list[type[BaseException]] = []
    for name in ("ConnectionError", "Timeout"):
        cls = getattr(module_exceptions, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            classes.append(cls)
    return tuple(classes)


_CONNECTION_EXCEPTIONS = _connection_exceptions()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI 引数を解析する（--help / 未知オプションは argparse が処理）。"""
    parser = argparse.ArgumentParser(
        description="コーパスをチャンク化し Ollama で埋め込み Qdrant へ投入する",
        epilog=(
            "未指定の設定は config.py / 環境変数から読み込む。"
            "--corpus-dir / --collection は環境変数より優先される。"
        ),
    )
    parser.add_argument(
        "--corpus-dir",
        default=None,
        metavar="DIR",
        help="コーパスディレクトリ（リポジトリ配下の相対パス / 絶対パス。既定: RAG_CORPUS_DIR）",
    )
    parser.add_argument(
        "--collection",
        default=None,
        metavar="NAME",
        help="Qdrant コレクション名（既定: RAG_COLLECTION）",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "コレクションを削除して再作成し全件再投入する"
            "（実削除には --yes が必須。--yes 無しでは終了コード 2 で停止）"
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="--recreate の実削除を承認する（単独指定では何も削除しない）",
    )
    return parser.parse_args(argv)


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    """明示された CLI 引数だけを env 由来の既定値より優先させる。"""
    global CORPUS_DIR, COLLECTION, _CLI_CORPUS_DIR
    if args.corpus_dir is not None:
        CORPUS_DIR = args.corpus_dir
        _CLI_CORPUS_DIR = args.corpus_dir
    if args.collection is not None:
        COLLECTION = args.collection


def main(argv: list[str] | None = None) -> int:
    """投入処理のエントリポイント。

    argv=None（既定）なら sys.argv を使うため、既存の直接呼び出し
    ``main()`` は後方互換でそのまま動く。接続失敗はトレースバックを
    出さず、マスク済みメッセージを stderr に出して 1 を返す。
    """
    args = _parse_args(argv)
    _apply_cli_overrides(args)

    # --recreate の破壊操作ガード。--yes が無い場合は HTTP セッションを一切作らず、
    # コレクションを削除せずに終了コード 2 で停止する。
    if args.recreate and not args.yes:
        print(
            f"コレクション {COLLECTION} を削除して再作成します。"
            "実行するには --yes を指定してください",
            file=sys.stderr,
            flush=True,
        )
        return 2

    # DELETE が完了したかどうか。PUT / 埋め込み / 投入の途中失敗時に
    # 「コレクションは削除済み」案内を出すための状態（スレッドローカルな単純フラグ）。
    deleted = False

    try:
        _check_endpoints()

        problems = validate_settings()
        if problems:
            raise RuntimeError("設定エラー: " + "; ".join(problems))

        corpus_dir = resolve_corpus_dir()
        files = list(iter_source_files(corpus_dir))
        if not files:
            raise RuntimeError(f"Markdown が見つかりません: {corpus_dir}")

        root = _project_root()
        points: list[dict] = []
        for path in files:
            points.extend(iter_chunks(path, root))
        if not points:
            raise RuntimeError("チャンクが生成されませんでした")

        print(
            f"[ingest] corpus={corpus_dir} files={len(files)} chunks={len(points)}",
            flush=True,
        )

        # --recreate はコレクションを作り直して全件投入するため、チェックポイント
        # によるスキップは行わない（Qdrant 実状態照合も不要）。
        completed_batches = 0 if args.recreate else _load_checkpoint(points)
        if completed_batches:
            print(
                f"[ingest] resume: checkpoint={CHECKPOINT_PATH} "
                f"completed_batches={completed_batches} (skip)",
                flush=True,
            )

        vector_size = resolved_vector_size()

        with _make_session() as session:
            if args.recreate:
                # --yes ガードは main() 冒頭で通過済み。DELETE の前に承認と
                # 実効パラメータを監査証跡へ原子的に記録する。
                _record_recreate_audit(vector_size)

                def _mark_deleted() -> None:
                    nonlocal deleted
                    deleted = True

                # DELETE → PUT。DELETE 成功（または 404）時点でフラグを立て、
                # 以降の PUT / 投入失敗を同じ経路で案内できるようにする。
                recreate_collection(
                    session,
                    vector_size=vector_size,
                    on_deleted=_mark_deleted,
                )
                deleted = True
                created = True
                print(
                    f"[ingest] collection={COLLECTION} deleted and recreated",
                    flush=True,
                )
            else:
                created = ensure_collection(session, vector_size=vector_size)
                print(
                    f"[ingest] collection={COLLECTION} "
                    + ("created" if created else "exists (skipped)"),
                    flush=True,
                )
            if created and completed_batches:
                # コレクションを作り直した場合は旧チェックポイントを無効化する。
                # 新規作成時は検証不要（対象コレクションは空のため）。
                print(
                    "[ingest] collection が新規作成されたためチェックポイントを無視します",
                    flush=True,
                )
                completed_batches = 0
            elif completed_batches:
                # resume 分岐: ensure_collection 後に Qdrant の実状態を検証する。
                # 不一致なら警告を出し、completed_batches を 0 にして全件再投入する。
                if not _verify_qdrant_state(
                    session, points, completed_batches, vector_size
                ):
                    completed_batches = 0
                    print(
                        "[ingest] Qdrant 状態検証に失敗したため全件再投入します",
                        flush=True,
                    )

            texts = [point["payload"][PAYLOAD_TEXT_KEY] for point in points]
            vectors = embed_texts(session, texts, vector_size=vector_size)
            print(f"[ingest] embedded={len(vectors)}/{len(points)}", flush=True)

            for point, vector in zip(points, vectors):
                point["vector"] = vector

            upserted_points = upsert_points(
                session,
                points,
                completed_batches=completed_batches,
                on_batch_done=lambda done: _save_checkpoint(points, done, vector_size),
            )
            print(f"[ingest] upserted={upserted_points}/{len(points)}", flush=True)

        print("[ingest] done", flush=True)
        return 0
    except _CONNECTION_EXCEPTIONS:
        # URL / ホスト / 認証情報は露出しない（固定文をマスクして stderr へ）。
        if deleted:
            _warn_collection_deleted()
        print(
            f"[ingest] {mask_secrets(_CONNECTION_ERROR_MESSAGE)}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    except RuntimeError as exc:
        # PUT の HTTP エラー（_raise_for_status）や埋め込み / 投入の失敗は
        # すべてここへ集約し、トレースバックを出さず終了コード 1 で返す。
        if deleted:
            _warn_collection_deleted()
        _warn(f"投入処理に失敗しました: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
