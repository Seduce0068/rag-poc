#!/usr/bin/env python3
"""opencodeGO の埋め込みベクトルを Qdrant へ投入する。

使い方:
  set -a; source ~/.config/opencode/dsh.env; set +a
  RAG_CLOUD_ENABLED=1 RAG_COLLECTION=rag_poc_cloud_full \
    python3 rag_poc/cloud_ingest.py

ingest.py のチャンク分割・point ID・アップサートを再利用し、埋め込み呼び出し
だけを cloud_client.embed に差し替える。チェックポイントは使用しない
（クラウド投入は都度フル投入）。認証情報の値は出力・保存しない。

起動前に ingest._check_endpoints() / config.validate_settings() / opencodeGO の
接続先検証を行い、失敗時はマスク済み stderr と終了コード 2 で終了する。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cloud_client
import config
import ingest
from config import PAYLOAD_TEXT_KEY

HERE = Path(__file__).resolve().parent

# 埋め込み次元の解決は config に集約する。正規は単数形、複数形は移行期互換。
ENV_CLOUD_EMBED_DIMENSION = config.ENV_CLOUD_EMBED_DIMENSION
ENV_CLOUD_DIMENSIONS = config.ENV_CLOUD_DIMENSIONS  # 後方互換用の別名

# 終了コード。3 は「一部バッチのみ投入済み」を表す（チェックポイント代替）。
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_VALIDATION = 2
EXIT_PARTIAL = 3

# 直近の upsert 進捗。チェックポイント file は担当外のため、モジュール状態と
# stdout/stderr / 戻り値で投入済みバッチ数を表現する（呼び出し側・テストから参照可）。
LAST_UPSERT_PROGRESS: dict[str, int] = {}


def _last_progress() -> dict[str, int]:
    """直近の投入進捗をコピーで返す公式アクセサ（R2-B）。

    部分バッチ進捗は例外の ``completed_batches`` 属性または
    ``upsert_points`` のコールバックから記録する。呼び出し側・テストは
    モジュール状態ではなく本関数を通じて参照する。
    """
    return dict(LAST_UPSERT_PROGRESS)


def _warn(message: object) -> None:
    """マスク済みのエラー／警告を stderr へ出す。"""
    print(f"[cloud-ingest] {config.mask_secrets(message)}", file=sys.stderr, flush=True)


def _configured_dimensions() -> int | None:
    """埋め込み次元の環境変数を config の共通リゾルバで読む（未設定なら None）。

    不正値は CloudUnavailable へ変換して送出する（値自体は例外文に出さない）。
    """
    try:
        return config.resolve_cloud_embed_dimension()
    except ValueError as exc:
        raise cloud_client.CloudUnavailable(str(exc)) from exc


def _expected_dim(model: str) -> int | None:
    """期待次元を解決する（環境変数リゾルバ > config のモデル次元表）。

    環境変数は config.resolve_cloud_embed_dimension()、モデル既定は
    cloud_client.expected_dim()（config.CLOUD_EMBED_DIMENSIONS を参照）に
    委譲する。ローカルな次元テーブルは持たない。
    """
    configured = _configured_dimensions()
    if configured is not None:
        return configured
    return cloud_client.expected_dim(model)


def _resolve_dimensions(model: str) -> int:
    """embed() へ渡す dimensions とコレクション次元を解決する。

    RAG_VECTOR_SIZE が明示設定され期待次元と不一致なら警告を出し、モデルの
    期待次元を採用する（モデル切り替え時に古い既定値でブロックしないため）。
    未知モデルで明示設定も無い場合は fail-closed とする。
    """
    expected = _expected_dim(model)
    configured_raw = os.environ.get("RAG_VECTOR_SIZE")
    if expected is None:
        if configured_raw is not None:
            # validate_settings() で正の int であることは検証済み。
            value = config.VECTOR_SIZE
            _warn(
                f"埋め込みモデル {model!r} は既知テーブルに無いため "
                f"RAG_VECTOR_SIZE={value} を期待次元として使用します"
            )
            return value
        raise RuntimeError(
            f"埋め込みモデル {model!r} の期待次元を解決できませんでした。"
            "RAG_CLOUD_EMBED_MODEL を text-embedding-3-small / "
            "qwen3-embedding-8b / gemini-embedding-2 のいずれかにするか、"
            "RAG_VECTOR_SIZE を明示してください。"
        )
    if configured_raw is not None and config.VECTOR_SIZE != expected:
        _warn(
            f"RAG_VECTOR_SIZE={config.VECTOR_SIZE} はモデル {model!r} の "
            f"期待次元 {expected} と不一致です。期待次元 {expected} を採用します。"
        )
    return expected


def _total_batches(total_points: int) -> int:
    size = ingest.UPSERT_BATCH_SIZE
    return (total_points + size - 1) // size


def _batches_to_points(total_points: int, completed_batches: int) -> int:
    """完了済みバッチ数に対応する点数を返す（末尾バッチは短い）。"""
    size = ingest.UPSERT_BATCH_SIZE
    done = max(0, min(completed_batches, _total_batches(total_points)))
    return sum(
        min(size, total_points - offset)
        for offset in range(0, done * size, size)
    )


def _record_progress(points: list[dict], completed_batches: int) -> dict[str, int]:
    """投入済みバッチ数／点数をモジュール状態へ記録してコピーを返す。"""
    LAST_UPSERT_PROGRESS.clear()
    LAST_UPSERT_PROGRESS.update(
        completed_batches=completed_batches,
        total_batches=_total_batches(len(points)),
        inserted_points=_batches_to_points(len(points), completed_batches),
        total_points=len(points),
    )
    return _last_progress()


def _validate_inputs(points: list[dict], texts: list[str]) -> None:
    """空チャンク・件数不一致を明示エラーにする。"""
    if not points:
        raise RuntimeError("チャンクが生成されませんでした")
    if len(texts) != len(points):
        raise RuntimeError(
            f"チャンクとテキストの件数が不一致です: "
            f"chunks={len(points)}, texts={len(texts)}"
        )
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"空チャンクを検出しました: index={index}")


def _validate_vectors(vectors: object, expected_count: int, size: int) -> None:
    """埋め込みベクトルの件数と次元を明示的に検証する。"""
    if not isinstance(vectors, list):
        raise RuntimeError(f"埋め込みベクトルの形式が不正です: {type(vectors).__name__}")
    if len(vectors) != expected_count:
        raise RuntimeError(
            f"ベクトル件数不一致: expected={expected_count}, actual={len(vectors)}"
        )
    for index, vector in enumerate(vectors):
        if not isinstance(vector, (list, tuple)) or len(vector) != size:
            actual = (
                len(vector)
                if isinstance(vector, (list, tuple))
                else type(vector).__name__
            )
            raise RuntimeError(
                f"埋め込み次元不一致: index={index}, expected={size}, actual={actual}"
            )


def _progress_line(progress: dict[str, int]) -> str:
    """部分進捗の stdout 行（EXIT_PARTIAL では checkpoint 行の代わりに出す）。"""
    return (
        "[cloud-ingest] progress: completed_batches=%d/%d inserted=%d/%d"
        % (
            progress["completed_batches"],
            progress["total_batches"],
            progress["inserted_points"],
            progress["total_points"],
        )
    )


def _warn_partial(points: list[dict], completed_batches: int, exc: object) -> None:
    """部分失敗時に「途中まで投入済み」をマスク済み stderr で案内する。

    例外が ``completed_batches`` 属性を持つ場合はその構造化値を優先する（R2-B）。
    """
    reported = getattr(exc, "completed_batches", None)
    if isinstance(reported, int) and not isinstance(reported, bool):
        completed_batches = reported
    progress = _record_progress(points, completed_batches)
    _warn(
        f"コレクション {ingest.COLLECTION} は途中まで投入済みです。"
        f"投入済みバッチ={progress['completed_batches']}/{progress['total_batches']}, "
        f"投入済み点数={progress['inserted_points']}/{progress['total_points']}"
    )
    _warn(f"upsert に失敗しました: {exc}")
    _warn(
        "checkpoint file は cloud_ingest では書きません。"
        "上記の投入済みバッチ数と終了コード 3 を利用してください"
    )


def _run_ingest() -> int:
    """実際のチャンク化〜投入。例外は呼び出し側でマスクして処理する。"""
    corpus = ingest.resolve_corpus_dir()
    files = sorted(ingest.iter_source_files(corpus))
    root = ingest._project_root()
    points: list[dict] = []
    for path in files:
        points.extend(ingest.iter_chunks(path, root))

    model = cloud_client.embed_model()
    size = _resolve_dimensions(model)

    # 既存の files/chunks 出力は維持する（空チャンク検出より前に出す）。
    print(
        "[cloud-ingest] files=%d chunks=%d model=%s dim=%d"
        % (len(files), len(points), model, size),
        flush=True,
    )

    texts = [point["payload"][PAYLOAD_TEXT_KEY] for point in points]
    _validate_inputs(points, texts)

    try:
        vectors = cloud_client.embed(texts, dimensions=size)
    except cloud_client.CloudUnavailable as exc:
        # 部分バッチ進捗は例外の構造化属性から読み、公式アクセサで参照可能にする。
        reported = getattr(exc, "completed_batches", None)
        if isinstance(reported, int) and not isinstance(reported, bool):
            progress = _record_progress(points, reported)
            print(_progress_line(progress), flush=True)
        raise
    _validate_vectors(vectors, len(texts), size)

    for point, vector in zip(points, vectors):
        point["vector"] = [float(value) for value in vector]

    session = ingest._make_session()
    created = ingest.ensure_collection(session, size)
    if created:
        print("[cloud-ingest] collection created", flush=True)

    progress = {"completed_batches": 0}

    def _on_batch_done(batch_no: int) -> None:
        progress["completed_batches"] = batch_no

    try:
        count = ingest.upsert_points(
            session, points, completed_batches=0, on_batch_done=_on_batch_done
        )
    except Exception as exc:
        # EXIT_PARTIAL では checkpoint 行を出さず、progress 行のみを出す。
        _warn_partial(points, progress["completed_batches"], exc)
        print(_progress_line(_last_progress()), flush=True)
        return EXIT_PARTIAL

    recorded = _record_progress(points, progress["completed_batches"])
    if count != len(points):
        _warn_partial(
            points,
            progress["completed_batches"],
            RuntimeError("upserted=%d/%d" % (count, len(points))),
        )
        print(_progress_line(_last_progress()), flush=True)
        return EXIT_PARTIAL
    print(
        "[cloud-ingest] checkpoint: completed_batches=%d/%d inserted=%d/%d"
        % (
            recorded["completed_batches"],
            recorded["total_batches"],
            recorded["inserted_points"],
            recorded["total_points"],
        ),
        flush=True,
    )
    print("[cloud-ingest] upserted=%d/%d" % (count, len(points)), flush=True)
    return EXIT_OK


def main() -> int:
    """エントリポイント。検証失敗は終了コード 2、部分投入は 3 を返す。"""
    # --- 起動時検証（先頭で実行。失敗時はマスク済みで終了コード 2） ---
    try:
        ingest._check_endpoints()
    except Exception as exc:
        _warn(f"接続先検証に失敗しました: {exc}")
        return EXIT_VALIDATION

    problems = config.validate_settings()
    if problems:
        for problem in problems:
            _warn(f"設定検証: {problem}")
        return EXIT_VALIDATION

    if not cloud_client.enabled():
        _warn("RAG_CLOUD_ENABLED=1 を指定してください")
        return EXIT_VALIDATION

    # opencodeGO 側の接続先も同じ基準で検証する（RAG_CLOUD_ENABLED=1 のとき）。
    cloud_problems = config.validate_endpoint(
        cloud_client.base_url(), "RAG_CLOUD_BASE_URL"
    )
    if cloud_problems:
        for problem in cloud_problems:
            _warn(f"接続先検証: {problem} (致命)")
        _warn(
            "RAG_ALLOW_REMOTE_ENDPOINTS=1 と RAG_ALLOWED_ENDPOINT_HOSTS に "
            "接続先ホストを指定してください"
        )
        return EXIT_VALIDATION

    # --- 投入処理（想定外の例外もトレースバックを出さずマスク） ---
    try:
        return _run_ingest()
    except Exception as exc:
        _warn(f"投入処理に失敗しました: {exc}")
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
