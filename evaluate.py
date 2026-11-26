#!/usr/bin/env python3
"""検索モード比較用の評価スクリプト（簡易指標 + held-out 真の recall@nDCG）。

使い方::

    python3 rag_poc/evaluate.py                    # 簡易評価 + held-out を両方実行
    python3 rag_poc/evaluate.py --heldout-only     # held-out のみ（簡易結果は既存 JSON を再利用）
    python3 rag_poc/evaluate.py --with-answer      # 検索に加えて生成結果も record に記録
    python3 rag_poc/evaluate.py --output /tmp/eval.json  # 指定パスへ書き出す（単一コレクションのみ）
    python3 rag_poc/evaluate.py --collections a,b,c      # 各コレクションを個別評価

``--collections`` と ``--output`` は併用できない。併用時は stderr に案内を出し、
終了コード 2 で拒否する（方針: 明示的拒否）。``--collections`` 指定時は各
コレクションの結果を ``eval-results-<collection>.json`` へ個別に保存する。
コレクション名にファイル名として安全でない文字が含まれる場合は ``_`` へ
置換し、さらに blake2b の短い安定 hex suffix を付与してサニタイズ後の
衝突（``a/b`` / ``a b`` / ``a_b`` 等）を回避する。英数字等のみの単純な
名前は suffix なしで従来どおりのファイル名を維持する（後方互換）。

出力先は ``rag_poc/eval-results.json`` 。``--output`` 未指定時は上書き前に
``eval-results-<UTC>.json`` へ自動退避する（退避失敗は警告のみで続行）。
退避先は notes にベース名のみを記録し、絶対パスは残さない。
``query.py`` の検索関数を直接 import し、``rag_poc/qrels.json`` から読み込んだ
日本語クエリ集合に対して dense / keyword / hybrid / hybrid-rerank の 4 モードを実行する。

- ``summary`` / ``results``: 従来の簡易指標（keyword any-match）を維持。
- ``heldout``: 実在ファイル名を正解ラベルとした query-level hit@k（recall@k）と
  nDCG@k をモード別に集計する。

外部ライブラリは evaluate 自身では使わない（標準 library のみ。numpy も使わず math で計算）。
検索関数は ``from query import search`` で直接 import し、失敗時は明示エラーで終了する。
レイテンシは ``RAG_EVAL_REPEAT`` 回数だけ計測する（既定 1 = ウォームアップなしの
1 回、実運用に近い）。2 以上を指定するとウォームアップ 1 回＋同一クエリ N 回で
avg / median / min / max / stdev と全サンプルを記録する。初回（warm-up）呼び出しの
実測値は ``cold_latency_sec`` に genuine cold として記録する（repeat=1 では samples[0]）。
再現メタ情報（embed_model / chat_model / vector_size / corpus_sha256 等）も payload に記録する。
起動時は ``config.current_settings()`` のスナップショットを設定の出所とし、
``config.validate_endpoint()`` で Ollama/Qdrant の接続先を検証する（既定はローカル限定、
``RAG_ALLOW_REMOTE_ENDPOINTS=1`` でも ``config.is_fatal_endpoint_problem()`` が真の問題は
致命として停止し、警告へ格下げしない）。必須ヘルパー
（validate_endpoint / is_fatal_endpoint_problem / current_settings / validate_settings /
qdrant_collection_path / mask_secrets）が欠けた場合は既定値へ落とさず明示エラーで終了する。
``--with-answer`` 指定時のみ ``query.generate_answer`` を呼び、``answer`` /
``sources`` / ``answer_latency_sec`` を record に追加する（失敗はマスク済みで
``answer_error`` へ分離し、検索エラー用の ``record.error`` を汚さず全体は止めない）。
既定は検索指標のみで既存スキーマを維持する。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MODES: tuple[str, ...] = ("dense", "keyword", "hybrid", "hybrid-rerank")

# held-out の評価深さ。上位 10 件を必要とするため、config.TOP_K が小さくても
# 取得だけは引き上げる（指標側は k でスライスする）。
HELD_OUT_RECALL_K: tuple[int, ...] = (1, 3, 5, 10)
HELD_OUT_NDCG_K: tuple[int, ...] = (3, 10)
HELD_OUT_MAX_K: int = max(HELD_OUT_RECALL_K + HELD_OUT_NDCG_K)

# query.search(query, mode, top_k) を直接 import して使う（動的探索は行わない）。
SEARCH_FN_NAME = "search"
REPEAT_ENV = "RAG_EVAL_REPEAT"

# samples を欠く旧形式では stdev は集計値として成立しない。数値キーは
# 後方互換（LegacyLatencyFallbackTest）のため残し、notes で明示する。
LEGACY_STDEV_NOTE = (
    "旧形式のため stdev は算出不可"
    "（samples が無く record 側 stdev の単純平均は統計的な集計値ではない）"
)

OUTPUT_PATH = os.path.join(HERE, "eval-results.json")

# 評価クエリと正解ラベルは qrels.json を唯一の情報源とする（モジュール内に
# ハードコードしない）。起動時に読み込んでスキーマを検証し、後方互換のため
# EVAL_QUERIES / HELD_OUT_QUERIES はその読み込み結果で初期化する。
QRELS_PATH = Path(__file__).resolve().parent / 'qrels.json'


class QrelsError(RuntimeError):
    """qrels.json の読み込み・スキーマ検証失敗を示す明示エラー。"""


def _load_qrels(path: Path | None = None) -> dict[str, Any]:
    """qrels ファイルを読み込み、スキーマを検証して返す。

    path 省略時は QRELS_PATH（rag_poc/qrels.json）。起動時と CLI --qrels の
    両方から呼ばれる。読み込み失敗・スキーマ不正は QrelsError を送出する。
    """
    target = path or QRELS_PATH
    try:
        with open(target, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError, TypeError) as exc:
        raise QrelsError(
            "%s の読み込みに失敗: %s: %s" % (target, type(exc).__name__, exc)
        ) from exc

    if not isinstance(data, dict):
        raise QrelsError("qrels.json の最上位は JSON オブジェクトである必要があります")
    schema_version = data.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version < 1
    ):
        raise QrelsError("qrels.json の schema_version は 1 以上の整数である必要があります")

    for key in ("simple_queries", "heldout_queries"):
        value = data.get(key)
        if not isinstance(value, list):
            raise QrelsError("qrels.json の %s は list である必要があります" % key)
        if not value:
            raise QrelsError("qrels.json の %s は空でない list である必要があります" % key)
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise QrelsError(
                    "qrels.json の %s[%d] はオブジェクトである必要があります" % (key, index)
                )
            for field in ("id", "query"):
                field_value = item.get(field)
                if not isinstance(field_value, str) or not field_value.strip():
                    raise QrelsError(
                        "qrels.json の %s[%d].%s は空でない文字列である必要があります"
                        % (key, index, field)
                    )

    # expected / relevant_sources は空でない文字列の list であること。
    for index, case in enumerate(data["simple_queries"]):
        expected = case.get("expected")
        if not isinstance(expected, list) or not expected:
            raise QrelsError(
                "qrels.json の simple_queries[%d].expected は空でない list である必要があります"
                % index
            )
        if not all(isinstance(kw, str) and kw.strip() for kw in expected):
            raise QrelsError(
                "qrels.json の simple_queries[%d].expected は空でない文字列の list である必要があります"
                % index
            )
    for index, case in enumerate(data["heldout_queries"]):
        relevant = case.get("relevant_sources")
        if not isinstance(relevant, list) or not relevant:
            raise QrelsError(
                "qrels.json の heldout_queries[%d].relevant_sources は空でない list である必要があります"
                % index
            )
        if not all(isinstance(name, str) and name.strip() for name in relevant):
            raise QrelsError(
                "qrels.json の heldout_queries[%d].relevant_sources は空でない文字列の list である必要があります"
                % index
            )
    return data


try:
    QRELS: dict[str, Any] = _load_qrels()
except QrelsError as exc:
    print("エラー: %s" % exc, file=sys.stderr)
    sys.exit(1)

# 後方互換: test_rag.py 等が直接参照する定数を qrels 由来で維持する。
EVAL_QUERIES: list[dict[str, Any]] = QRELS["simple_queries"]
HELD_OUT_QUERIES: list[dict[str, Any]] = QRELS["heldout_queries"]
QRELS_SCHEMA_VERSION: int = QRELS["schema_version"]

SOURCE_KEYS: tuple[str, ...] = (
    "source",
    "path",
    "file",
    "file_path",
    "filename",
    "unit",
    "doc_id",
    "id",
    "title",
    "name",
)
CONTENT_KEYS: tuple[str, ...] = ("text", "content", "page_content", "chunk", "snippet", "body")
NESTED_KEYS: tuple[str, ...] = ("metadata", "meta", "payload")


# ---------------------------------------------------------------------------
# 設定・再現メタ情報・検索関数の読み込み
# ---------------------------------------------------------------------------

REPO_ROOT = Path(HERE).resolve().parent


class ConfigError(RuntimeError):
    """config の import 失敗・必須ヘルパー欠落を示す明示エラー。"""


REQUIRED_CONFIG_HELPERS: tuple[str, ...] = (
    "validate_endpoint",
    "is_fatal_endpoint_problem",
    "current_settings",
    "validate_settings",
    "qdrant_collection_path",
    "mask_secrets",
)

# 既定値フォールバックを許可する任意キー（必須ヘルパーは対象外）。
OPTIONAL_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "SCHEMA_VERSION",
        "TOP_K",
        "COLLECTION",
        "CORPUS_DIR",
        "EMBED_MODEL",
        "CHAT_MODEL",
        "VECTOR_SIZE",
        "OLLAMA_URL",
        "QDRANT_URL",
        "RAG_ANSWER_TIMEOUT",
        "mask_for_provenance",
        "resolve_corpus_dir",
    }
)

_CONFIG_MODULE: Any = None


def _import_config() -> Any:
    """config を import し、必須ヘルパーを検証して返す。

    import 失敗・ヘルパー欠落は ConfigError を送出する（既定値へ黙って
    フォールバックしない fail-closed 動作）。
    """
    global _CONFIG_MODULE
    if _CONFIG_MODULE is None:
        try:
            import config  # type: ignore
        except Exception as exc:
            raise ConfigError(
                "config の import に失敗しました: %s: %s" % (type(exc).__name__, exc)
            ) from exc
        missing = [
            name
            for name in REQUIRED_CONFIG_HELPERS
            if not callable(getattr(config, name, None))
        ]
        if missing:
            raise ConfigError("config に必須ヘルパーがありません: %s" % ", ".join(missing))
        _CONFIG_MODULE = config
    return _CONFIG_MODULE


def _config_value(name: str, default: Any) -> Any:
    """config の値を返す。import 失敗・未許可キーは ConfigError（fail-closed）。

    既定値フォールバックは OPTIONAL_CONFIG_KEYS に列挙した任意キーに限る。
    必須ヘルパーには既定値を与えず、欠落時は _import_config() が明示エラーにする。
    """
    config = _import_config()
    if name in REQUIRED_CONFIG_HELPERS:
        return getattr(config, name)
    if name not in OPTIONAL_CONFIG_KEYS:
        raise ConfigError(
            "未許可の config キー %s には既定値フォールバックを適用しません" % name
        )
    return getattr(config, name, default)


def _mask(value: Any, name: str = "") -> str:
    """設定値をマスクし、notes へ安全に記載できる文字列にする。"""
    masker = _config_value("mask_for_provenance", None)
    if callable(masker):
        try:
            return str(masker(name, value))
        except Exception:
            pass
    masker = _config_value("mask_secrets", None)
    if callable(masker):
        try:
            return str(masker(value))
        except Exception:
            pass
    return "<masked>"


def _mask_error(value: Any) -> str:
    """例外メッセージを config.mask_secrets でマスクする（record.error 用）。"""
    try:
        return str(_import_config().mask_secrets(str(value)))
    except Exception:
        return "<masked>"


def _endpoint_problems(settings: dict[str, Any]) -> tuple[list[str], list[str]]:
    """settings の Ollama/Qdrant 接続先を検証する。

    既定はローカル限定。``RAG_ALLOW_REMOTE_ENDPOINTS=1`` でも
    ``config.is_fatal_endpoint_problem()`` が真を返す問題（allowlist 不一致・
    平文 HTTP・不正 scheme/port・検証例外）は致命として返す。警告へ格下げする
    のは致命ではない問題だけ（現在は loopback の情報のみ）。問題文の文字列一致
    による判定は行わない。戻り値は (致命的な問題, 警告)。
    """
    validator = _config_value("validate_endpoint", None)
    fatal_checker = _config_value("is_fatal_endpoint_problem", None)
    if not callable(validator) or not callable(fatal_checker):
        return [], []
    fatal: list[str] = []
    warnings: list[str] = []
    relaxed = os.environ.get("RAG_ALLOW_REMOTE_ENDPOINTS", "") == "1"
    for key, label in (("OLLAMA_URL", "OLLAMA_URL"), ("QDRANT_URL", "QDRANT_URL")):
        url = settings.get(key, _config_value(key, None))
        if not isinstance(url, str) or not url:
            continue
        try:
            problems = list(validator(url, label))
        except Exception as exc:
            problems = [
                "%s の URL 検証に失敗: %s: %s" % (label, type(exc).__name__, exc)
            ]
        for problem in problems:
            if fatal_checker(problem) or not relaxed:
                fatal.append(problem)
            else:
                warnings.append("%s: %s" % (label, problem))
    return fatal, warnings


def _load_top_k(settings: dict[str, Any] | None = None) -> int:
    """設定スナップショットの TOP_K があれば使う。なければ config/既定値。"""
    if settings is not None:
        try:
            return int(settings["TOP_K"])
        except (KeyError, TypeError, ValueError):
            pass
    try:
        return int(_config_value("TOP_K", 10))
    except (TypeError, ValueError):
        return 10


def _load_collection(settings: dict[str, Any] | None = None) -> str | None:
    """eval-results.json に記録するコーパス名（設定スナップショット優先）。"""
    if settings is not None:
        value = str(settings.get("COLLECTION") or "")
    else:
        value = str(_config_value("COLLECTION", "") or "")
    return value or None


def _load_repeat(settings: dict[str, Any] | None = None) -> int:
    """レイテンシ計測回数。RAG_EVAL_REPEAT（既定 1、最小 1）。

    既定 1 はウォームアップなしの 1 回計測（実運用に近い値）。
    複数回統計を取る場合は明示的に 2 以上を指定する。
    """
    raw: Any = None
    if settings is not None:
        raw = settings.get(REPEAT_ENV)
    if raw is None:
        raw = os.environ.get(REPEAT_ENV, "1")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 1
    return max(1, value)


def _validate_settings() -> list[str]:
    """config.validate_settings() を起動時に呼び、問題を文字列で返す。

    ヘルパー欠落・実行例外は ConfigError として送出する（空列表へ落とさない）。
    """
    config = _import_config()
    try:
        return list(config.validate_settings())
    except Exception as exc:
        raise ConfigError(
            "config.validate_settings() の実行に失敗: %s: %s" % (type(exc).__name__, exc)
        ) from exc


def _corpus_dir(settings: dict[str, Any] | None = None) -> Path:
    """コーパスディレクトリを解決する（設定スナップショットを優先）。

    settings が無い場合は config.resolve_corpus_dir() に委譲し、それも無い場合のみ
    リポジトリルート基準で解決する（ingest と同じ手順）。
    """
    raw: Any = None
    if settings is not None:
        candidate = settings.get("CORPUS_DIR")
        if isinstance(candidate, str) and candidate.strip():
            raw = candidate
    if raw is None:
        resolver = _config_value("resolve_corpus_dir", None)
        if callable(resolver):
            try:
                return Path(resolver())
            except Exception:
                pass
        # フォールバック: resolver が無い場合のみリポジトリルート基準で解決。
        raw = str(
            _config_value("CORPUS_DIR", "passkey-gateway/docs") or "passkey-gateway/docs"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _corpus_fingerprint(settings: dict[str, Any] | None = None) -> tuple[int, str]:
    """Corpus 配下の .md をパス順にソートし、本文を連結した SHA-256 を返す。"""
    root = _corpus_dir(settings)
    files = sorted(
        (path for path in root.rglob("*.md") if path.is_file()),
        key=lambda path: str(path),
    )
    digest = hashlib.sha256()
    counted = 0
    for path in files:
        try:
            with open(path, "rb") as handle:
                digest.update(handle.read())
        except OSError:
            continue
        counted += 1
    return counted, digest.hexdigest()


def _eval_metadata(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """eval-results.json に記録する再現メタ情報（設定スナップショット優先）。

    current_settings() は embed_model / chat_model を含まないため、main() で
    環境変数から補完した settings を参照する（設定の出所を 1 箇所に集約）。
    """
    file_count, corpus_sha256 = _corpus_fingerprint(settings)
    if settings is not None:
        embed_model = str(settings.get("EMBED_MODEL") or "")
        chat_model = str(settings.get("CHAT_MODEL") or "")
        raw_vector = settings.get("VECTOR_SIZE", 0)
    else:
        embed_model = str(_config_value("EMBED_MODEL", "") or "")
        chat_model = str(_config_value("CHAT_MODEL", "") or "")
        raw_vector = _config_value("VECTOR_SIZE", 0)
    try:
        vector_size = int(raw_vector)
    except (TypeError, ValueError):
        vector_size = 0
    # クラウド埋め込みが有効な場合は実際のモデル ID を記録する（誤解防止）。
    try:
        import cloud_client

        if cloud_client.enabled():
            embed_model = "opencode-go/" + cloud_client.embed_model()
    except ImportError:
        pass
    return {
        "embed_model": embed_model,
        "chat_model": chat_model,
        "vector_size": vector_size,
        "corpus_file_count": file_count,
        "corpus_sha256": corpus_sha256,
    }


def _load_search_function() -> tuple[Callable[..., Any], Any]:
    """``from query import search`` で直接 import する。失敗時は例外で終了。"""
    try:
        import query  # type: ignore

        from query import search
    except Exception as exc:
        raise RuntimeError(
            "query.search の import に失敗しました（既定の検索関数へ黙って切り替えません）: %s: %s"
            % (type(exc).__name__, exc)
        ) from exc
    return search, query


def _empty_latency_stats() -> dict[str, Any]:
    """レイテンシ統計の空スロット（record の構造を常に一定にする）。"""
    return {
        "count": 0,
        "avg": None,
        "median": None,
        "min": None,
        "max": None,
        "stdev": None,
    }


def _call_search(
    func: Callable[..., Any],
    question: str,
    mode: str,
    top_k: int,
    repeat: int | None = None,
    errors_getter: Callable[..., Any] | None = None,
) -> tuple[Any, str, dict[str, Any], list[dict[str, Any]]]:
    """search(query, mode, top_k) を直接呼び、レイテンシを N 回計測する。

    repeat > 1 のときは最初の呼び出し（warm-up）のレイテンシを genuine cold として
    cold_latency_sec に記録し、その後同一クエリを repeat 回呼んで samples に格納する。
    repeat == 1 ならウォームアップなしの 1 回のみで、samples[0] を cold とする。
    各ラウンドの search が記録したエラーは全件集約して返す（最終回だけを拾うと
    断続的な障害が隠れるため）。戻り値は (最後の結果, 説明, 統計, エラー一覧)。
    """
    rounds = _load_repeat() if repeat is None else max(1, int(repeat))
    all_errors: list[dict[str, Any]] = []

    def collect_errors() -> None:
        if errors_getter is None:
            return
        try:
            errors = errors_getter()
        except Exception:  # noqa: BLE001 - エラー取得自体は計測を止めない
            return
        if isinstance(errors, list):
            all_errors.extend(item for item in errors if isinstance(item, dict))

    # genuine cold: 最初の呼び出し（warm-up）を計測して cold_latency_sec に格納する。
    # 旧実装は warm-up を計測せず「warm-up 後の samples[0]」を cold と呼んでいたため
    # 実態は warm だった。repeat == 1 では warm-up が無いため samples[0] を cold とする。
    cold_latency: float | None = None
    if rounds > 1:
        started = time.perf_counter()
        func(question, mode, top_k)  # warm-up（結果は捨てるが初回レイテンシは残す）
        cold_latency = time.perf_counter() - started
        collect_errors()
    samples: list[float] = []
    raw: Any = None
    for _ in range(rounds):
        started = time.perf_counter()
        raw = func(question, mode, top_k)
        samples.append(time.perf_counter() - started)
        collect_errors()
    stats = {
        "count": rounds,
        "avg": round(sum(samples) / rounds, 6),
        "median": round(statistics.median(samples), 6),
        "min": round(min(samples), 6),
        "max": round(max(samples), 6),
        "stdev": round(statistics.pstdev(samples), 6),
        "samples": [round(value, 6) for value in samples],
        "cold_latency_sec": (
            round(cold_latency, 6)
            if cold_latency is not None
            else (round(samples[0], 6) if samples else None)
        ),
    }
    description = "search(query=..., mode=%r, top_k=%d)" % (mode, top_k)
    return raw, description, stats, all_errors


def _append_error_text(record: dict[str, Any], text: Any) -> None:
    """整形済みエラーを重複排除して record.error へ追記する。"""
    message = str(text).strip()
    if not message:
        return
    existing = str(record.get("error") or "")
    if message in existing:
        return
    record["error"] = "%s; %s" % (existing, message) if existing else message


def _merge_search_errors(
    record: dict[str, Any],
    getter: Callable[..., Any] | None,
    errors: list[dict[str, Any]] | None = None,
) -> None:
    """検索エラーを record.error に反映する（唯一の情報源は search_errors）。

    query.py は [{"stage", "mode", "type", "message"}, ...] を返すため、
    "stage/mode: Type: message" 形式に整形してから連結する。
    同一 (stage, type, message) は 1 度だけ記録する（mode はキーに含めない）。
    """

    def as_message(item: Any) -> str:
        if isinstance(item, dict):
            stage = str(item.get("stage", "") or "")
            mode = str(item.get("mode", "") or "")
            error_type = str(item.get("type", "") or "")
            message = str(item.get("message", "") or "")
            label = "/".join(part for part in (stage, mode) if part)
            head = "%s: " % label if label else ""
            return "%s%s: %s" % (head, error_type or "Error", message)
        return str(item).strip()

    def as_key(item: Any) -> tuple[str, str, str]:
        if isinstance(item, dict):
            return (
                str(item.get("stage", "") or ""),
                str(item.get("type", "") or ""),
                str(item.get("message", "") or ""),
            )
        return ("", "", str(item).strip())

    if errors is not None:
        value: Any = errors
    elif getter is not None:
        try:
            value = getter()
        except Exception:
            return
    else:
        return
    if isinstance(value, str):
        items: list[Any] = [value] if value.strip() else []
    elif isinstance(value, (list, tuple)):
        items = list(value)
    elif value:
        items = [value]
    else:
        items = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = as_key(item)
        if key in seen:
            continue
        seen.add(key)
        _append_error_text(record, as_message(item))


def _aggregate_latency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """record の latency_stats.samples を全件集約して summary を算出する。

    samples が無い旧形式では record 側の統計の平均にフォールバックする
    （意味は「record 統計の要約」であり全サンプルの統計ではない）。
    旧形式の stdev は record 値の単純平均にすぎず集計値として成立しないため、
    ``LEGACY_STDEV_NOTE`` を notes に記録する。数値は後方互換のため保持するが、
    None / 0 へ置き換えて既存の互換テストを壊すことはしない。
    """
    samples: list[float] = []
    fallback: list[dict[str, Any]] = []
    for row in rows:
        stats = row.get("latency_stats")
        if not isinstance(stats, dict) or not stats.get("count"):
            continue
        values = stats.get("samples")
        if isinstance(values, list):
            samples.extend(float(value) for value in values)
        else:
            fallback.append(stats)
    if samples:
        return {
            "count": len(samples),
            "avg": round(sum(samples) / len(samples), 6),
            "median": round(statistics.median(samples), 6),
            "min": round(min(samples), 6),
            "max": round(max(samples), 6),
            "stdev": round(statistics.pstdev(samples), 6),
        }
    if not fallback:
        return _empty_latency_stats()
    avgs = [float(item["avg"]) for item in fallback]
    return {
        "count": sum(int(item["count"]) for item in fallback),
        "avg": round(sum(avgs) / len(avgs), 6),
        "median": round(statistics.median([float(item["median"]) for item in fallback]), 6),
        "min": round(min(float(item["min"]) for item in fallback), 6),
        "max": round(max(float(item["max"]) for item in fallback), 6),
        "stdev": round(sum(float(item["stdev"]) for item in fallback) / len(fallback), 6),
        "notes": [LEGACY_STDEV_NOTE],
    }


def _as_items(result: Any) -> list[Any]:
    """search() の戻り値を結果リストに正規化する。"""
    if result is None:
        return []
    if isinstance(result, dict):
        for key in ("results", "hits", "documents", "items", "chunks", "passages"):
            value = result.get(key)
            if isinstance(value, (list, tuple)):
                return list(value)
        return [result]
    if isinstance(result, (list, tuple)):
        return list(result)
    return [result]


def _value_from(item: Any, keys: tuple[str, ...]) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, (list, tuple)):
        for part in item:
            value = _value_from(part, keys)
            if value:
                return value
        return ""
    if isinstance(item, dict):
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in NESTED_KEYS:
            nested = item.get(key)
            if isinstance(nested, dict):
                for nested_key in keys:
                    value = nested.get(nested_key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        return ""
    return str(item).strip()


def _source_of(item: Any) -> str:
    return _value_from(item, SOURCE_KEYS)


def _content_of(item: Any) -> str:
    return _value_from(item, CONTENT_KEYS)


def _matched(expected: list[str], text: str) -> list[str]:
    lowered = (text or "").lower()
    return [kw for kw in expected if kw.lower() in lowered]


def _answer_timeout_seconds() -> float:
    """config.RAG_ANSWER_TIMEOUT（既定 60 秒）を返す。不正値は既定へ落とす。"""
    try:
        value = float(_config_value("RAG_ANSWER_TIMEOUT", 60.0))
    except (TypeError, ValueError):
        return 60.0
    if not math.isfinite(value) or value <= 0:
        return 60.0
    return value


def _with_answer_timeout(
    answer_fn: Callable[..., Any], query_module: Any
) -> Callable[..., Any]:
    """generate_answer 呼び出し中だけ Ollama タイムアウトを締めるラッパー。

    ``query.generate_answer`` は timeout 引数を持たないため、シグナル／スレッドは
    使わず、HTTP 側が参照する ``query.OLLAMA_TIMEOUT`` を呼び出し中だけ
    ``config.RAG_ANSWER_TIMEOUT`` へ差し替える（finally で必ず復元）。
    現在の値より長くなる場合は上書きしない（安全優先・タイムアウトを伸ばさない）。
    差し替えられない場合は警告のみとし、恒久対策は recommendation に記載する。
    """
    timeout = _answer_timeout_seconds()
    original = getattr(query_module, "OLLAMA_TIMEOUT", None)
    can_tighten = (
        isinstance(original, (int, float))
        and not isinstance(original, bool)
        and math.isfinite(float(original))
        and timeout < float(original)
    )
    if not can_tighten:
        print(
            "警告: 回答生成のタイムアウトを適用できませんでした"
            "（config.RAG_ANSWER_TIMEOUT=%ss、現行 OLLAMA_TIMEOUT=%r）。"
            "query.generate_answer 側に timeout 引数を追加してください。"
            % (timeout, original),
            file=sys.stderr,
            flush=True,
        )

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not can_tighten:
            return answer_fn(*args, **kwargs)
        query_module.OLLAMA_TIMEOUT = timeout
        try:
            return answer_fn(*args, **kwargs)
        finally:
            query_module.OLLAMA_TIMEOUT = original

    wrapped.__name__ = getattr(answer_fn, "__name__", "generate_answer")
    return wrapped


def _record_generated_answer(
    record: dict[str, Any],
    question: str,
    items: list[Any],
    answer_fn: Callable[..., Any] | None,
    *,
    sources_key: str = "sources",
) -> None:
    """query.generate_answer を呼び answer / sources / レイテンシを追加する。

    失敗はマスク済みで record.error に記録し、例外は外へ出さない（全体を止めない）。
    held-out は既存の ``sources`` が検索指標用のため ``answer_sources`` を使う。
    """
    answer_sources = list(
        dict.fromkeys(src for src in (_source_of(item) for item in items) if src)
    )
    record[sources_key] = answer_sources
    record["answer"] = None
    record["answer_latency_sec"] = None
    if not callable(answer_fn):
        record["answer_error"] = "generate_answer: 呼び出し可能な関数がありません"
        return
    started = time.perf_counter()
    try:
        answer = answer_fn(question, items)
        record["answer"] = answer if isinstance(answer, str) else str(answer)
    except Exception as exc:  # 1 ケースの生成失敗で全体を止めない
        # 検索失敗フラグ (record.error) を汚さず集計の分母を壊さないよう
        # 生成失敗は専用フィールドへ分離する。
        record["answer_error"] = "generate_answer: %s: %s" % (
            type(exc).__name__,
            _mask_error(exc),
        )
    finally:
        record["answer_latency_sec"] = round(time.perf_counter() - started, 6)


def _evaluate_one(
    func: Callable[..., Any],
    question: str,
    expected: list[str],
    mode: str,
    top_k: int,
    repeat: int | None = None,
    errors_getter: Callable[..., Any] | None = None,
    with_answer: bool = False,
    answer_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "query": question,
        "expected": expected,
        "mode": mode,
        "hits": 0,
        "recall": False,
        "matched_keywords": [],
        "content_recall": False,
        "matched_content_keywords": [],
        "unit_uniqueness": {
            "top3_sources": [],
            "unique_source_count": 0,
            "unique_ratio": 0.0,
        },
        "latency_sec": None,
        "latency_stats": _empty_latency_stats(),
        "call": None,
        "error": None,
    }
    started = time.perf_counter()
    try:
        raw, call_description, latency, search_errors = _call_search(
            func, question, mode, top_k, repeat, errors_getter
        )
        record["call"] = call_description
        record["latency_stats"] = latency
        record["latency_sec"] = latency["avg"]  # 後方互換: 平均値を維持
        items = _as_items(raw)
    except Exception as exc:  # 1 ケースの失敗で全体を止めない
        record["error"] = "%s: %s" % (type(exc).__name__, _mask_error(exc))
        record["latency_sec"] = round(time.perf_counter() - started, 6)
        _merge_search_errors(record, errors_getter)
        return record

    # search_errors を唯一の情報源にする（末尾の getter 再マージは削除）。
    _merge_search_errors(record, errors_getter, search_errors)

    record["hits"] = len(items)

    sources = [src for src in (_source_of(item) for item in items[:3]) if src]
    unique_sources = list(dict.fromkeys(sources))
    denominator = len(sources) if sources else 0
    record["unit_uniqueness"] = {
        "top3_sources": sources,
        "unique_source_count": len(unique_sources),
        "unique_ratio": round(len(unique_sources) / denominator, 4) if denominator else 0.0,
    }

    source_text = "\n".join(sources)
    content_text = "\n".join(_content_of(item) for item in items)
    record["matched_keywords"] = _matched(expected, source_text)
    record["recall"] = bool(record["matched_keywords"])
    record["matched_content_keywords"] = _matched(expected, content_text)
    record["content_recall"] = bool(record["matched_content_keywords"])
    if with_answer:
        _record_generated_answer(record, question, items, answer_fn, sources_key="sources")
    return record


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in MODES:
        all_rows = [r for r in records if r["mode"] == mode]
        rows = [r for r in all_rows if r["error"] is None]
        count = len(all_rows)  # 失敗行も分母に含め、生存者バイアスを避ける
        summary[mode] = {
            "cases": count,
            "successes": len(rows),
            "source_keyword_hit_at3": (
                round(sum(bool(r["recall"]) for r in all_rows) / count, 4) if count else None
            ),
            "content_keyword_hit_all_hits": (
                round(sum(bool(r["content_recall"]) for r in all_rows) / count, 4)
                if count
                else None
            ),
            "avg_distinct_source_ratio_at3": (
                round(sum(r["unit_uniqueness"]["unique_ratio"] for r in rows) / len(rows), 4)
                if rows
                else None
            ),
            "avg_latency_sec": (
                round(sum(float(r["latency_sec"]) for r in rows) / len(rows), 6) if rows else None
            ),
            "latency_stats": _aggregate_latency(rows),
            "avg_hits": round(sum(r["hits"] for r in rows) / len(rows), 2) if rows else None,
            "errors": sum(1 for r in all_rows if r["error"] is not None),
        }
    return summary


# ---------------------------------------------------------------------------
# held-out 評価（正解ラベル付き recall@k / nDCG@k）
# ---------------------------------------------------------------------------


def _basename(value: str) -> str:
    """os.path.basename の Windows 区切り・前後空白・大小文字を吸収した版。"""
    return os.path.basename(value.replace("\\", "/").strip()).lower()


def _source_matches(source: str, relevant_sources: list[str]) -> bool:
    """検索結果の source が正解ファイル名のいずれかかを判定する。

    ingest.py は payload.source をリポジトリルートからの相対パス
    (例 ``passkey-gateway/docs/rag-methods-selection-2026-11-26.md``) で保存する。
    ラベルはファイル名で持つため、完全一致・相対パス suffix・basename の3通りで照合する。
    """
    if not source:
        return False
    normalized = source.replace("\\", "/").strip().lower()
    base = _basename(normalized)
    for expected in relevant_sources:
        target = str(expected).replace("\\", "/").strip().lower()
        if not target:
            continue
        if normalized == target or normalized.endswith("/" + target.lstrip("/")):
            return True
        if base == _basename(target):
            return True
    return False


def _evaluate_heldout_one(
    func: Callable[..., Any],
    case: dict[str, Any],
    mode: str,
    retrieval_k: int,
    repeat: int | None = None,
    errors_getter: Callable[..., Any] | None = None,
    with_answer: bool = False,
    answer_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """1 クエリ × 1 モードの recall@k / nDCG@k を計算する。"""
    relevant = list(case["relevant_sources"])
    record: dict[str, Any] = {
        "id": case["id"],
        "query": case["query"],
        "relevant_sources": relevant,
        "mode": mode,
        "retrieval_k": retrieval_k,
        "hits": 0,
        "sources": [],
        "matched_sources": [],
        "recall_at": {str(k): False for k in HELD_OUT_RECALL_K},
        "ndcg_at": {str(k): 0.0 for k in HELD_OUT_NDCG_K},
        "latency_sec": None,
        "latency_stats": _empty_latency_stats(),
        "call": None,
        "error": None,
    }
    started = time.perf_counter()
    try:
        raw, call_description, latency, search_errors = _call_search(
            func, case["query"], mode, retrieval_k, repeat, errors_getter
        )
        record["call"] = call_description
        record["latency_stats"] = latency
        record["latency_sec"] = latency["avg"]  # 後方互換: 平均値を維持
        items = _as_items(raw)
    except Exception as exc:  # 1 ケースの失敗で全体を止めない
        record["error"] = "%s: %s" % (type(exc).__name__, _mask_error(exc))
        record["latency_sec"] = round(time.perf_counter() - started, 6)
        _merge_search_errors(record, errors_getter)
        return record

    # search_errors を唯一の情報源にする（末尾の getter 再マージは削除）。
    _merge_search_errors(record, errors_getter, search_errors)
    record["hits"] = len(items)
    sources = [src for src in (_source_of(item) for item in items[:retrieval_k]) if src]
    record["sources"] = sources
    record["matched_sources"] = sorted({s for s in sources if _source_matches(s, relevant)})

    # recall@k: 上位 k 件に正解が 1 件でもあれば 1（クエリ平均の hit@k）。
    for k in HELD_OUT_RECALL_K:
        record["recall_at"][str(k)] = any(_source_matches(s, relevant) for s in sources[:k])

    # nDCG@k: gain=1 / discount=log2(rank+1)（rank は 1 起点）。
    # 正解はファイル単位なので、同じファイルの複数チャンクは最上位の1回だけ数え、
    # 1 ファイルにつき最大 1 gain とする（nDCG が 1 を超えないようにする）。
    for k in HELD_OUT_NDCG_K:
        dcg = 0.0
        credited: set[str] = set()
        for rank, source in enumerate(sources[:k], start=1):
            for expected in relevant:
                key = str(expected).lower()
                if key in credited:
                    continue
                if _source_matches(source, [expected]):
                    credited.add(key)
                    dcg += 1.0 / math.log2(rank + 1)
                    break
        ideal_k = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_k + 1))
        record["ndcg_at"][str(k)] = round(dcg / idcg, 6) if idcg > 0 else 0.0
    if with_answer:
        _record_generated_answer(
            record, case["query"], items, answer_fn, sources_key="answer_sources"
        )
    return record


def _summarize_heldout(records: list[dict[str, Any]]) -> dict[str, Any]:
    """モード別・クエリ平均。失敗ケースも 0 として分母に残す。"""
    summary: dict[str, Any] = {}
    for mode in MODES:
        rows = [r for r in records if r["mode"] == mode]
        errors = sum(1 for r in rows if r["error"] is not None)
        entry: dict[str, Any] = {
            "queries": len(rows),
            "successes": len(rows) - errors,
            "errors": errors,
            "recall_at": {},
            "ndcg_at": {},
            "latency_stats": _aggregate_latency(rows),
        }
        for k in HELD_OUT_RECALL_K:
            hits = sum(1 for r in rows if r["recall_at"].get(str(k)) is True)
            entry["recall_at"][str(k)] = round(hits / len(rows), 4) if rows else None
        for k in HELD_OUT_NDCG_K:
            total = sum(float(r["ndcg_at"].get(str(k), 0.0)) for r in rows)
            entry["ndcg_at"][str(k)] = round(total / len(rows), 4) if rows else None
        summary[mode] = entry
    return summary


# ---------------------------------------------------------------------------
# CLI / オーケストレーション
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG PoC の簡易評価 + held-out 評価")
    parser.add_argument(
        "--heldout-only",
        action="store_true",
        help="held-out のみ実行する（簡易結果は既存 eval-results.json を再利用）",
    )
    parser.add_argument(
        "--qrels",
        type=str,
        default=None,
        help="qrels ファイルのパス（既定: rag_poc/qrels.json）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "書き出し先パス（既定: rag_poc/eval-results.json）。"
            "--collections と併用不可（併用時は終了コード 2 で拒否）"
        ),
    )
    parser.add_argument(
        "--with-answer",
        action="store_true",
        help="query.generate_answer を呼び answer / sources / answer_latency_sec を記録する",
    )
    parser.add_argument(
        "--collections",
        type=str,
        default=None,
        metavar="A,B,C",
        help=(
            "評価対象コレクションをカンマ区切りで指定"
            "（各コレクションを個別評価し eval-results-<collection>.json へ保存）。"
            "--output と併用不可（併用時は終了コード 2 で拒否）"
        ),
    )
    return parser.parse_args(argv)


def _run_simple_eval(
    func: Callable[..., Any],
    top_k: int,
    repeat: int | None = None,
    errors_getter: Callable[..., Any] | None = None,
    with_answer: bool = False,
    answer_fn: Callable[..., Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    for case in QRELS["simple_queries"]:
        for mode in MODES:
            records.append(
                _evaluate_one(
                    func,
                    case["query"],
                    case["expected"],
                    mode,
                    top_k,
                    repeat,
                    errors_getter,
                    with_answer=with_answer,
                    answer_fn=answer_fn,
                )
            )
    return records, []


def _run_heldout_eval(
    func: Callable[..., Any],
    retrieval_k: int,
    repeat: int | None = None,
    errors_getter: Callable[..., Any] | None = None,
    with_answer: bool = False,
    answer_fn: Callable[..., Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    for case in QRELS["heldout_queries"]:
        for mode in MODES:
            records.append(
                _evaluate_heldout_one(
                    func,
                    case,
                    mode,
                    retrieval_k,
                    repeat,
                    errors_getter,
                    with_answer=with_answer,
                    answer_fn=answer_fn,
                )
            )
    return records, []


def _reuse_simple_results(
    settings: dict[str, Any] | None = None,
    qrels_label: str | None = None,
    qrels_schema_version: int | None = None,
    path: str | None = None,
    with_answer: bool = False,
) -> tuple[list[dict[str, Any]] | None, list[str]]:
    """--heldout-only で既存 JSON の簡易結果を再利用する（無ければ None）。

    同一性は collection / corpus_sha256 / corpus_file_count / embed_model /
    chat_model / vector_size / top_k / schema_version / qrels_label /
    qrels_schema_version / with_answer を照合し、1 つでも不一致または欠落なら
    再計測する（欠落を「検証不能」として許容すると旧形式と新形式が混在するため）。
    with_answer を照合キーに含めることで、answer なしの再利用レコードを
    ``--with-answer`` 指定時に流用する矛盾を防ぐ。旧形式に with_answer 記録が
    無い場合は False（回答生成なし）とみなす（後方互換）。

    qrels_label / qrels_schema_version は main() が算出した値を渡す。引数なしの
    直接呼び出し（後方互換）では既定の qrels.json / モジュールの
    QRELS_SCHEMA_VERSION を使う。schema_version は config.SCHEMA_VERSION を
    _config_value 経由で参照し、リテラルは使わない。path 指定時はそのファイルを
    再利用元にする（--output と組み合わせた場合）。
    """
    try:
        with open(path or OUTPUT_PATH, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        records = previous.get("results")
        if isinstance(records, list) and isinstance(previous.get("summary"), dict):
            current_collection = _load_collection(settings)
            old_collection = previous.get("collection")
            current_metadata = _eval_metadata(settings)
            current_values: dict[str, Any] = {
                "schema_version": _config_value("SCHEMA_VERSION", 2),
                "corpus_sha256": current_metadata.get("corpus_sha256"),
                "corpus_file_count": current_metadata.get("corpus_file_count"),
                "embed_model": current_metadata.get("embed_model"),
                "chat_model": current_metadata.get("chat_model"),
                "vector_size": current_metadata.get("vector_size"),
                "top_k": _load_top_k(settings),
                "qrels_label": qrels_label if qrels_label is not None else "qrels.json",
                "qrels_schema_version": (
                    qrels_schema_version
                    if qrels_schema_version is not None
                    else QRELS_SCHEMA_VERSION
                ),
                "with_answer": bool(with_answer),
            }
            # 整数として照合するキー（bool は None として扱い不一致にする）。
            int_keys = (
                "schema_version", "corpus_file_count", "vector_size", "top_k",
                "qrels_schema_version",
            )

            def as_comparable_int(value: Any) -> int | None:
                if value is None or isinstance(value, bool):
                    return None
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None

            mismatches: list[str] = []
            notes: list[str] = []
            if old_collection is None:
                mismatches.append("既存結果に collection 記録が無い")
            elif old_collection != current_collection:
                mismatches.append(
                    "collection=%s（現行 %s）" % (old_collection, current_collection)
                )
            for key in (
                "schema_version", "corpus_sha256", "corpus_file_count",
                "embed_model", "chat_model", "vector_size", "top_k",
                "qrels_label", "qrels_schema_version", "with_answer",
            ):
                if key == "with_answer":
                    # 旧形式には with_answer 記録が無いが、回答生成は未指定（False）と同義。
                    # 欠落を False と正規化したうえで現行フラグと照合する。
                    old_value = bool(previous.get("with_answer", False))
                else:
                    old_value = previous.get(key)
                current_value = current_values[key]
                if key in int_keys:
                    old_value = as_comparable_int(old_value)
                    current_value = as_comparable_int(current_value)
                if old_value is None:
                    mismatches.append("既存結果に %s 記録が無い" % key)
                elif old_value != current_value:
                    mismatches.append(
                        "%s=%s（現行 %s）" % (key, old_value, current_value)
                    )
            if mismatches:
                return None, [
                    "--heldout-only: 既存結果と現行設定が異なるため再利用せず再計測（%s）"
                    % "；".join(mismatches)
                ]
            notes.append(
                "--heldout-only: 簡易指標は既存 eval-results.json の結果を再利用（再計測なし）"
            )
            return records, notes
    except (OSError, ValueError, TypeError):
        pass
    return None, []


def _backup_output(path: str) -> str | None:
    """既存結果を UTC タイムスタンプ付き eval-results-<UTC>.json へ退避する。

    退避は上書き前に行う。コピー失敗（ファイル不在・権限等）は警告のみで
    続行し、eval 本体を止めない。``--output`` 指定時は呼び出さない。
    戻り値は退避先のベース名のみ（notes に絶対パスを残さないため）。
    """
    try:
        source = Path(path)
        if not source.is_file():
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = source.with_name("eval-results-%s.json" % stamp)
        shutil.copy2(source, target)
        return target.name
    except Exception as exc:  # noqa: BLE001 - 退避失敗で評価を止めない
        print(
            "警告: 既存結果の退避に失敗しました（続行します）: %s: %s"
            % (type(exc).__name__, exc),
            file=sys.stderr,
        )
        return None


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


# ---------------------------------------------------------------------------
# 複数コレクション対応（--collections）
# ---------------------------------------------------------------------------


def _parse_collections(raw: str | None) -> list[str]:
    """--collections のカンマ区切り文字列をコレクション名の list にする。

    None / 空文字 / カンマだけの場合は空 list（＝単一コレクション動作）を返す。
    前後空白を除去し、空要素と重複を最初の出現順で排除する。
    """
    if raw is None:
        return []
    names = [part.strip() for part in raw.split(",")]
    names = [name for name in names if name]
    return list(dict.fromkeys(names))


def _collection_output_path(name: str) -> str:
    """コレクション名に対応する eval-results-<collection>.json のパスを返す。

    ファイル名に使えない文字は ``_`` へ置換する。サニタイズ後の名前が元の名前と
    異なる場合（``a/b`` / ``a b`` / ``a_b!`` 等）は、元の名前から算出した
    blake2b の短い安定 hex suffix を付与して衝突を回避する。サニタイズが
    不要な単純な名前（英数字・``.`` / ``_`` / ``-`` のみ）は suffix なしで
    従来どおりのファイル名を維持する（後方互換）。
    """
    original = name.strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", original)
    safe = safe.strip("._") or "collection"
    if safe != original:
        # 8bit（2 hex 文字）は短すぎて衝突確率が無視できないため 16bit（4 hex 文字）を使う。
        # 元の名前（前後空白を除く）をハッシュするので同名は常に同じ suffix になる。
        suffix = hashlib.blake2b(original.encode("utf-8"), digest_size=2).hexdigest()
        safe = "%s_%s" % (safe, suffix)
    return os.path.join(HERE, "eval-results-%s.json" % safe)


def _collection_search(
    func: Callable[..., Any], query_module: Any, name: str
) -> Callable[..., Any]:
    """検索呼び出しの間だけ query.COLLECTION を name に切り替えるラッパー。

    query.py は COLLECTION を import 時にモジュール変数として保持するため、
    コレクションごとの評価ではこの値を差し替える必要がある。呼び出し後は
    必ず元の値へ戻し、他コレクションの評価へ影響しないようにする。
    """

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        # search(question, mode, top_k) と generate_answer(question, hits) の
        # 両方へ渡せるよう、元関数のシグネチャをそのまま尊重する。
        previous = getattr(query_module, "COLLECTION", None)
        query_module.COLLECTION = name
        try:
            return func(*args, **kwargs)
        finally:
            query_module.COLLECTION = previous

    wrapped.__name__ = getattr(func, "__name__", SEARCH_FN_NAME)
    return wrapped


def _print_collection_summary(rows: list[dict[str, Any]]) -> None:
    """複数コレクションの要約表を stderr へ出力する。"""
    header = ("collection", "mode", "recall@10", "nDCG@10", "latency_avg_sec")
    lines = ["%-20s %-14s %10s %10s %16s" % header]
    for row in rows:
        lines.append(
            "%-20s %-14s %10s %10s %16s"
            % (
                row["collection"],
                row["mode"],
                _fmt(row.get("recall_at_10")),
                _fmt(row.get("ndcg_at_10")),
                _fmt(row.get("latency_avg_sec")),
            )
        )
    print("\n".join(lines), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    collections = _parse_collections(args.collections)

    # --collections と --output は明示的に拒否する（方針は docstring / CLI help に記載）。
    # 各コレクションの保存先は eval-results-<collection>.json に固定されるため、
    # --output を同時に受け付けると書き出し先が曖昧になる。
    if collections and args.output is not None:
        print(
            "エラー: --collections と --output は併用できません。"
            "--collections を指定する場合は --output を外してください"
            "（各コレクションは eval-results-<collection>.json へ保存されます）。",
            file=sys.stderr,
        )
        return 2

    if not collections:
        return _evaluate_once(
            args,
            args.output or OUTPUT_PATH,
            backup=args.output is None,
        )

    # 複数コレクション: 各コレクションを個別に評価し、最後に要約表を出す。
    summary_rows: list[dict[str, Any]] = []
    for name in collections:
        output_path = _collection_output_path(name)
        code = _evaluate_once(
            args,
            output_path,
            collection_override=name,
            backup=False,
        )
        if code != 0:
            return code
        try:
            payload = json.loads(Path(output_path).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            print(
                "要約用の結果読み込みに失敗しました: %s: %s"
                % (output_path, type(exc).__name__),
                file=sys.stderr,
            )
            return 1
        heldout_summary = (payload.get("heldout") or {}).get("summary") or {}
        for mode in MODES:
            stats = heldout_summary.get(mode) or {}
            latency = (stats.get("latency_stats") or {}).get("avg")
            recall_at = stats.get("recall_at") or {}
            ndcg_at = stats.get("ndcg_at") or {}
            summary_rows.append(
                {
                    "collection": name,
                    "mode": mode,
                    "recall_at_10": recall_at.get("10"),
                    "ndcg_at_10": ndcg_at.get("10"),
                    "latency_avg_sec": latency,
                }
            )
    _print_collection_summary(summary_rows)
    return 0


def _evaluate_once(
    args: argparse.Namespace,
    output_path: str,
    collection_override: str | None = None,
    *,
    backup: bool = False,
) -> int:
    """1 コレクション分の評価を実行し output_path へ書き出す。

    collection_override が指定された場合は検索先コレクションと記録値を差し替え、
    ``--heldout-only`` に関わらず簡易＋held-out の両方を実行する。
    backup が True のときだけ既存結果の退避を行う（単一コレクション既定動作）。
    """

    # --qrels が指定された場合は別ラベルセットへ切り替える（別ドメイン評価用）。
    qrels_label = "qrels.json"
    if args.qrels:
        global QRELS, EVAL_QUERIES, HELD_OUT_QUERIES, QRELS_SCHEMA_VERSION
        try:
            qrels_path = Path(args.qrels)
            if not qrels_path.is_absolute():
                # README の手順どおり、リポジトリルート基準で解決する。
                qrels_path = (Path(__file__).resolve().parent.parent / qrels_path).resolve()
                # スクリプト親ディレクトリ基準の指定も許容（後方互換）。
                if not qrels_path.is_file():
                    alt = (Path(__file__).resolve().parent / args.qrels).resolve()
                    if alt.is_file():
                        qrels_path = alt
            QRELS = _load_qrels(qrels_path)
        except QrelsError as exc:
            print("エラー: %s" % exc, file=sys.stderr)
            return 1
        EVAL_QUERIES = QRELS["simple_queries"]
        HELD_OUT_QUERIES = QRELS["heldout_queries"]
        QRELS_SCHEMA_VERSION = QRELS["schema_version"]
        qrels_label = qrels_path.name

    # config の必須ヘルパーを起動時に検証する（欠落・import 失敗は明示エラーで終了）。
    try:
        _import_config()
    except ConfigError as exc:
        print("エラー: %s" % exc, file=sys.stderr)
        return 1

    # 設定の出所を 1 箇所に集約（環境変数を起動時に遅延評価したスナップショット）。
    current_settings = _config_value("current_settings", None)
    snapshot = current_settings() if callable(current_settings) else None
    settings: dict[str, Any] = dict(snapshot) if isinstance(snapshot, dict) else {}
    # current_settings() が持たない evaluate 固有キーを環境変数から補完する。
    settings.setdefault(
        "EMBED_MODEL",
        os.environ.get("EMBED_MODEL", str(_config_value("EMBED_MODEL", "") or "")),
    )
    settings.setdefault(
        "CHAT_MODEL",
        os.environ.get("CHAT_MODEL", str(_config_value("CHAT_MODEL", "") or "")),
    )
    settings.setdefault(REPEAT_ENV, os.environ.get(REPEAT_ENV, "1"))

    try:
        problems = _validate_settings()
    except ConfigError as exc:
        print("エラー: %s" % exc, file=sys.stderr)
        return 1
    if problems:
        print("設定値が不正なため中断します: %s" % "; ".join(problems), file=sys.stderr)
        return 1

    fatal_endpoints, endpoint_warnings = _endpoint_problems(settings)
    if fatal_endpoints:
        print(
            "接続先が不正なため中断します: %s（リモートを使う場合は RAG_ALLOW_REMOTE_ENDPOINTS=1 を明示）"
            % "; ".join(fatal_endpoints),
            file=sys.stderr,
        )
        return 1
    for warning in endpoint_warnings:
        print("警告: %s（RAG_ALLOW_REMOTE_ENDPOINTS=1 で緩和中）" % warning, file=sys.stderr)

    try:
        func, query_module = _load_search_function()
    except RuntimeError as exc:
        print("エラー: %s" % exc, file=sys.stderr)
        return 1

    # 複数コレクション指定時は検索先を切り替えるラッパーを使う（既存の func は
    # メタ情報記録用に保持し、実行時だけ差し替える）。
    heldout_only = args.heldout_only and collection_override is None
    if collection_override is not None:
        settings["COLLECTION"] = collection_override
        func = _collection_search(func, query_module, collection_override)

    errors_getter = getattr(query_module, "last_search_errors", None)
    if not callable(errors_getter):
        errors_getter = None

    answer_fn: Callable[..., Any] | None = None
    if args.with_answer:
        candidate = getattr(query_module, "generate_answer", None)
        if not callable(candidate):
            print(
                "エラー: query.generate_answer が無いため --with-answer を実行できません",
                file=sys.stderr,
            )
            return 1
        answer_fn = candidate
        if collection_override is not None:
            answer_fn = _collection_search(answer_fn, query_module, collection_override)
        # 回答生成の HTTP タイムアウトを config.RAG_ANSWER_TIMEOUT へ締める。
        answer_fn = _with_answer_timeout(answer_fn, query_module)

    repeat = _load_repeat(settings)
    top_k = _load_top_k(settings)
    retrieval_k = max(top_k, HELD_OUT_MAX_K)
    collection = _load_collection(settings)
    metadata = _eval_metadata(settings)

    notes: list[str] = []
    for warning in endpoint_warnings:
        notes.append(warning)
    cli_collections = _parse_collections(args.collections)
    if cli_collections:
        notes.append(
            "--collections の対象（マスク済み）: %s"
            % ", ".join(_mask("COLLECTION", name) for name in cli_collections)
        )
    if args.with_answer:
        notes.append(
            "--with-answer: query.generate_answer を呼び answer / sources / "
            "answer_latency_sec を記録（失敗はマスク済みで answer_error に分離、"
            "タイムアウトは config.RAG_ANSWER_TIMEOUT=%ss を適用）"
            % _answer_timeout_seconds()
        )
    settings_sources = (
        ("RAG_TOP_K", "TOP_K"),
        ("RAG_VECTOR_SIZE", "VECTOR_SIZE"),
        ("RAG_COLLECTION", "COLLECTION"),
        ("RAG_CORPUS_DIR", "CORPUS_DIR"),
        ("OLLAMA_URL", "OLLAMA_URL"),
        ("QDRANT_URL", "QDRANT_URL"),
        ("EMBED_MODEL", "EMBED_MODEL"),
        ("CHAT_MODEL", "CHAT_MODEL"),
        (REPEAT_ENV, REPEAT_ENV),
    )
    notes.append(
        "設定の出所: config.current_settings() + 環境変数（値はマスク済み）: %s"
        % ", ".join(
            "%s=%s" % (env_name, _mask(settings.get(key), env_name))
            for env_name, key in settings_sources
        )
    )
    if errors_getter is None:
        notes.append(
            "query.last_search_errors() が無いため検索エラー詳細は record.error に反映しない"
        )
    corpus_root = _corpus_dir(settings)
    missing_labels = [
        source
        for case in QRELS["heldout_queries"]
        for source in case["relevant_sources"]
        if not (corpus_root / source).is_file()
    ]
    if missing_labels:
        notes.append("held-out 正解ファイルがコーパスに不在: %s" % ", ".join(missing_labels))

    def run_heldout() -> tuple[list[dict[str, Any]], list[str]]:
        if args.with_answer:
            return _run_heldout_eval(
                func,
                retrieval_k,
                repeat,
                errors_getter,
                with_answer=True,
                answer_fn=answer_fn,
            )
        return _run_heldout_eval(func, retrieval_k, repeat, errors_getter)

    def run_simple() -> tuple[list[dict[str, Any]], list[str]]:
        if args.with_answer:
            return _run_simple_eval(
                func,
                top_k,
                repeat,
                errors_getter,
                with_answer=True,
                answer_fn=answer_fn,
            )
        return _run_simple_eval(func, top_k, repeat, errors_getter)

    heldout_records, heldout_notes = run_heldout()
    notes.extend(heldout_notes)
    empty_hits = sum(1 for record in heldout_records if record.get("hits") == 0)
    if empty_hits:
        notes.append(
            "held-out で hits=0 が %d 件（検索障害またはコーパス不一致の可能性）" % empty_hits
        )
    if retrieval_k > top_k:
        notes.append(
            "held-out は recall@10/nDCG@10 のため取得件数を %d → %d に引き上げた"
            % (top_k, retrieval_k)
        )

    if heldout_only:
        simple_records, reuse_notes = _reuse_simple_results(
            settings,
            qrels_label,
            QRELS_SCHEMA_VERSION,
            path=output_path,
            with_answer=bool(args.with_answer),
        )
        notes.extend(reuse_notes)
        if simple_records is None:
            notes.append("--heldout-only: 既存の簡易結果が無いため簡易評価も実行した")
            simple_records, simple_notes = run_simple()
            notes.extend(simple_notes)
    else:
        simple_records, simple_notes = run_simple()
        notes.extend(simple_notes)

    notes.append(
        "評価クエリは %s（schema_version=%d）から読み込み"
        % (qrels_label, QRELS_SCHEMA_VERSION)
    )
    notes.append(
        "簡易指標は %d クエリの any-match、held-out は %d クエリの正解ラベル付き"
        % (len(QRELS["simple_queries"]), len(QRELS["heldout_queries"]))
    )
    notes.append("コーパスは config.COLLECTION（環境変数 RAG_COLLECTION で上書き）に依存")
    if repeat == 1:
        notes.append(
            "レイテンシはウォームアップなし1回計測（%s で変更可能）" % REPEAT_ENV
        )
    else:
        notes.append(
            "レイテンシはウォームアップ1回＋%d回計測（%s で変更可能）"
            % (repeat, REPEAT_ENV)
        )
    notes.append(
        "再現メタ情報: embed_model=%s chat_model=%s vector_size=%d corpus_files=%d corpus_sha256=%s"
        % (
            metadata["embed_model"],
            metadata["chat_model"],
            metadata["vector_size"],
            metadata["corpus_file_count"],
            metadata["corpus_sha256"],
        )
    )

    heldout_definitions = {
        "recall_at_k": (
            "各クエリについて、検索結果上位 k 件の source に relevant_sources のいずれかが"
            "含まれれば 1・否則 0 とし、全 held-out クエリで平均（query-level hit@k）。"
            "正解ファイル全体のうち何件回収したかという集合 recall ではない。"
        ),
        "ndcg_at_k": (
            "gain=1 / discount=log2(rank+1) の DCG を上位 k 件で計算し、"
            "IDCG=sum_{r=1..min(|relevant|,k)} 1/log2(r+1) で割ってクエリ平均。"
            "同一ファイルの複数チャンクは最上位の1回だけ計上する。"
        ),
        "label_matching": (
            "正解は passkey-gateway/docs 内の実在ファイル名。ingest.py の source は"
            "リポジトリルート相対パスなので、完全一致・相対パス suffix・basename で照合する。"
        ),
        "failure_policy": "検索失敗ケースは 0 として分母に残す（生存者バイアス回避）。",
        "k_scope": "上位 k 件は source 非空の結果に対して適用する（query.py の hits は常に source を持つため実害なし）。",
        "preflight": "実行前に全 relevant_sources が config.CORPUS_DIR に実在するか確認し、欠落は notes に記録する。",
        "reuse_policy": (
            "--heldout-only は既存 JSON の collection / corpus_sha256 / corpus_file_count / "
            "embed_model / chat_model / vector_size / top_k / schema_version / qrels_label / "
            "qrels_schema_version / with_answer が現行と一致するときのみ簡易結果を再利用する"
            "（欠落キーは再計測の対象。with_answer 未記録の旧形式は False とみなす）。"
        ),
    }
    metric_definitions = {
        "source_keyword_hit_at3": "上位3件の source に expected 語が1つ以上含まれるか（any-match の真偽値。真の recall@k ではない）",
        "content_keyword_hit_all_hits": "全ヒットの text に expected 語が1つ以上含まれるか。source 側と取得深度が異なる",
        "avg_distinct_source_ratio_at3": "上位3件 source のUnit一意性。検索品質ではなく多様性の指標",
        "avg_hits": "モード別の実返却件数（hybrid-rerank も全 top_k を返すよう修正済み）",
        "latency": (
            "RAG_EVAL_REPEAT 回（既定1＝ウォームアップなし1回、2以上でwarmup1回＋N回）"
            "の実測。コーパス規模とLLM呼び出しに強く依存"
        ),
        "latency_stats": (
            "record / summary の avg / median / min / max / stdev（秒）。"
            "avg_latency_sec は後方互換のため avg と同値。"
            "samples 欠落の旧形式では stdev は集計不能として notes に記録し、"
            "数値は後方互換の参考値としてのみ保持する。"
        ),
        "heldout": heldout_definitions,
    }
    if args.with_answer:
        metric_definitions["answer_generation"] = (
            "--with-answer 指定時のみ query.generate_answer(question, hits) を呼び、"
            "answer / sources / answer_latency_sec を record に記録する。"
            "失敗はマスク済みで answer_error へ分離し全体は止めない。"
        )

    # 単一コレクションで --output 未指定時は既存 eval-results.json を上書き前に退避する。
    # 退避失敗は _backup_output 内で警告に格下げされ、ここでは続行する。
    if backup:
        backup_name = _backup_output(output_path)
        if backup_name:
            # 絶対パスは notes に残さない（ベース名のみ）。
            notes.append("既存結果を %s へ退避しました（上書き前）" % backup_name)

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cli_options": {
            # 絶対パスを残さないためベース名のみを記録する。
            "output_path": os.path.basename(output_path),
            "collections": list(cli_collections),
            "with_answer": bool(args.with_answer),
            "heldout_only": bool(args.heldout_only),
        },
        "schema_version": _config_value("SCHEMA_VERSION", 2),
        "qrels_schema_version": QRELS_SCHEMA_VERSION,
        "qrels_label": qrels_label,
        "script": "rag_poc/evaluate.py",
        "search_function": {
            "module": "query",
            "name": getattr(func, "__name__", SEARCH_FN_NAME),
            "preferred_name": SEARCH_FN_NAME,
        },
        "collection": collection,
        "embed_model": metadata["embed_model"],
        "chat_model": metadata["chat_model"],
        "vector_size": metadata["vector_size"],
        "corpus_file_count": metadata["corpus_file_count"],
        "corpus_sha256": metadata["corpus_sha256"],
        "repeat": repeat,
        "modes": list(MODES),
        "top_k": top_k,
        "heldout_retrieval_k": retrieval_k,
        "metric_definitions": metric_definitions,
        "notes": notes,
        "summary": _summarize(simple_records),
        "results": simple_records,
        "heldout": {
            "definitions": heldout_definitions,
            "recall_k": list(HELD_OUT_RECALL_K),
            "ndcg_k": list(HELD_OUT_NDCG_K),
            "queries": QRELS["heldout_queries"],
            "summary": _summarize_heldout(heldout_records),
            "results": heldout_records,
        },
    }
    # 旧形式フォールバックの stdev は集計不能なので、その旨を payload の notes へ集約する。
    for section in (payload["summary"], payload["heldout"]["summary"]):
        for mode_stats in section.values():
            for note in (mode_stats.get("latency_stats") or {}).get("notes") or []:
                if note not in notes:
                    notes.append(note)
    if args.with_answer:
        payload["with_answer"] = True

    try:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except OSError as exc:
        print("%s の書き込みに失敗: %s" % (output_path, exc), file=sys.stderr)
        return 1

    if not heldout_only:
        for mode in MODES:
            stats = payload["summary"][mode]
            print(
                "%-14s src_hit@3=%s content_hit=%s distinct=%s hits=%s latency=%s errors=%d"
                % (
                    mode,
                    stats["source_keyword_hit_at3"],
                    stats["content_keyword_hit_all_hits"],
                    stats["avg_distinct_source_ratio_at3"],
                    stats["avg_hits"],
                    stats["avg_latency_sec"],
                    stats["errors"],
                )
            )
    for mode in MODES:
        stats = payload["heldout"]["summary"][mode]
        print(
            "%-14s heldout recall@1/3/5/10=%s/%s/%s/%s ndcg@3/10=%s/%s errors=%d"
            % (
                mode,
                _fmt(stats["recall_at"].get("1")),
                _fmt(stats["recall_at"].get("3")),
                _fmt(stats["recall_at"].get("5")),
                _fmt(stats["recall_at"].get("10")),
                _fmt(stats["ndcg_at"].get("3")),
                _fmt(stats["ndcg_at"].get("10")),
                stats["errors"],
            )
        )
    print("出力: %s" % output_path)
    for note in notes:
        print("注意: %s" % note, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
