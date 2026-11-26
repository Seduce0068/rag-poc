"""埋め込みモデルのコサイン類似度を比較する CLI（Qdrant 不使用）。

使い方（リポジトリルート / rag_poc 内のどちらからでも可）::

    python3 rag_poc/compare_embeddings.py
    python3 rag_poc/compare_embeddings.py --models qwen2.5:3b qwen2.5:7b
    python3 rag_poc/compare_embeddings.py --with-qrels rag_poc/qrels.json

サブセット (rag_poc/corpus-subset/) の Markdown を Ollama /api/embed で埋め込み、
固定の比較クエリ 5 件とのコサイン類似度をインメモリで計算し、
rag_poc/embedding-comparison.json へ出力する。

``--with-qrels`` 指定時は qrels の heldout_queries を固定クエリと同じ埋め込み
バッチへ含め、ファイル埋め込みを再利用して top1 ファイル名の hit@1 をモデルごとに
記録する（Ollama 呼び出し回数は増えない）。top_score 自体は qrels の
Recall@k / nDCG@k ではない（notes に明記する）。

依存は標準 library + requests（+ numpy: 無い場合は pure Python にフォールバック）。
接続先・タイムアウトは config.py（OLLAMA_URL / OLLAMA_TIMEOUT）を参照する。
このスクリプト自身は HTTP 呼び出しを実行するだけで、外部サービスへは送信しない
（実行するのは呼び出し元。既定のループバック以外は拒否する）。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import requests

from config import (
    COMPARISON_QUERIES,
    DEFAULT_MODELS,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    is_fatal_endpoint_problem,
    mask_secrets,
    validate_endpoint,
)
from rust_bridge import RustUnavailable, cosine_batch, run_op

try:
    # Ollama 埋め込みの共通 API。HTTP / リトライ / レスポンス検証を
    # 一箇所に集約するため、ここでは _ollama_embed 経由で呼び出す。
    from ollama_client import embed as _ollama_embed
except ImportError:  # 共通クライアント未導入でも --help は動かす
    _ollama_embed = None  # type: ignore[assignment]

try:  # numpy は利用可だが、無い環境でも動くように pure Python へ退避する
    import numpy as np
except ImportError:  # pragma: no cover - 環境依存のフォールバック
    np = None  # type: ignore[assignment]

# DEFAULT_MODELS / COMPARISON_QUERIES は config の定義を唯一の情報源とする
# （読み取りで両者の値が同一であることを確認済み）。
# MAX_CHARS は config.RERANK_MAX_CHARS(1200) と値が異なるため置換せず、
# 既存の 2000 を維持する（挙動変更を避けるため）。
MAX_CHARS = 2000
SCRIPT_DIR = Path(__file__).resolve().parent
CORPUS_SUBSET_DIR = SCRIPT_DIR / "corpus-subset"
OUTPUT_PATH = SCRIPT_DIR / "embedding-comparison.json"

# Rust bridge の利用／フォールバックを notes に記録するための定型文
BRIDGE_COSINE_NOTE = "cosine_batch は Rust bridge を使用（質問×ファイルを1回で計算）"
FALLBACK_COSINE_NOTE = "Rust bridge unavailable, fell back to Python cosine"
FALLBACK_DISTRIBUTION_NOTE = (
    "Rust distribution op unavailable, fell back to Python distribution"
)
# コサイン代理値と qrels 指標の違いを明示する必須注記。
PROXY_METRIC_NOTE = (
    "top_score は固定5クエリ×6ファイルのコサイン代理値であり、"
    "qrels の Recall@k / nDCG@k ではない"
)
QRELS_HIT_NOTE = (
    "--with-qrels: qrels の heldout_queries を固定クエリと同一バッチで埋め込み、"
    "ファイル埋め込みを再利用して top1 ファイル名の hit@1 を記録"
    "（Ollama 追加呼び出しなし）"
)


def _add_note(notes: list[str] | None, message: str) -> None:
    """同じ注記を二重に追加しないように notes へ追記する。"""
    if notes is not None and message not in notes:
        notes.append(message)


def normalize(text: str) -> str:
    """改行を含む空白類を単一スペースへ正規化し、先頭 MAX_CHARS 文字に切り詰める。

    全モデル・全クエリで同じ関数を通すことで比較条件を完全に揃える。
    """
    return " ".join(text.split())[:MAX_CHARS]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """コサイン類似度を返す（numpy 優先、無ければ pure Python）。"""
    if len(a) != len(b):
        raise ValueError(f"埋め込み次元が不一致です: {len(a)} vs {len(b)}")
    if np is not None:
        va = np.asarray(a, dtype=float)
        vb = np.asarray(b, dtype=float)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _distribution(
    scores: Sequence[float], notes: list[str] | None = None
) -> dict[str, float]:
    """6 ファイル分のスコア分布（min / max / mean / 母集団 stdev）。

    Rust の distribution op を優先し、使えなければ既存の Python 実装へ
    フォールバックする。丸めは呼び出し側（ここ）で round(..., 6) を行う。
    """
    keys = ("min", "max", "mean", "stdev")
    try:
        result = run_op("distribution", {"scores": list(scores)})
        if isinstance(result, dict) and all(key in result for key in keys):
            return {key: round(float(result[key]), 6) for key in keys}
    except RustUnavailable as exc:
        _add_note(notes, f"{FALLBACK_DISTRIBUTION_NOTE}: {exc}")
    else:
        _add_note(notes, f"{FALLBACK_DISTRIBUTION_NOTE}: unexpected result shape")
    return {
        "min": round(min(scores), 6),
        "max": round(max(scores), 6),
        "mean": round(statistics.mean(scores), 6),
        "stdev": round(statistics.pstdev(scores), 6),  # ddof=0
    }


def _embed_texts(
    session: requests.Session, model: str, texts: Sequence[str]
) -> list[list[float]]:
    """Ollama 埋め込みを ollama_client.embed 経由で取得する。

    prefix は使わず、モデルは引数の ``model`` を明示する。タイムアウトは
    config.OLLAMA_TIMEOUT 准拠。``session`` は既存 call site 互換のため
    受け取るが、HTTP / リトライは共通クライアントへ委譲したため未使用。
    """
    if _ollama_embed is None:
        raise RuntimeError("ollama_client.embed を import できませんでした")
    return _ollama_embed(
        list(texts),
        model=model,
        prefix="",
        batch_size=None,
        validate_dim=None,
        url=OLLAMA_URL,
        timeout=OLLAMA_TIMEOUT,
    )


def _source_matches(source: str, relevant_sources: Sequence[str]) -> bool:
    """top1 ファイル名が正解ラベルのいずれかと一致するかを判定する。

    qrels の ``relevant_sources`` は basename だが、相対パス形式もあり得るため
    evaluate.py の ``_source_matches`` と同じく完全一致・suffix・basename で照合する。
    """
    normalized = source.replace("\\", "/").strip().lower()
    if not normalized:
        return False
    base = normalized.rsplit("/", 1)[-1]
    for expected in relevant_sources:
        target = expected.replace("\\", "/").strip().lower()
        if not target:
            continue
        if normalized == target or normalized.endswith("/" + target.lstrip("/")):
            return True
        if base == target.rsplit("/", 1)[-1]:
            return True
    return False


def _resolve_qrels_path(raw: str) -> Path:
    """--with-qrels のパスを解決する（リポジトリルート→スクリプト directory の順）。"""
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    candidates = (SCRIPT_DIR.parent / path, SCRIPT_DIR / path)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return candidates[0].resolve()


def _load_heldout(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """qrels JSON の heldout_queries を読み込み最小検証する。

    戻り値は (正規化済みケース一覧, エラーメッセージ)。エラー時はケース一覧は空。
    エラー出力にクエリ本文やパスは含めない（呼び出し側でマスクするため）。
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        return [], f"qrels の読み込みに失敗: {type(exc).__name__}: {exc}"
    if not isinstance(data, dict):
        return [], "qrels の最上位は JSON オブジェクトである必要があります"
    cases = data.get("heldout_queries")
    if not isinstance(cases, list) or not cases:
        return [], "qrels に heldout_queries がありません"
    normalized: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            return [], f"heldout_queries[{index}] がオブジェクトではありません"
        query = case.get("query")
        relevant = case.get("relevant_sources")
        if not isinstance(query, str) or not query.strip():
            return [], f"heldout_queries[{index}].query が不正です"
        if (
            not isinstance(relevant, list)
            or not relevant
            or not all(isinstance(item, str) and item.strip() for item in relevant)
        ):
            return [], f"heldout_queries[{index}].relevant_sources が不正です"
        normalized.append(
            {
                "id": str(case.get("id", "") or ""),
                "query": query,
                "relevant_sources": [item.strip() for item in relevant],
            }
        )
    return normalized, None


def _compare_model(
    session: requests.Session,
    model: str,
    files: Sequence[Path],
    file_texts: Sequence[str],
    query_texts: Sequence[str],
    notes: list[str] | None = None,
    qrels_cases: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """1 モデル分の比較結果を組み立てる。

    ``qrels_cases`` が指定された場合、heldout クエリを固定クエリと同じ埋め込み
    バッチへ含め（Ollama 呼び出し回数は増えない）、既存のファイル埋め込みを
    再利用して top1 ファイル名の hit@1 を計算する。
    """
    # ファイルは 1 ファイル 1 リクエスト（計 6 回）。クエリは 1 バッチで 1 回。
    # heldout クエリがあっても同じバッチに追加するだけなので呼び出し回数は同じ。
    file_vectors = [_embed_texts(session, model, [text])[0] for text in file_texts]
    extra_texts = [normalize(str(case["query"])) for case in (qrels_cases or [])]
    all_query_texts = list(query_texts) + extra_texts
    all_query_vectors = _embed_texts(session, model, all_query_texts)
    query_vectors = all_query_vectors[: len(query_texts)]
    heldout_vectors = all_query_vectors[len(query_texts) :]

    dim = len(file_vectors[0])
    if any(len(vector) != dim for vector in file_vectors) or any(
        len(vector) != dim for vector in all_query_vectors
    ):
        raise RuntimeError(f"埋め込み次元がモデル内で不一致です (dim={dim})")

    # 質問×ファイルの全ペアを 1 回の cosine_batch 呼び出しで計算する。
    # Rust が使えない場合は _cosine にフォールバックする（notes に記録）。
    n_files = len(file_vectors)
    pairs = [
        (vector, file_vector)
        for vector in all_query_vectors
        for file_vector in file_vectors
    ]
    raw_scores: list[float] | None = None
    try:
        raw_scores = cosine_batch(pairs)
        if len(raw_scores) != len(pairs):
            raise RuntimeError("cosine_batch の結果件数がペア数と不一致です")
        _add_note(notes, BRIDGE_COSINE_NOTE)
    except RustUnavailable as exc:
        raw_scores = None
        _add_note(notes, f"{FALLBACK_COSINE_NOTE}: {exc}")

    def scores_for(vector: Sequence[float], index: int) -> list[float]:
        """インデックスに対応するファイル側スコア列を返す（生値）。"""
        if raw_scores is not None:
            offset = index * n_files
            return raw_scores[offset : offset + n_files]
        return [_cosine(vector, file_vector) for file_vector in file_vectors]

    queries: list[dict[str, Any]] = []
    top_scores: list[float] = []
    for index, query_vector in enumerate(query_vectors):
        scores = scores_for(query_vector, index)
        top_index = max(range(len(scores)), key=lambda i: scores[i])
        top_score = scores[top_index]
        top_scores.append(top_score)
        queries.append(
            {
                "query": COMPARISON_QUERIES[index],
                "top_source": files[top_index].name,
                "top_score": round(top_score, 6),
                "distribution": _distribution(scores, notes),
            }
        )

    result: dict[str, Any] = {
        "dim": dim,
        "queries": queries,
        "avg_top_score": round(statistics.mean(top_scores), 6),
    }
    if qrels_cases is not None:
        case_results: list[dict[str, Any]] = []
        hits = 0
        for index, case in enumerate(qrels_cases):
            scores = scores_for(heldout_vectors[index], len(query_vectors) + index)
            top_index = max(range(len(scores)), key=lambda i: scores[i])
            top_source = files[top_index].name
            relevant = [
                str(value).strip()
                for value in case.get("relevant_sources", [])
                if str(value).strip()
            ]
            hit = _source_matches(top_source, relevant)
            hits += int(hit)
            case_results.append(
                {
                    "id": str(case.get("id", "") or ""),
                    "query": str(case.get("query", "") or ""),
                    "top_source": top_source,
                    "relevant_sources": relevant,
                    "hit": hit,
                }
            )
        result["qrels"] = {
            "queries": len(case_results),
            "hits": hits,
            "hit_at_1": round(hits / len(case_results), 6) if case_results else None,
            "cases": case_results,
        }
    return result


def _write_output(
    generated_at: str, models: dict[str, Any], notes: list[str]
) -> None:
    """指定スキーマの JSON を一時ファイル経由で原子的に書き出す。"""
    payload = {"generated_at": generated_at, "models": models, "notes": notes}
    tmp_path = OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(OUTPUT_PATH)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="埋め込みモデルのコサイン類似度を比較する（Qdrant 不使用・固定クエリ 5 件）"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        metavar="MODEL",
        help=f"比較する Ollama モデル名（既定: {' '.join(DEFAULT_MODELS)}）",
    )
    parser.add_argument(
        "--with-qrels",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "qrels JSON の heldout_queries を使って各モデルの top1 ファイル名 hit@1 を記録する"
            "（Ollama 追加呼び出しなし）"
        ),
    )
    args = parser.parse_args(argv)
    models_arg = list(dict.fromkeys(args.models))  # 重複除去・順序保持

    notes = [
        PROXY_METRIC_NOTE,
        "埋め込みは Ollama /api/embed のみを使用。Qdrant は使わずコサイン類似度はインメモリで算出。",
        "前処理は全モデル共通: 改行を含む空白類を単一スペースへ正規化し先頭 2000 文字で切り詰め。",
        "ファイル埋め込みは1ファイル1リクエスト（6回）、クエリ埋め込みはモデルごとに1バッチリクエスト。",
        "distribution は6ファイル分のコサイン類似度の min/max/mean/母集団 stdev（ddof=0）。",
        "top_source はサブセット内のファイル名（basename）。比較クエリは固定5件。",
    ]

    qrels_cases: list[dict[str, Any]] | None = None
    if args.with_qrels:
        loaded, error = _load_heldout(_resolve_qrels_path(args.with_qrels))
        if error is not None:
            notes.append(f"--with-qrels: {mask_secrets(error)}（hit@1 は記録しません）")
        else:
            qrels_cases = loaded
            notes.append(QRELS_HIT_NOTE)

    generated_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )

    files = sorted(path for path in CORPUS_SUBSET_DIR.glob("*.md") if path.is_file())
    if not files:
        notes.append(f"サブセットの Markdown が見つかりません: {CORPUS_SUBSET_DIR}")
        _write_output(generated_at, {}, notes)
        print("[compare] サブセットの Markdown が見つかりません", file=sys.stderr, flush=True)
        return 1

    file_texts = [
        normalize(path.read_text(encoding="utf-8", errors="replace")) for path in files
    ]
    query_texts = [normalize(query) for query in COMPARISON_QUERIES]

    problems = validate_endpoint(OLLAMA_URL, "OLLAMA_URL")
    allow_remote = os.environ.get("RAG_ALLOW_REMOTE_ENDPOINTS") == "1"
    fatal = [p for p in problems if is_fatal_endpoint_problem(p)]
    warnings = [p for p in problems if not is_fatal_endpoint_problem(p)]
    if fatal or (problems and not allow_remote):
        # allowlist 不一致・平文 HTTP・不正 scheme/port はオプトイン済みでも致命。
        # 文字列一致ではなく config.is_fatal_endpoint_problem() で判定する。
        for problem in (fatal or problems):
            notes.append(f"接続先検証: {mask_secrets(problem)} (致命)")
            print(
                f"[compare] 接続先検証: {mask_secrets(problem)} (致命)",
                file=sys.stderr,
                flush=True,
            )
        _write_output(generated_at, {}, notes)
        print(
            "[compare] 接続先に致命的な問題があるため比較を実行しません",
            file=sys.stderr,
            flush=True,
        )
        return 2
    for problem in warnings:
        notes.append(f"接続先検証: {mask_secrets(problem)} (警告)")
        print(
            f"[compare] 接続先検証: {mask_secrets(problem)} (警告)",
            file=sys.stderr,
            flush=True,
        )
    if warnings:
        notes.append("RAG_ALLOW_REMOTE_ENDPOINTS=1 のため警告を無視して継続")

    print(
        f"[compare] files={len(files)} queries={len(query_texts)} models={len(models_arg)}",
        flush=True,
    )
    results: dict[str, Any] = {}
    with requests.Session() as session:
        for model in models_arg:
            try:
                results[model] = _compare_model(
                    session, model, files, file_texts, query_texts, notes,
                    qrels_cases=qrels_cases,
                )
                print(f"[compare] {model}: dim={results[model]['dim']}", flush=True)
            except Exception as exc:  # noqa: BLE001 - モデル単位で失敗を記録して継続
                message = mask_secrets(str(exc))
                notes.append(f"モデル {model} の比較に失敗: {type(exc).__name__}: {message}")
                print(
                    f"[compare] {model}: 失敗 ({type(exc).__name__})",
                    file=sys.stderr,
                    flush=True,
                )

    _write_output(generated_at, results, notes)
    print(f"[compare] wrote {OUTPUT_PATH}", flush=True)
    return 0 if len(results) == len(models_arg) else 1


if __name__ == "__main__":
    sys.exit(main())
