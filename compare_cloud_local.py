#!/usr/bin/env python3
"""クラウド/ローカルの評価成果物を1コマンドで比較する。

新たな API 呼び出しは行わず、既存の eval-results*.json を読み取って
条件（コーパス / チャンク数 / クエリ数 / 指標定義）を明示した比較表を
標準出力と JSON へ出す。成果物は rag_poc/ 配下に置く。

使い方:
  python3 rag_poc/compare_cloud_local.py --output rag_poc/cloud-local-comparison.json
  python3 rag_poc/compare_cloud_local.py --filter cloud --sort ndcg
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# ファイル名 -> (表示名, 指標定義の出所)
DEFAULT_ARTIFACTS: list[tuple[str, str, str]] = [
    ("eval-results-baseline.json", "local/qwen2.5:3b (full)", "evaluate query-level hit@k"),
    ("eval-results-ruri-full.json", "local/Ruri (full)", "evaluate query-level hit@k"),
    ("eval-results-cloud-full.json", "cloud/text-embedding-3-small (full)", "cloud_bench top10 + hit@k"),
    ("eval-results-cloud-pipeline.json", "cloud/text-embedding-3-small (Qdrant)", "evaluate query-level hit@k"),
    ("eval-results-cloud-small-subset.json", "cloud/text-embedding-3-small (subset)", "cloud_bench top10 + hit@k"),
    ("eval-results-cloud-qwen3-subset.json", "cloud/qwen3-embedding-8b (subset)", "cloud_bench top10 + hit@k"),
]


def _pick(d: dict[str, Any], *keys: str) -> Any:
    current: Any = d
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _row(name: str, definition: str, data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("summary") or {}
    heldout_summary = _pick(data, "heldout", "summary") or {}
    # evaluate.py 系は heldout.summary.<mode>、cloud_bench 系は top-level summary。
    # hybrid を決め打ちせず、利用可能なモードから選ぶ（hybrid 優先）。
    eval_hybrid = heldout_summary.get("hybrid") or {}
    if not eval_hybrid:
        for mode_summary in heldout_summary.values():
            if isinstance(mode_summary, dict) and mode_summary:
                eval_hybrid = mode_summary
                break
    recall_10 = _pick(eval_hybrid, "recall_at", "10")
    if recall_10 is None:
        recall_10 = summary.get("set_recall_at_10", summary.get("recall_at_10"))
    ndcg_10 = _pick(eval_hybrid, "ndcg_at", "10")
    if ndcg_10 is None:
        ndcg_10 = summary.get("ndcg_at_10")
    hit_1 = summary.get("hit_at_1")
    if hit_1 is None:
        hit_1 = _pick(eval_hybrid, "recall_at", "1")
    latency = summary.get("total_search_latency_sec")
    if latency is None:
        latency = _pick(eval_hybrid, "latency_stats", "avg")
    rerank = summary.get("rerank_ndcg_at_5")
    return {
        "name": name,
        "source": data.get("qrels_label") or data.get("corpus"),
        "definition": definition,
        "corpus_file_count": data.get("corpus_file_count"),
        "chunk_count": data.get("chunk_count"),
        "queries": data.get("queries") or _pick(data, "heldout", "queries"),
        "repeat": data.get("repeat"),
        "hit_at_1": hit_1,
        "set_recall_at_10": recall_10,
        "ndcg_at_10": ndcg_10,
        "rerank_ndcg_at_5": rerank,
        "rerank_query_count": summary.get("rerank_query_count"),
        "latency_sec": latency,
        "embed_model": data.get("embed_model") or data.get("model"),
        "generated_at": data.get("generated_at"),
    }


def _filtered_rows(rows: list[dict[str, Any]], needle: str | None) -> list[dict[str, Any]]:
    """name / embed_model / definition の部分一致で行を絞り込む（大小文字無視）。"""
    if not needle or not needle.strip():
        return rows
    lowered = needle.strip().lower()
    return [
        row
        for row in rows
        if any(
            lowered in str(row.get(field) or "").lower()
            for field in ("name", "embed_model", "definition")
        )
    ]


def _sort_number(row: dict[str, Any], field: str) -> float | None:
    """ソート対象の数値を返す（bool / 非数値 / 欠損は None＝末尾へ）。"""
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _sorted_rows(rows: list[dict[str, Any]], sort_key: str | None) -> list[dict[str, Any]]:
    """ndcg / hit は降順、latency は昇順。欠損値は末尾、同値は現行順を維持。"""
    if sort_key is None:
        return rows
    field = {"ndcg": "ndcg_at_10", "hit": "hit_at_1", "latency": "latency_sec"}[sort_key]

    def key(row: dict[str, Any]) -> tuple[bool, float]:
        value = _sort_number(row, field)
        missing = value is None
        number = value if value is not None else 0.0
        return (missing, number if sort_key == "latency" else -number)

    return sorted(rows, key=key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="比較 JSON の出力先")
    parser.add_argument(
        "--filter",
        default=None,
        help="名前 / embed_model / definition の部分一致で行を絞り込む",
    )
    parser.add_argument(
        "--sort",
        choices=("ndcg", "latency", "hit"),
        default=None,
        help="並び順: ndcg は ndcg_at_10 降順、latency は latency_sec 昇順、hit は hit_at_1 降順（既定は現行順）",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for filename, label, definition in DEFAULT_ARTIFACTS:
        path = HERE / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_row(label, definition, data))

    rows = _filtered_rows(rows, args.filter)
    rows = _sorted_rows(rows, args.sort)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "既存の eval-results*.json を読み取った比較であり、新たな API 呼び出しは無い。",
            "set_recall_at_10 は集合 recall、hit_at_1 は evaluate.py と同じ query-level hit@1。",
            "ndcg_at_10 は成果物ごとの定義（evaluate は上位10件・ファイル1回、cloud_bench は top10）に依存するため定義列を参照。",
            "コーパス / チャンク数 / クエリ数 / repeat が異なる行は直接比較不可であるため、比較条件を揃えてから判定すること。",
        ],
        "rows": rows,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output).resolve()
        out.write_text(text, encoding="utf-8")
        # ログは stderr へ。stdout は常に JSON のみにする。
        print("[compare] wrote", out, file=sys.stderr, flush=True)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
