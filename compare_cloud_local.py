#!/usr/bin/env python3
"""クラウド/ローカルの評価成果物を1コマンドで比較する。

新たな API 呼び出しは行わず、既存の eval-results*.json を読み取って
条件（コーパス / チャンク数 / クエリ数 / 指標定義）を明示した比較表を
標準出力と JSON へ出す。成果物は rag_poc/ 配下に置く。

使い方:
  python3 rag_poc/compare_cloud_local.py --output rag_poc/cloud-local-comparison.json
  python3 rag_poc/compare_cloud_local.py --filter cloud --sort ndcg
  python3 rag_poc/compare_cloud_local.py --run-ab --sort ndcg

--run-ab は既存 eval-results*.json だけを横断し、ローカル / クラウドを
条件別にグループ化した A/B 比較表を出す。外部 API 呼び出し・新規計測は
行わない（実計測は ingest.py / evaluate.py / cloud_bench.py を手順どおり
実行し、比較だけを本コマンドに委譲する）。
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
    ("eval-results-cloud-gemini-subset.json", "cloud/gemini-embedding-2 (subset)", "cloud_bench top10 + hit@k"),
]

# A/B 比較（--run-ab）で使うファイル名の判定規則。eval-results-cloud-* を
# クラウド側、それ以外（baseline / ruri-* / alt 等）をローカル側として扱う。
CLOUD_FILENAME_PREFIX = "eval-results-cloud-"

# A/B 比較表に載せる指標（欠損は null、比較は条件が揃った行のみ）。
AB_METRICS: tuple[str, ...] = (
    "ndcg_at_10",
    "set_recall_at_10",
    "hit_at_1",
    "latency_sec",
)


def _pick(d: dict[str, Any], *keys: str) -> Any:
    current: Any = d
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _query_count(data: dict[str, Any], summary: dict[str, Any]) -> int | None:
    """クエリ件数を整数で返す（配列は長さ、数値はそのまま、欠損は None）。"""
    candidates: list[Any] = [
        summary.get("queries"),
        _pick(data, "heldout", "queries"),
        data.get("queries"),
    ]
    heldout_summary = _pick(data, "heldout", "summary") or {}
    if isinstance(heldout_summary, dict):
        for mode_summary in heldout_summary.values():
            if isinstance(mode_summary, dict):
                candidates.append(mode_summary.get("queries"))
    for value in candidates:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, list):
            return len(value)
    return None


def _row(name: str, definition: str, data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("summary") or {}
    heldout_summary = _pick(data, "heldout", "summary") or {}
    # evaluate.py 系は heldout.summary.<mode>、cloud_bench 系は top-level summary。
    # hybrid を優先し、無い場合は辞書順先頭のモードを採用してモード名を保持する。
    eval_mode: str | None = None
    eval_hybrid = heldout_summary.get("hybrid") or {}
    if eval_hybrid:
        eval_mode = "hybrid"
    else:
        for mode_name in sorted(heldout_summary, key=str):
            candidate = heldout_summary[mode_name]
            if isinstance(candidate, dict) and candidate:
                eval_hybrid = candidate
                eval_mode = mode_name
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
        "queries": _query_count(data, summary),
        "repeat": data.get("repeat"),
        "mode": eval_mode,
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


def _condition_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """比較条件（コーパス / チャンク数 / クエリ数 / repeat / 指標定義）のグループ鍵。"""
    return (
        row.get("source"),
        row.get("corpus_file_count"),
        row.get("chunk_count"),
        row.get("queries"),
        row.get("repeat"),
        row.get("definition"),
    )


def _sorted_rows(rows: list[dict[str, Any]], sort_key: str | None) -> list[dict[str, Any]]:
    """同一条件グループ内のみ並べ替える（ndcg/hit 降順、latency 昇順）。

    条件が異なる行はグループ化してグループ順を維持し、相互には並べ替えない。
    欠損値はグループ内で末尾、同値は現行順を維持。
    """
    if sort_key is None:
        return rows
    field = {"ndcg": "ndcg_at_10", "hit": "hit_at_1", "latency": "latency_sec"}[sort_key]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    order: list[tuple[Any, ...]] = []
    for row in rows:
        condition = _condition_key(row)
        if condition not in groups:
            groups[condition] = []
            order.append(condition)
        groups[condition].append(row)

    def sort_key_of(row: dict[str, Any]) -> tuple[bool, float]:
        value = _sort_number(row, field)
        missing = value is None
        number = value if value is not None else 0.0
        return (missing, number if sort_key == "latency" else -number)

    result: list[dict[str, Any]] = []
    for condition in order:
        result.extend(sorted(groups[condition], key=sort_key_of))
    return result


# ---------------------------------------------------------------------------
# --run-ab: 既存成果物だけを読む A/B 比較（外部 API 呼び出し・実計測なし）
# ---------------------------------------------------------------------------


def _side_of(filename: str) -> str:
    """ファイル名から比較編を判定する（eval-results-cloud-* は cloud、他は local）。"""
    return "cloud" if filename.startswith(CLOUD_FILENAME_PREFIX) else "local"


def _display_name(filename: str, side: str) -> str:
    """ファイル名から表示名を作る（例: cloud/text-embedding-3-small-full）。"""
    stem = filename
    if stem.startswith("eval-results-"):
        stem = stem[len("eval-results-"):]
    if stem.endswith(".json"):
        stem = stem[:-len(".json")]
    return f"{side}/{stem}"


def _infer_definition(data: dict[str, Any]) -> str:
    """成果物の構造から指標定義を推定する（A/B 比較用）。

    heldout.summary があれば evaluate.py 系、トップレベル summary に
    ndcg_at_10 / set_recall_at_10 があれば cloud_bench 系とみなす。
    """
    if _pick(data, "heldout", "summary"):
        return "evaluate query-level hit@k"
    summary = data.get("summary")
    if isinstance(summary, dict) and (
        "ndcg_at_10" in summary or "set_recall_at_10" in summary
    ):
        return "cloud_bench top10 + hit@k"
    return "unknown"


def _discover_result_files() -> list[Path]:
    """HERE 直下の eval-results*.json をファイル名順で列挙する。"""
    return sorted(HERE.glob("eval-results*.json"), key=lambda path: path.name)


def _ab_rows() -> tuple[list[dict[str, Any]], list[str]]:
    """既存 eval-results*.json を横断読み取り、A/B 比較行を構築する。

    外部 API 呼び出し・新規計測は一切行わない。壊れた JSON は skipped へ
    記録してスキップし、比較表全体が止まらないようにする。
    """
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for path in _discover_result_files():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append(f"{path.name}: {type(exc).__name__}")
            continue
        if not isinstance(data, dict):
            skipped.append(f"{path.name}: JSON object ではない")
            continue
        side = _side_of(path.name)
        row = _row(_display_name(path.name, side), _infer_definition(data), data)
        row["side"] = side
        row["file"] = path.name
        rows.append(row)
    return rows, skipped


def _condition_of(row: dict[str, Any]) -> dict[str, Any]:
    """比較条件を人間が読める辞書で返す（定義を含む 5 条件）。"""
    return {
        "corpus": row.get("source"),
        "corpus_file_count": row.get("corpus_file_count"),
        "chunk_count": row.get("chunk_count"),
        "queries": row.get("queries"),
        "repeat": row.get("repeat"),
        "definition": row.get("definition"),
    }


def _mark_condition_match(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """反対側に同一条件の行があるかを明示フラグとして各 row へ付ける。

    条件鍵は現行 --sort と同じ _condition_key（定義を含む）を使う。
    反対側に一致が無い行は「条件不一致」として直接比較不可を明示する。
    """
    keys_by_side: dict[str, set[tuple[Any, ...]]] = {"local": set(), "cloud": set()}
    for row in rows:
        keys_by_side[row["side"]].add(_condition_key(row))
    for row in rows:
        other = "cloud" if row["side"] == "local" else "local"
        matched = _condition_key(row) in keys_by_side[other]
        row["condition_match"] = matched
        row["condition_mismatch"] = not matched
    return rows


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    """比較表の 1 行をコンパクトな辞書で返す。"""
    return {
        "file": row.get("file"),
        "name": row.get("name"),
        "side": row.get("side"),
        "embed_model": row.get("embed_model"),
        "mode": row.get("mode"),
        "definition": row.get("definition"),
        "condition_mismatch": row.get("condition_mismatch"),
        "metrics": {metric: row.get(metric) for metric in AB_METRICS},
    }


def _ab_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """比較条件ごとに local / cloud の行をまとめた比較表を返す。

    グループ鍵は _condition_key（現行 --sort と同じ条件）。同一条件の
    両側が揃ったグループだけ comparable=True とし、片側しか無い
    グループには比較不可の理由を note として付ける。
    """
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    order: list[tuple[Any, ...]] = []
    for row in rows:
        key = _condition_key(row)
        if key not in groups:
            groups[key] = {
                "condition": _condition_of(row),
                "local": [],
                "cloud": [],
            }
            order.append(key)
        groups[key][row["side"]].append(_compact_row(row))

    result: list[dict[str, Any]] = []
    for key in order:
        group = groups[key]
        local_rows = group["local"]
        cloud_rows = group["cloud"]
        comparable = bool(local_rows and cloud_rows)
        entry: dict[str, Any] = {
            "condition": group["condition"],
            "comparable": comparable,
            "local": local_rows,
            "cloud": cloud_rows,
        }
        if not comparable:
            missing = "cloud" if local_rows else "local"
            entry["note"] = f"反対側（{missing}）に同一条件の成果物が無いため直接比較不可"
        result.append(entry)
    return result


def _print_ab_table(rows: list[dict[str, Any]], stream: Any) -> None:
    """A/B 比較表を人間向けに stderr へ出力する（stdout は JSON のみ）。"""
    print(
        "[compare] A/B: 既存 eval-results*.json のみ（外部 API 呼び出しなし）",
        file=stream,
    )
    print(
        "[compare] [条件不一致] は反対側に同一条件の行が無いことを示す。"
        "ソートは同一条件グループ内のみ。",
        file=stream,
    )
    for row in rows:
        flag = "[条件不一致]" if row.get("condition_mismatch") else "[条件一致]"
        metrics = " ".join(
            f"{metric}={row.get(metric)}" for metric in AB_METRICS
        )
        print(
            f"[compare] {row.get('side'):<5} {flag} {row.get('file')} | {metrics}",
            file=stream,
        )


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
        help="並び順: ndcg は ndcg_at_10 降順、latency は latency_sec 昇順、hit は hit_at_1 降順。同一条件グループ内のみ有効（既定は現行順）",
    )
    parser.add_argument(
        "--run-ab",
        action="store_true",
        help="既存 eval-results*.json を local/cloud 別・条件別にグループ化した A/B 比較表を出す（外部 API 呼び出しなし・新規計測なし）",
    )
    args = parser.parse_args()

    if args.run_ab:
        rows, skipped = _ab_rows()
        rows = _filtered_rows(rows, args.filter)
        rows = _mark_condition_match(rows)
        rows = _sorted_rows(rows, args.sort)
        payload: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "run-ab",
            "notes": [
                "既存の eval-results*.json を読み取った A/B 比較であり、新たな API 呼び出し・実計測は無い。",
                "side はファイル名で判定する（eval-results-cloud-* を cloud、その他を local）。",
                "condition_match は反対側（local/cloud）に _condition_key（コーパス / チャンク数 / クエリ数 / repeat / 指標定義）が同一の行があるかを示す。false の行には [条件不一致] フラグを付け、直接比較不可であることを明示する。",
                "ab_comparison は同一条件グループごとに local / cloud の行をまとめた比較表。comparable=false のグループは片側のみで比較不可。",
                "--sort は現行と同じグループ化を再利用し、同一条件の行グループ内でのみ有効。条件が異なる行は相互に並べ替えない。",
                "set_recall_at_10 は集合 recall、hit_at_1 は query-level hit@1。ndcg_at_10 は definition（evaluate / cloud_bench）に依存するため条件列を参照。",
            ],
            "skipped": skipped,
            "ab_comparison": _ab_summary(rows),
            "rows": rows,
        }
    else:
        rows = []
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
                "--sort はコーパス / チャンク数 / クエリ数 / repeat / 指標定義が同一の行グループ内でのみ有効。条件が異なる行はグループ化して順序を維持し、相互には並べ替えない。",
                "mode は heldout.summary から指標を採用したモード名（hybrid 優先、無ければ辞書順先頭。cloud_bench 系は null）。",
            ],
            "rows": rows,
        }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output).resolve()
        out.write_text(text, encoding="utf-8")
        # ログは stderr へ。stdout は常に JSON のみにする。
        print("[compare] wrote", out, file=sys.stderr, flush=True)
    if args.run_ab:
        # 人間向け比較表は stderr へ。stdout は常に JSON のみにする。
        _print_ab_table(rows, sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
