#!/usr/bin/env python3
"""opencodeGO 埋め込みでコーパス-subset の検索品質を計測する。

使い方:
  # dsh.env の OPENCODE_GO_API_KEY を環境変数へ読み込んで実行
  set -a; source ~/.config/opencode/dsh.env; set +a
  RAG_CLOUD_ENABLED=1 python3 rag_poc/cloud_bench.py
  RAG_CLOUD_ENABLED=1 python3 rag_poc/cloud_bench.py --repeat 3 --with-generation

結果は eval-results-cloud-subset.json へ保存する（退避対象）。
秘密情報の値は出力・保存しない。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cloud_client
from config import mask_secrets

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus-subset"
QRELS = HERE / "qrels-subset.json"
OUTPUT = HERE / "eval-results-cloud-subset.json"

# コーパスハッシュはハードコードしない。設定ファイル／環境変数から読み、
# 無ければ payload へ入れない（eval-results-cloud-pipeline.json の値と
# 同一にしたい場合は利用側でこの設定を与える）。
CORPUS_SHA256_ENV = "RAG_CLOUD_CORPUS_SHA256"
CORPUS_SHA256_CONFIG = HERE / "cloud-bench-config.json"

RECALL_K = (1, 3, 5, 10)
NDCG_K = 10
RERANK_K = 5
GENERATION_QUERY_LIMIT = 3


def _chunk(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    words = text.split()
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, len(words), step)]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _source(text: str) -> str:
    first = text.splitlines()[0] if text else ""
    if first.startswith("title:"):
        return first.split(":", 1)[1].strip()
    return first.strip()


def _configured_corpus_sha256() -> str | None:
    """設定された corpus_sha256 を返す（無ければ None＝payload から省略）。"""
    value = os.environ.get(CORPUS_SHA256_ENV)
    if value and value.strip():
        return value.strip()
    try:
        data = json.loads(CORPUS_SHA256_CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("corpus_sha256")
    if isinstance(raw, (str, int)) and str(raw).strip():
        return str(raw).strip()
    return None


def _extract_rank_ids(reply: str) -> list[int]:
    """JSON の ``rank`` キー（整数配列）のみを抽出する。

    ``{"rank": [...]}`` 前後に説明文が混入していても許容する。
    rank が整数配列でない場合・JSON が見つからない場合は ValueError。
    """
    text = str(reply)
    decoder = json.JSONDecoder()
    attempts = [text.strip()]
    attempts.extend(text[pos:] for pos, ch in enumerate(text) if ch == "{")
    for attempt in attempts:
        if not attempt:
            continue
        try:
            data, _ = decoder.raw_decode(attempt)
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict) or "rank" not in data:
            continue
        rank = data["rank"]
        if not isinstance(rank, list) or any(
            not isinstance(item, int) or isinstance(item, bool) for item in rank
        ):
            raise ValueError("rank は整数配列である必要があります")
        return rank
    raise ValueError("JSON の rank 整数配列が見つかりません")


def _embedding_metrics(ranked: list[tuple[str, float]], relevant: set[str]) -> dict:
    """1 クエリの検索指標を evaluate.py と同じ定義で計算する。"""
    hits = [src in relevant for src, _ in ranked]

    # evaluate.py の recall@k と同じ query-level hit@k（上位 k に正解が1件でもあれば 1）。
    hit_at = {str(k): 1 if any(hits[:k]) else 0 for k in RECALL_K}

    # 旧来の集合 recall は別キーへ退避する。互換のため旧キーにも同じ値を残す。
    set_recall_at = {
        str(k): round(sum(hits[:k]) / max(1, len(relevant)), 4) for k in RECALL_K
    }

    # nDCG@10 は hits[:10] のみ。ファイル単位で1回だけ計上（evaluate.py 準拠）。
    dcg = 0.0
    credited: set[str] = set()
    for rank, (src, _) in enumerate(ranked[:NDCG_K], start=1):
        key = src.lower()
        if src in relevant and key not in credited:
            credited.add(key)
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(
        1.0 / math.log2(i + 2) for i in range(min(len(relevant), NDCG_K))
    )
    return {
        "hit_at": hit_at,
        "set_recall_at": set_recall_at,
        "recall_at": dict(set_recall_at),
        "ndcg_at_10": round(dcg / ideal, 4) if ideal else 0.0,
    }


def _flatten_metrics(metrics: dict) -> dict[str, float]:
    """repeat 回分の平均化のため指標を数値フラット辞書へ変換する。"""
    flat: dict[str, float] = {}
    for name in ("hit_at", "set_recall_at"):
        for k, v in metrics[name].items():
            flat["%s_%s" % (name, k)] = float(v)
    flat["ndcg_at_10"] = float(metrics["ndcg_at_10"])
    return flat


def _run_generation(record: dict, query: dict, candidates: list) -> float:
    """1 クエリ分の rerank + 回答生成を実行し、レイテンシ秒を返す。

    保存する reply / answer / error は必ず config.mask_secrets を通す。
    パース失敗時は generation_error に記録し、nDCG キーは作成しない（0 にしない）。
    """
    errors: list[str] = []
    listing = "\n".join(
        "[%d] %s: %s" % (i + 1, src, chunk[:300])
        for i, (_, src, chunk) in enumerate(candidates)
    )
    prompt = (
        "質問: %s\n\n候補:\n%s\n\n"
        "最も関連する候補番号を JSON {\"rank\": [番号,...]} で返してください。"
        % (query["query"], listing)
    )
    started = time.perf_counter()
    try:
        reply = cloud_client.chat([
            {"role": "system", "content": "あなたは検索結果の関連度を判定するアシスタントです。JSON のみを返してください。"},
            {"role": "user", "content": prompt},
        ])
        record["rerank_reply"] = mask_secrets(reply[:500])
        record["rerank_candidates"] = [src for _, src, _ in candidates]
        try:
            ids = _extract_rank_ids(reply)
            ordered = list(dict.fromkeys(
                record["rerank_candidates"][i - 1]
                for i in ids
                if 1 <= i <= len(record["rerank_candidates"])
            ))
            rel = set(query["relevant_sources"])
            dcg = sum(
                1.0 / math.log2(i + 2)
                for i, src in enumerate(ordered[:RERANK_K])
                if src in rel
            )
            # ideal はファイル単位で一意な候補と正解の積集合で算出する。
            available = len(set(record["rerank_candidates"]) & rel)
            ideal = sum(
                1.0 / math.log2(i + 2)
                for i in range(min(available, RERANK_K))
            )
            if ideal:
                record["rerank_ndcg_at_5"] = round(dcg / ideal, 4)
        except (ValueError, TypeError) as exc:
            errors.append("rerank rank parse: %s" % mask_secrets(str(exc)))

        answer_prompt = "質問: %s\n\n参考文書:\n%s\n\n参考文書のみに基づき短く回答してください。" % (
            query["query"],
            "\n".join(chunk[:500] for _, _, chunk in candidates),
        )
        record["answer"] = mask_secrets(
            cloud_client.chat([
                {"role": "system", "content": "あなたは日本語の質問に答えるアシスタントです。"},
                {"role": "user", "content": answer_prompt},
            ])[:1000]
        )
    except Exception as exc:
        errors.append(mask_secrets(str(exc))[:300])
    if errors:
        record["generation_error"] = "; ".join(errors)[:300]
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=None, help="コーパスディレクトリ（既定 corpus-subset）")
    parser.add_argument("--qrels", default=None, help="qrels ファイル（既定 qrels-subset.json）")
    parser.add_argument("--output", default=None, help="出力 JSON（既定 eval-results-cloud-subset.json）")
    parser.add_argument("--with-generation", action="store_true", help="生成モデルで rerank / 回答を計測")
    parser.add_argument("--repeat", type=int, default=1, help="埋め込み検索の計測回数（既定1、指標とレイテンシを平均）")
    args = parser.parse_args()

    if not cloud_client.enabled():
        print("RAG_CLOUD_ENABLED=1 を指定してください", file=sys.stderr)
        return 2
    if args.repeat < 1:
        print("--repeat は1以上を指定してください", file=sys.stderr)
        return 2
    corpus_dir = Path(args.corpus).resolve() if args.corpus else CORPUS
    qrels_path = Path(args.qrels).resolve() if args.qrels else QRELS
    output_path = Path(args.output).resolve() if args.output else OUTPUT
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    files = sorted(corpus_dir.glob("*.md"))
    passages: list[tuple[str, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for chunk in _chunk(text):
            if chunk.strip():
                passages.append((path.name, chunk))
    print(f"[cloud] files={len(files)} chunks={len(passages)} model={cloud_client.embed_model()} repeat={args.repeat}")

    queries = qrels["heldout_queries"]
    passage_texts = [p for _, p in passages]
    query_texts = [q["query"] for q in queries]

    # 埋め込み検索を repeat 回実行し、指標・レイテンシを保存する。
    runs: list[dict] = []
    for i in range(args.repeat):
        started = time.perf_counter()
        p_vecs = cloud_client.embed(passage_texts)
        q_vecs = cloud_client.embed(query_texts)
        embedding_sec = time.perf_counter() - started

        started = time.perf_counter()
        scored_all: list[list] = []
        ranked_all: list[list[tuple[str, float]]] = []
        metrics_all: list[dict] = []
        for q, qv in zip(queries, q_vecs):
            scored = sorted(
                (
                    (_cosine(qv, pv), src, chunk)
                    for pv, (src, chunk) in zip(p_vecs, passages)
                ),
                reverse=True,
            )
            # ファイル単位の best score を採用
            best: dict[str, float] = {}
            for score, src, _ in scored:
                best[src] = max(best.get(src, 0.0), score)
            ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
            relevant = set(q["relevant_sources"])
            scored_all.append(scored)
            ranked_all.append(ranked)
            metrics_all.append(_flatten_metrics(_embedding_metrics(ranked, relevant)))
        search_sec = time.perf_counter() - started
        runs.append({
            "embedding_sec": embedding_sec,
            "search_sec": search_sec,
            "scored": scored_all,
            "ranked": ranked_all,
            "metrics": metrics_all,
        })
        print(
            "[cloud] run %d/%d embedding=%.3fs search=%.3fs"
            % (i + 1, args.repeat, embedding_sec, search_sec)
        )

    records = []
    for qi, q in enumerate(queries):
        flats = [run["metrics"][qi] for run in runs]
        avg = {k: round(sum(v[k] for v in flats) / len(flats), 4) for k in flats[0]}
        ranked0 = runs[0]["ranked"][qi]
        set_recall_at = {k: avg["set_recall_at_%s" % k] for k in map(str, RECALL_K)}
        records.append({
            "id": q["id"],
            "top_source": ranked0[0][0] if ranked0 else None,
            "relevant_sources": sorted(set(q["relevant_sources"])),
            "hit_at": {k: avg["hit_at_%s" % k] for k in map(str, RECALL_K)},
            "set_recall_at": set_recall_at,
            "recall_at": dict(set_recall_at),  # 互換キー（旧集合 recall）
            "ndcg_at_10": avg["ndcg_at_10"],
        })

    # 生成モデルはコスト配慮で先頭3クエリのみ（埋め込み repeat とは独立に1回）。
    generation_latencies: list[float] = []
    if args.with_generation:
        for qi, record in enumerate(records[:GENERATION_QUERY_LIMIT]):
            candidates = runs[0]["scored"][qi][:RERANK_K]
            generation_latencies.append(_run_generation(record, queries[qi], candidates))

    denominator = len(records) or 1
    embed_avg = round(sum(r["embedding_sec"] for r in runs) / len(runs), 6)
    search_avg = round(sum(r["search_sec"] for r in runs) / len(runs), 6)
    rerank_scores = [
        r["rerank_ndcg_at_5"] for r in records if "rerank_ndcg_at_5" in r
    ]
    summary = {
        "queries": len(records),
        "repeat": args.repeat,
        "hit_at_1": round(sum(r["hit_at"]["1"] for r in records) / denominator, 4),
        "hit_at_10": round(sum(r["hit_at"]["10"] for r in records) / denominator, 4),
        "recall_at_1": round(sum(r["recall_at"]["1"] for r in records) / denominator, 4),
        "recall_at_10": round(sum(r["recall_at"]["10"] for r in records) / denominator, 4),
        "set_recall_at_1": round(sum(r["set_recall_at"]["1"] for r in records) / denominator, 4),
        "set_recall_at_10": round(sum(r["set_recall_at"]["10"] for r in records) / denominator, 4),
        "ndcg_at_10": round(sum(r["ndcg_at_10"] for r in records) / denominator, 4),
        "embedding_latency_sec": embed_avg,
        "search_latency_sec": search_avg,
        "total_search_latency_sec": round(embed_avg + search_avg, 6),
        "rerank_query_count": len(rerank_scores),
    }
    if generation_latencies:
        summary["generation_latency_sec"] = round(
            sum(generation_latencies) / len(generation_latencies), 6
        )
    if rerank_scores:
        summary["rerank_ndcg_at_5"] = round(
            sum(rerank_scores) / len(rerank_scores), 4
        )

    corpus_sha256 = _configured_corpus_sha256()
    notes = [
        "埋め込みは opencodeGO の OpenAI 互換 /embeddings API を使用。",
        "スコアはコサイン類似度、ファイル単位の best score でランク付け。",
        "nDCG@10 は上位10件のみ・ファイル単位1回で計上（evaluate.py と同一定義）。",
        "hit_at は evaluate.py の query-level hit@k と同一定義。",
        "set_recall_at は旧 recall_at（集合 recall）の別名で、互換のため recall_at にも同一値を残す。",
        "レイテンシは time.perf_counter で計測し repeat 回の平均を summary に記録。",
        "秘密情報の値は出力・保存しない（生成テキストは mask_secrets を通過）。",
    ]
    if args.with_generation:
        notes.extend([
            "生成モデルは opencodeGO の OpenAI 互換 chat completions を使用（先頭3クエリのみ、rerank_query_count を参照）。",
            "rerank は JSON の rank 整数配列のみを採用し、パース失敗時は generation_error に記録して nDCG を 0 としない。",
        ])
    if not corpus_sha256:
        notes.append(
            "corpus_sha256 は設定（%s / %s）が無いため省略。"
            % (CORPUS_SHA256_ENV, CORPUS_SHA256_CONFIG.name)
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": cloud_client.embed_model(),
        "generation_model": cloud_client.model(),
        "corpus": corpus_dir.name,
        "corpus_file_count": len(files),
        "chunk_count": len(passages),
        "qrels_label": qrels_path.name,
        "chunk_size": 300,
        "repeat": args.repeat,
        "latency_runs": [
            {
                "embedding_sec": round(r["embedding_sec"], 6),
                "search_sec": round(r["search_sec"], 6),
            }
            for r in runs
        ],
        "summary": summary,
        "results": records,
        "notes": notes,
    }
    if corpus_sha256:
        payload["corpus_sha256"] = corpus_sha256

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[cloud] summary:", summary)
    print("[cloud] wrote", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
