#!/usr/bin/env python3
"""opencodeGO 埋め込みでコーパス-subset の検索品質を計測する。

使い方:
  # dsh.env の OPENCODE_GO_API_KEY を環境変数へ読み込んで実行
  set -a; source ~/.config/opencode/dsh.env; set +a
  RAG_CLOUD_ENABLED=1 python3 rag_poc/cloud_bench.py

結果は eval-results-cloud-subset.json へ保存する（退避対象）。
秘密情報の値は出力・保存しない。
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cloud_client

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus-subset"
QRELS = HERE / "qrels-subset.json"
OUTPUT = HERE / "eval-results-cloud-subset.json"


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


def main() -> int:
    if not cloud_client.enabled():
        print("RAG_CLOUD_ENABLED=1 を指定してください", file=sys.stderr)
        return 2
    qrels = json.loads(QRELS.read_text(encoding="utf-8"))
    files = sorted(CORPUS.glob("*.md"))
    passages: list[tuple[str, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for chunk in _chunk(text):
            if chunk.strip():
                passages.append((path.name, chunk))
    print(f"[cloud] files={len(files)} chunks={len(passages)} model={cloud_client.embed_model()}")

    p_vecs = cloud_client.embed([p for _, p in passages])
    queries = qrels["heldout_queries"]
    q_vecs = cloud_client.embed([q["query"] for q in queries])

    results = []
    for q, qv in zip(queries, q_vecs):
        scored = sorted(
            ((_cosine(qv, pv), src) for pv, (src, _) in zip(p_vecs, passages)),
            reverse=True,
        )
        # ファイル単位の best score を採用
        best: dict[str, float] = {}
        for score, src in scored:
            best[src] = max(best.get(src, 0.0), score)
        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
        relevant = set(q["relevant_sources"])
        hits = [src in relevant for src, _ in ranked]
        recall_at = {str(k): round(sum(hits[:k]) / max(1, len(relevant)), 4) for k in (1, 3, 5, 10)}
        dcg = sum((1 / math.log2(i + 2)) for i, hit in enumerate(hits) if hit)
        ideal = sum((1 / math.log2(i + 2)) for i in range(min(len(relevant), len(hits))))
        results.append({
            "id": q["id"],
            "top_source": ranked[0][0] if ranked else None,
            "relevant_sources": sorted(relevant),
            "recall_at": recall_at,
            "ndcg_at_10": round(dcg / ideal, 4) if ideal else 0.0,
        })

    summary = {
        "recall_at_1": round(sum(r["recall_at"]["1"] for r in results) / len(results), 4),
        "recall_at_10": round(sum(r["recall_at"]["10"] for r in results) / len(results), 4),
        "ndcg_at_10": round(sum(r["ndcg_at_10"] for r in results) / len(results), 4),
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": cloud_client.embed_model(),
        "corpus": "corpus-subset",
        "qrels_label": "qrels-subset.json",
        "chunk_size": 300,
        "summary": summary,
        "results": results,
        "notes": [
            "埋め込みは opencodeGO の OpenAI 互換 /embeddings API を使用。",
            "スコアはコサイン類似度、ファイル単位の best score でランク付け。",
            "秘密情報の値は出力・保存していない。",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[cloud] summary:", summary)
    print("[cloud] wrote", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
