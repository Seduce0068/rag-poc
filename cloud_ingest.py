#!/usr/bin/env python3
"""opencodeGO の埋め込みベクトルを Qdrant へ投入する。

使い方:
  set -a; source ~/.config/opencode/dsh.env; set +a
  RAG_CLOUD_ENABLED=1 RAG_COLLECTION=rag_poc_cloud_full \
    python3 rag_poc/cloud_ingest.py

ingest.py のチャンク分割・point ID・アップサートを再利用し、埋め込み呼び出し
だけを cloud_client.embed に差し替える。チェックポイントは使用しない
（クラウド投入は都度フル投入）。認証情報の値は出力・保存しない。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cloud_client
import ingest
from config import PAYLOAD_TEXT_KEY

HERE = Path(__file__).resolve().parent


def main() -> int:
    if not cloud_client.enabled():
        print("RAG_CLOUD_ENABLED=1 を指定してください", file=sys.stderr)
        return 2
    corpus = ingest.resolve_corpus_dir()
    files = sorted(ingest.iter_source_files(corpus))
    root = ingest._project_root()
    points: list[dict] = []
    for path in files:
        points.extend(ingest.iter_chunks(path, root))
    if not points:
        print("チャンクが生成されませんでした", file=sys.stderr)
        return 1
    # text-embedding-3-small の既定次元は 1536（dimensions 指定なし）。
    size = 1536
    print(
        "[cloud-ingest] files=%d chunks=%d model=%s dim=%d"
        % (len(files), len(points), cloud_client.embed_model(), size),
        flush=True,
    )

    texts = [point["payload"][PAYLOAD_TEXT_KEY] for point in points]
    vectors = cloud_client.embed(texts)
    for point, vector in zip(points, vectors):
        point["vector"] = vector

    session = ingest._make_session()
    created = ingest.ensure_collection(session, size)
    if created:
        print("[cloud-ingest] collection created", flush=True)
    count = ingest.upsert_points(session, points, completed_batches=0)
    print("[cloud-ingest] upserted=%d/%d" % (count, len(points)), flush=True)
    return 0 if count == len(points) else 1


if __name__ == "__main__":
    raise SystemExit(main())
