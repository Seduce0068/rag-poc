#!/usr/bin/env python3
"""opencodeGO 埋め込みでコーパス-subset の検索品質を計測する。

使い方:
  # dsh.env の OPENCODE_GO_API_KEY を環境変数へ読み込んで実行
  set -a; source ~/.config/opencode/dsh.env; set +a
  RAG_CLOUD_ENABLED=1 python3 rag_poc/cloud_bench.py
  RAG_CLOUD_ENABLED=1 python3 rag_poc/cloud_bench.py --repeat 3 --with-generation
  RAG_CLOUD_ENABLED=1 python3 rag_poc/cloud_bench.py --with-generation --generation-model deepseek-v4.1-flash
   RAG_CLOUD_ENABLED=1 python3 rag_poc/cloud_bench.py --with-generation --generation-max-queries 3  # コスト制限時のみ先頭3件
  RAG_CLOUD_ENABLED=1 python3 rag_poc/cloud_bench.py --with-generation --strict-rank  # 範囲外/重複 rank を generation_error へ昇格
  RAG_CLOUD_ENABLED=1 python3 rag_poc/cloud_bench.py --with-generation --rerank-file-dedupe  # 同一ファイルの best score チャンクのみ rerank
  RAG_CLOUD_ENABLED=1 python3 rag_poc/cloud_bench.py --list-models  # /v1/models から利用可能 ID を取得

起動時に config.validate_settings() を呼ぶ。RAG_CLOUD_ENABLED=1 では
RAG_CLOUD_EMBED_MODEL の明示指定が必要で、暗黙既定モデルは fail-closed とする
（cloud_ingest.py と同じ経路）。上の例では RAG_CLOUD_EMBED_MODEL と整合する
RAG_VECTOR_SIZE も合わせて指定してください。

結果は eval-results-cloud-subset.json へ保存する（退避対象）。
秘密情報の値は出力・保存しない。

チャンク分割は ingest.iter_chunks に委譲する。生成リクエストのセッション UUID は
cloud_client.chat 側で付与し、ここでは生成しない。cloud_client.last_usage() が
利用可能なら呼び出し回数・トークン使用量・cost を payload.usage へ集計し、
summary.estimated_cost_usd に課金見込みを記録する（--with-generation 指定時は
生成対象件数と見込みを stderr へ警告する）。
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cloud_client
import config
import ingest
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
# 生成対象クエリ数は --generation-max-queries で制御する（既定 0＝全件）。

# 埋め込み単価（USD / 1M 入力トークン）。README の単価表を cloud_bench 内にも保持し、
# API が cost を返さない場合の estimated_cost_usd 推定に使う。
# 出典: README「クラウドモデル（opencodeGO）」の単価表
#       （https://opencode.ai/docs/console/go/、2026-11-26 取得）。
EMBED_PRICE_USD_PER_MILLION: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "qwen3-embedding-8b": 0.01,
    "gemini-embedding-2": 0.20,
}
PRICE_SOURCE_NOTE = "README の単価表（opencode.ai/docs/console/go/）"
PRICE_SOURCE_DATE = "2026-11-26"

# 現行の指標定義。旧 ndcg_at_10=0.9034 は全順位ベースで算出した値であり、
# 現行定義（上位10件・ファイル1回）とは直接比較できない。
METRIC_DEFINITION: dict[str, object] = {
    "ndcg_at_10": "上位10件のみを採用し、同一ファイルは最上位の1回だけ計上（gain=1 / discount=log2(rank+1)、IDCG は relevant_sources 数）。evaluate.py 準拠。",
    "hit_at": "query-level hit@k（上位 k に正解が1件でもあれば1）。evaluate.py 準拠。",
    "set_recall_at": "集合 recall（上位 k の正解数 / relevant_sources 数）。recall_at は互換キーとして同一値。",
    "rerank_ndcg_at_5": "生成モデルが返した rank の上位5件、qrels の relevant_sources を母集団。",
    "legacy_ndcg_full_ranking": True,
    "legacy_ndcg_at_10": "旧値 0.9034 は全順位ベース（現行定義と直接比較不可）。",
}

USAGE_KEYS: tuple[str, ...] = (
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost",
)
USAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "calls": ("calls", "call_count", "request_count", "requests", "num_calls"),
    "prompt_tokens": ("prompt_tokens", "input_tokens", "promptTokens", "inputTokens"),
    "completion_tokens": (
        "completion_tokens",
        "output_tokens",
        "completionTokens",
        "outputTokens",
    ),
    "total_tokens": ("total_tokens", "totalTokens"),
    # cloud_client._accumulate_usage() は nested な cost を "cost.usd" へ平坦化する。
    "cost": ("cost", "cost_usd", "cost.usd"),
}


def _usage_int(value: object) -> int | None:
    """使用量の数値を非負 int へ正規化する（bool と不正値は無効）。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        # NaN / ±inf は int() で例外または不正値になるため先に弾く。
        if isinstance(value, float) and not math.isfinite(value):
            return None
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def _usage_cost(value: object) -> float | None:
    """cost を非負の有限 float へ正規化する（bool・文字列・不正値は無効）。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _normalize_usage(raw: object) -> dict[str, int | float] | None:
    """last_usage() の戻り値を既知キーへ正規化する（解釈不能なら None）。"""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        merged: dict[str, int | float] = {}
        for item in raw:
            part = _normalize_usage(item)
            if not part:
                continue
            for key, value in part.items():
                merged[key] = merged.get(key, 0) + value
        if not merged:
            return None
    else:
        if isinstance(raw, dict):
            sources: list[dict[str, object]] = [raw]
            nested = raw.get("usage")
            if isinstance(nested, dict):
                sources.append(nested)
        else:
            try:
                attributes = vars(raw)
            except TypeError:
                return None
            sources = [attributes] if isinstance(attributes, dict) else []
        merged = {}
        for source in sources:
            for key, aliases in USAGE_ALIASES.items():
                if key in merged:
                    continue
                for alias in aliases:
                    raw_value = source.get(alias)
                    if key == "cost" and isinstance(raw_value, dict):
                        # 入れ子の {"cost": {"usd": ...}} も解釈する。
                        raw_value = raw_value.get("usd")
                    # cost は小数（例 0.01）を保持するため int 正規化を使わない。
                    number = (
                        _usage_cost(raw_value)
                        if key == "cost"
                        else _usage_int(raw_value)
                    )
                    if number is not None:
                        merged[key] = number
                        break
        if not merged:
            # 入れ子辞書（例: {"embedding": {...}, "chat": {...}}）も合算する。
            for value in (sources[0].values() if sources else ()):
                part = _normalize_usage(value)
                if not part:
                    continue
                for key, number in part.items():
                    merged[key] = merged.get(key, 0) + number
        if not merged:
            return None
    if "total_tokens" not in merged:
        prompt = merged.get("prompt_tokens")
        completion = merged.get("completion_tokens")
        if prompt is not None or completion is not None:
            merged["total_tokens"] = (prompt or 0) + (completion or 0)
    return merged


def _warn(message: object) -> None:
    """マスク済みの警告を stderr へ出す（cloud_ingest.py と同じ方針）。"""
    print("[cloud] %s" % mask_secrets(message), file=sys.stderr, flush=True)


def _estimated_cost_usd(
    usage_payload: dict[str, int | float] | None, embed_model: str
) -> float | None:
    """API の cost を優先し、無ければ単価 × total_tokens で推定する。

    単価は埋め込みモデルごとの 1M 入力トークン単価（README の単価表）。
    API が cost を返す場合はその実値を採用する。キー値は扱わない。
    """
    if not usage_payload:
        return None
    api_cost = usage_payload.get("cost")
    if api_cost is not None:
        value = _usage_cost(api_cost)
        if value is not None:
            return round(value, 6)
    price = EMBED_PRICE_USD_PER_MILLION.get(embed_model)
    total_tokens = usage_payload.get("total_tokens")
    if price is None or total_tokens is None:
        return None
    tokens = _usage_cost(total_tokens)
    if tokens is None:
        return None
    return round(price * tokens / 1_000_000, 6)


def _usage_cost_source(
    usage_payload: dict[str, int | float] | None, embed_model: str
) -> str | None:
    """usage のコスト出所を 'api' / 'estimated' / None で返す。

    API が cost を返していれば実値（'api'）を優先する。cost が無く、既知の
    埋め込み単価と total_tokens があれば単価推定（'estimated'）。どちらも
    無ければ None（判定不能として usage に None を残す）。
    """
    if not usage_payload:
        return None
    if _usage_cost(usage_payload.get("cost")) is not None:
        return "api"
    if (
        _usage_cost(usage_payload.get("total_tokens")) is not None
        and embed_model in EMBED_PRICE_USD_PER_MILLION
    ):
        return "estimated"
    return None


def _token_cost_usd(total_tokens: object, embed_model: str) -> float | None:
    """単価 × total_tokens の推定値を返す（単価不明なら None）。

    summary.estimated_cost_usd は 6 桁丸めのため、トークン数が少ないと 0.0 に
    潰れる。その場合でもトークン由来の見込みを残すため、ここでは 12 桁で丸める。
    """
    price = EMBED_PRICE_USD_PER_MILLION.get(embed_model)
    tokens = _usage_cost(total_tokens)
    if price is None or tokens is None:
        return None
    return round(price * tokens / 1_000_000, 12)


def _warn_generation_billing(
    count: int,
    total: int,
    generation_model: str,
    embed_model: str,
    usage_payload: dict[str, int | float] | None,
) -> None:
    """生成対象件数と課金見込みをマスク済み stderr で警告する。

    既定（--generation-max-queries=0）でも「全件＝N 件」を明示する。
    """
    price = EMBED_PRICE_USD_PER_MILLION.get(embed_model)
    if price is not None:
        price_note = "埋め込み単価 %s: $%.2f / 1M tokens（出典日 %s）" % (
            embed_model, price, PRICE_SOURCE_DATE,
        )
    else:
        price_note = "埋め込み単価: 未知モデルのため推定不可"
    estimate = _estimated_cost_usd(usage_payload, embed_model)
    if estimate is not None:
        billing = "現在の推定コスト $%.6f（API cost があれば実値を優先）" % estimate
    else:
        billing = "usage が無いため推定コストは算出不可"
    _warn(
        "--with-generation: 生成対象 %d 件 / 全 %d 件（chat 呼び出し最大 %d 回、"
        "generation_model=%s）。%s。外部送信＋トークン課金が発生します。"
        "課金見込み: %s。最終値は summary.estimated_cost_usd に記録します。"
        % (count, total, count * 2, generation_model, price_note, billing)
    )


class _UsageCollector:
    """cloud_client.last_usage() のスナップショットを任意で集約する。

    last_usage() が無い／失敗／未知形式なら payload.usage を作らず notes に残す。
    per-call 形式は合計、累積形式は最終値（最大値）を採用する。
    """

    def __init__(self) -> None:
        getter = getattr(cloud_client, "last_usage", None)
        self._getter = getter if callable(getter) else None
        self._samples: list[dict[str, int | float]] = []
        self._notes: list[str] = []
        if self._getter is None:
            self._note("cloud_client.last_usage() が未提供のため usage 集計をスキップ")

    @property
    def notes(self) -> list[str]:
        return list(self._notes)

    def _note(self, text: str) -> None:
        message = mask_secrets("usage: " + text)
        if message not in self._notes:
            self._notes.append(message)

    def sample(self) -> None:
        """各クラウド呼び出し直後にスナップショットを記録する。"""
        if self._getter is None:
            return
        try:
            raw = self._getter()
        except Exception as exc:  # 集計失敗で本体の計測は止めない
            self._note("last_usage() の呼び出しに失敗: %s" % exc)
            return
        normalized = _normalize_usage(raw)
        if normalized is None:
            if raw is not None:
                self._note("last_usage() の戻り値を解釈できず集計をスキップ")
            return
        if normalized:
            self._samples.append(normalized)

    def payload(self) -> dict[str, int | float] | None:
        """集計結果を返す（可能なデータが無ければ None）。"""
        if not self._samples:
            return None
        calls = [s["calls"] for s in self._samples if "calls" in s]
        cumulative = (
            len(calls) == len(self._samples)
            and len(calls) > 1
            and all(a <= b for a, b in zip(calls, calls[1:]))
            and calls[-1] > calls[0]
        )
        result: dict[str, int | float] = {}
        for key in USAGE_KEYS:
            values = [s[key] for s in self._samples if key in s]
            if values:
                result[key] = max(values) if cumulative else sum(values)
        if "calls" not in result:
            result["calls"] = len(self._samples)
        return result


def _chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    *,
    isolated: bool = False,
) -> str:
    """生成モデルを実効化して cloud_client.chat を呼ぶ（UUID は委譲）。

    ``isolated=True`` が使えるなら cloud_client.chat(isolated=True) へ渡し、
    呼び出し単位の失敗を CloudUnavailable へ閉じ込める（ad-query が追加した
    API）。未対応の古い cloud_client では引数を渡さず、呼び出し側の try/except
    に委譲する。既存の ``_chat(messages, model)`` 呼び出しはそのまま成立する。
    """
    try:
        params = inspect.signature(cloud_client.chat).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs: dict[str, object] = {}
    if isolated and "isolated" in params:
        kwargs["isolated"] = True
    if not model:
        return cloud_client.chat(messages, **kwargs)
    if "model" in params:
        kwargs["model"] = model
        return cloud_client.chat(messages, **kwargs)
    # 現行 chat は cloud_client.model()（RAG_CLOUD_MODEL）を参照するため、
    # 呼び出し中だけ実効モデルへ差し替え、finally で必ず復元する。
    env_name = str(getattr(cloud_client, "ENV_MODEL", "RAG_CLOUD_MODEL"))
    previous = os.environ.get(env_name)
    os.environ[env_name] = model
    try:
        return cloud_client.chat(messages, **kwargs)
    finally:
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous


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

    範囲外・重複 rank の判定は候補数を知る呼び出し側の責務とする（API は
    従来どおり 1 引数・list[int] 返しを維持）。``_rank_diagnostics()`` で
    黙って捨てずに診断を残すこと。
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


def _rank_diagnostics(ids: list[int], candidate_count: int) -> list[str]:
    """範囲外 rank と重複 rank を診断メッセージへ変換する（スコアは変更しない）。

    ``_run_generation`` は従来どおり範囲外を除外し重複を一意化して nDCG を
    算出するが、その際に本関数で診断を残す（黙って除外しない）。rank 番号は
    秘密情報ではないが、名称だけを扱う既存方針に合わせて ``mask_secrets`` を通す。
    """
    warnings: list[str] = []
    out_of_range = sorted({i for i in ids if not 1 <= i <= candidate_count})
    if out_of_range:
        warnings.append(
            mask_secrets("rank out of range: %s" % ", ".join(str(i) for i in out_of_range))
        )
    seen: set[int] = set()
    duplicates: list[int] = []
    for i in ids:
        if i in seen and i not in duplicates:
            duplicates.append(i)
        seen.add(i)
    if duplicates:
        warnings.append(
            mask_secrets("duplicate rank: %s" % ", ".join(str(i) for i in duplicates))
        )
    return warnings


def _dedupe_candidates_by_file(candidates: list) -> list:
    """同一ファイルの best score チャンクのみを残す（スコア降順を維持）。

    ``candidates`` は (score, source, chunk) のリストを想定する。ファイルごとに
    最大 score のエントリを1つだけ採用し、スコアの高い順へ並べ直す。既定の
    挙動を変えず、``--rerank-file-dedupe`` 指定時のみ呼び出す。
    """
    best: dict[str, tuple] = {}
    for score, src, chunk in candidates:
        current = best.get(src)
        if current is None or score > current[0]:
            best[src] = (score, src, chunk)
    return sorted(best.values(), key=lambda item: item[0], reverse=True)


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


def _run_generation(
    record: dict,
    query: dict,
    candidates: list,
    *,
    generation_model: str | None = None,
    usage: _UsageCollector | None = None,
    strict_rank: bool = False,
) -> float:
    """1 クエリ分の rerank + 回答生成を実行し、レイテンシ秒を返す。

    保存する reply / answer / error は必ず config.mask_secrets を通す。
    rerank と answer は別 try/except で実行する。rerank が失敗しても answer を
    続行し、answer が失敗しても算出済みの rerank_ndcg_at_5 は保持する。
    パース失敗時は generation_error に記録し、nDCG キーは作成しない（0 にしない）。

    範囲外・重複 rank は既定では ``rerank_warnings`` に記録して既存のスコア
    挙動（nDCG=1.0）を維持する。``strict_rank=True`` では generation_error
    へ昇格してスコア算出を見送る（呼び出し側のサニタイズ方針で選択可能）。
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

    # --- rerank フェーズ: 失敗しても answer を続行する。 ---
    try:
        reply = _chat([
            {"role": "system", "content": "あなたは検索結果の関連度を判定するアシスタントです。JSON のみを返してください。"},
            {"role": "user", "content": prompt},
        ], generation_model, isolated=True)
        if usage is not None:
            usage.sample()
        record["rerank_reply"] = mask_secrets(reply[:500])
        record["rerank_candidates"] = [src for _, src, _ in candidates]
        try:
            ids = _extract_rank_ids(reply)
            warnings = _rank_diagnostics(ids, len(record["rerank_candidates"]))
            if warnings:
                record["rerank_warnings"] = warnings
            if strict_rank and warnings:
                errors.extend(warnings)
            else:
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
                # ideal は qrels の relevant_sources 全体（最大 RERANK_K 件）を
                # 母集団に算出する。候補内に正解が無い場合は dcg=0 となり、
                # 検索 nDCG と比較可能な定義へ揃える（旧来の「候補内の正解数」は
                # 使わない）。
                ideal = sum(
                    1.0 / math.log2(i + 2)
                    for i in range(min(len(rel), RERANK_K))
                )
                if ideal:
                    record["rerank_ndcg_at_5"] = round(dcg / ideal, 4)
        except (ValueError, TypeError) as exc:
            errors.append("rerank rank parse: %s" % mask_secrets(str(exc)))
    except Exception as exc:
        errors.append("rerank: %s" % mask_secrets(str(exc))[:300])

    # --- answer フェーズ: rerank の成否に依存せず実行する。 ---
    try:
        answer_prompt = "質問: %s\n\n参考文書:\n%s\n\n参考文書のみに基づき短く回答してください。" % (
            query["query"],
            "\n".join(chunk[:500] for _, _, chunk in candidates),
        )
        answer = _chat([
            {"role": "system", "content": "あなたは日本語の質問に答えるアシスタントです。"},
            {"role": "user", "content": answer_prompt},
        ], generation_model, isolated=True)
        if usage is not None:
            usage.sample()
        record["answer"] = mask_secrets(answer[:1000])
    except Exception as exc:
        errors.append("answer: %s" % mask_secrets(str(exc))[:300])

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
    parser.add_argument(
        "--generation-model",
        default=None,
        help="生成モデル ID（既定 cloud_client.model()）",
    )
    parser.add_argument(
        "--generation-max-queries",
        type=int,
        default=0,
        help="生成モデルで計測するクエリ数（既定0＝全件、N指定で先頭N件に制限）",
    )
    parser.add_argument(
        "--strict-rank",
        action="store_true",
        help="rerank の範囲外・重複 rank を generation_error へ昇格しスコア算出を見送る（既定は警告のみでスコア算出）",
    )
    parser.add_argument(
        "--rerank-file-dedupe",
        action="store_true",
        help="rerank 前に候補をファイル単位で一意化し同一ファイルの best score チャンクのみを残す（既定は現行挙動）",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="cloud_client.fetch_models() で利用可能なモデル ID を表示して終了（既定は通信しない）",
    )
    args = parser.parse_args()

    # --list-models は起動時検証より前に処理する。明示指定時だけ /v1/models を
    # 呼び、結果の ID を stdout へ1行ずつ出す。失敗はマスク済み stderr と終了
    # コード 1（既定の計測経路は従来どおり通信しない）。
    if args.list_models:
        try:
            model_ids = cloud_client.fetch_models()
            for model_id in model_ids:
                print(model_id, flush=True)
        except Exception as exc:
            _warn(f"モデル一覧の取得に失敗しました: {exc}")
            return 1
        return 0

    # 起動時検証（cloud_ingest.py と同じ経路）。RAG_CLOUD_EMBED_MODEL 未指定の
    # 暗黙既定モデルは fail-closed で終了コード 2 とする。--help は argparse が
    # 先に処理するため、ここには到達しない。
    problems = config.validate_settings()
    if problems:
        for problem in problems:
            _warn("設定検証: %s" % problem)
        return 2

    if not cloud_client.enabled():
        print("RAG_CLOUD_ENABLED=1 を指定してください", file=sys.stderr)
        return 2
    if args.repeat < 1:
        print("--repeat は1以上を指定してください", file=sys.stderr)
        return 2
    if args.generation_max_queries < 0:
        print("--generation-max-queries は0以上を指定してください", file=sys.stderr)
        return 2
    generation_model = (
        args.generation_model.strip()
        if args.generation_model and args.generation_model.strip()
        else cloud_client.model()
    )
    embed_model = cloud_client.embed_model()
    usage = _UsageCollector()
    corpus_dir = Path(args.corpus).resolve() if args.corpus else CORPUS
    qrels_path = Path(args.qrels).resolve() if args.qrels else QRELS
    output_path = Path(args.output).resolve() if args.output else OUTPUT
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    files = sorted(corpus_dir.glob("*.md"))
    passages: list[tuple[str, str]] = []
    for path in files:
        # ingest.iter_chunks に委譲する。root に corpus_dir を渡すことで
        # payload.source はファイル名になり、既存の qrels 正解ラベルと一致する。
        for point in ingest.iter_chunks(path, corpus_dir):
            payload_chunk = point.get("payload", {})
            source = str(payload_chunk.get(ingest.PAYLOAD_SOURCE_KEY, path.name))
            text = str(payload_chunk.get(ingest.PAYLOAD_TEXT_KEY, ""))
            if text.strip():
                passages.append((source, text))
    print(
        "[cloud] files=%d chunks=%d model=%s generation_model=%s repeat=%d"
        % (len(files), len(passages), embed_model, generation_model, args.repeat)
    )

    queries = qrels["heldout_queries"]
    passage_texts = [p for _, p in passages]
    query_texts = [q["query"] for q in queries]

    # 埋め込み検索を repeat 回実行し、指標・レイテンシを保存する。
    runs: list[dict] = []
    for i in range(args.repeat):
        started = time.perf_counter()
        p_vecs = cloud_client.embed(passage_texts)
        usage.sample()
        q_vecs = cloud_client.embed(query_texts)
        usage.sample()
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

    # 生成対象は既定で全クエリ。--generation-max-queries N 指定時は
    # コスト配慮で先頭 N 件のみ（埋め込み repeat とは独立に1回）。
    # 0（既定）は「制限なし＝全件」を表す。
    generation_query_count = (
        len(queries)
        if args.generation_max_queries == 0
        else min(args.generation_max_queries, len(queries))
    )
    generation_latencies: list[float] = []
    if args.with_generation:
        # 既定（--generation-max-queries=0＝全件）でも件数を明示し、課金見込みを警告する。
        _warn_generation_billing(
            generation_query_count, len(queries), generation_model, embed_model,
            usage.payload(),
        )
        for qi, record in enumerate(records[:generation_query_count]):
            candidates = runs[0]["scored"][qi]
            if args.rerank_file_dedupe:
                # ファイル単位で best score チャンクのみを残してから上位 K 件を選択する。
                candidates = _dedupe_candidates_by_file(candidates)
            candidates = candidates[:RERANK_K]
            generation_latencies.append(
                _run_generation(
                    record,
                    queries[qi],
                    candidates,
                    generation_model=generation_model,
                    usage=usage,
                    strict_rank=args.strict_rank,
                )
            )

    usage_payload = usage.payload()
    denominator = len(records) or 1
    embed_avg = round(sum(r["embedding_sec"] for r in runs) / len(runs), 6)
    search_avg = round(sum(r["search_sec"] for r in runs) / len(runs), 6)
    # 実効的に生成を実行したクエリ（既定は全件、--generation-max-queries 指定時は
    # 先頭 N 件）のうち nDCG が算出できたものを集約する。パース失敗などで
    # キーが無いクエリは除外し、0 として扱わない（record 側の規約を維持）。
    rerank_scores = [
        r["rerank_ndcg_at_5"] for r in records if "rerank_ndcg_at_5" in r
    ]
    # 警告は record 側の rerank_warnings に埋もれないよう summary にも集約する。
    # warning_count は警告メッセージ総数、invalid_rank_count は警告が出たクエリ数。
    rerank_warning_count = sum(
        len(r.get("rerank_warnings", [])) for r in records
    )
    rerank_invalid_rank_count = sum(
        1 for r in records if r.get("rerank_warnings")
    )
    estimated_cost = _estimated_cost_usd(usage_payload, embed_model)
    cost_source = _usage_cost_source(usage_payload, embed_model)
    total_tokens = (
        _usage_cost(usage_payload.get("total_tokens")) if usage_payload else None
    )
    # 6 桁丸めで 0.0 に潰れた場合でもトークン由来の見込みを残す。
    token_cost_applicable = (
        estimated_cost == 0.0 and total_tokens is not None and total_tokens > 0
    )
    token_cost = (
        _token_cost_usd(total_tokens, embed_model) if token_cost_applicable else None
    )
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
        "rerank_query_count": len(rerank_scores),  # 実効クエリ数（スコア換算できた件数）
        "rerank_warning_count": rerank_warning_count,
        "rerank_invalid_rank_count": rerank_invalid_rank_count,
        "estimated_cost_usd": estimated_cost,
    }
    if token_cost_applicable:
        summary["estimated_cost_from_tokens_usd"] = token_cost
    if generation_latencies:
        summary["generation_latency_sec"] = round(
            sum(generation_latencies) / len(generation_latencies), 6
        )
    if rerank_scores:
        # 実効クエリ数（rerank_query_count）を分母にした平均。
        summary["rerank_ndcg_at_5"] = round(
            sum(rerank_scores) / len(rerank_scores), 4
        )

    corpus_sha256 = _configured_corpus_sha256()
    notes = [
        "埋め込みは opencodeGO の OpenAI 互換 /embeddings API を使用。",
        "スコアはコサイン類似度、ファイル単位の best score でランク付け。",
        "チャンク分割は ingest.iter_chunks に委譲（chunk_size は ingest.CHUNK_SIZE の実効値）。",
        "指標定義: set_recall_at は集合 recall、hit_at は evaluate.py と同じ query-level hit@k、ndcg_at_10 は上位10件ベース（ファイル単位1回で計上）。",
        "metric_definition は現行定義（ndcg_at_10 は上位10件・ファイル1回、hit_at は query-level、set_recall_at は集合 recall）を payload に記録する。",
        "旧 ndcg_at_10=0.9034 は全順位ベースで算出した値です。現行の ndcg_at_10 は上位10件・ファイル1回の定義であり、evaluate.py の同名指標と直接比較できません（比較表では legacy_ndcg_full_ranking=true を参照）。",
        "埋め込み単価は %s を採用（単価出典日: %s）。API が cost を返さない場合は単価 × total_tokens で summary.estimated_cost_usd を推定する。"
        % (PRICE_SOURCE_NOTE, PRICE_SOURCE_DATE),
        "互換のため recall_at には set_recall_at（集合 recall）と同一値を残す。",
        "generation_model は --generation-model の指定値を記録し、未指定なら cloud_client.model() を実効値とする。",
        "レイテンシは time.perf_counter で計測し repeat 回の平均を summary に記録。",
        "秘密情報の値は出力・保存しない（生成テキストは mask_secrets を通過）。",
    ]
    if args.with_generation:
        notes.extend([
            "生成モデルは opencodeGO の OpenAI 互換 chat completions を使用"
            "（生成対象 %d 件、rerank_query_count を参照）。" % generation_query_count,
            "生成呼び出しは外部送信＋トークン課金を伴う。total_tokens には生成呼び出しのトークンも含まれるため estimated_cost_usd は概算（API cost があれば実値を優先）。",
            "生成リクエストのセッション UUID は cloud_client.chat 側で付与し cloud_bench では生成しない。",
            "rerank は JSON の rank 整数配列のみを採用し、パース失敗時は generation_error に記録して nDCG を 0 としない。",
            "範囲外 rank / 重複 rank は rerank_warnings に記録してスコア算出は従来どおり（--strict-rank 指定時は generation_error へ昇格しスコアを見送る）。",
        ])
    if args.rerank_file_dedupe:
        notes.append(
            "rerank 候補は --rerank-file-dedupe によりファイル単位で一意化し、"
            "同一ファイルの best score チャンクのみを残す（スコア降順で上位 %d 件を再選択）。"
            % RERANK_K
        )
    notes.extend(usage.notes)
    if not corpus_sha256:
        notes.append(
            "corpus_sha256 は設定（%s / %s）が無いため省略。"
            % (CORPUS_SHA256_ENV, CORPUS_SHA256_CONFIG.name)
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": embed_model,
        "generation_model": generation_model,
        "corpus": corpus_dir.name,
        "corpus_file_count": len(files),
        "chunk_count": len(passages),
        "qrels_label": qrels_path.name,
        "chunk_size": ingest.CHUNK_SIZE,
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
        "metric_definition": METRIC_DEFINITION,
        "legacy_ndcg_full_ranking": True,
        "notes": notes,
    }
    if corpus_sha256:
        payload["corpus_sha256"] = corpus_sha256
    if usage_payload is not None:
        # API cost / 単価推定 / 判定不能(None) を cost_source で明示する。
        usage_payload["cost_source"] = cost_source
        if token_cost_applicable:
            usage_payload["estimated_cost_from_tokens_usd"] = token_cost
        payload["usage"] = usage_payload

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[cloud] summary:", summary)
    print("[cloud] wrote", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
