"""rag_poc の単体テスト。外部サービスへは接続しない。

直接実行:
    python3 rag_poc/test_rag.py

注: ディレクトリ名にハイフンがあるため、
    python3 -m unittest rag_poc/test_rag.py はモジュール名に変換できず失敗する。
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import statistics
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# requests が未導入でもモジュール import だけは通す（通信は行わない）。
if "requests" not in sys.modules:
    try:
        import requests  # noqa: F401
    except ImportError:
        _stub = types.ModuleType("requests")
        _stub.post = lambda *a, **k: None  # type: ignore[attr-defined]
        sys.modules["requests"] = _stub

import cloud_bench
import cloud_client
import cloud_ingest
import compare_cloud_local
import config
import evaluate
import ingest
import ollama_client
import query

# Rust 同等性テストで skip された理由を集計する（固定メッセージのみ、秘密情報は入れない）。
# RAG_REQUIRE_RUST=1 の最終ゲートで skip を失敗に変換するために使う。
_RUST_SKIP_REASONS: list[str] = []


@contextlib.contextmanager
def _temporary_env(**values):
    """環境変数を一時的に差し替える（None は削除）。実通信・実書き込みなし。"""
    with mock.patch.dict(os.environ, {}, clear=False):
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)
        yield


class ConfigTest(unittest.TestCase):
    """config の定数が読み込めること。"""

    NAMES = (
        "OLLAMA_URL",
        "EMBED_MODEL",
        "CHAT_MODEL",
        "QDRANT_URL",
        "COLLECTION",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K",
        "RERANK_MAX_DOCS",
        "VECTOR_SIZE",
        "CORPUS_DIR",
    )
    INT_NAMES = {"CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "RERANK_MAX_DOCS", "VECTOR_SIZE"}

    def test_constants_exist_with_expected_types(self):
        for name in self.NAMES:
            with self.subTest(name=name):
                self.assertTrue(hasattr(config, name), name)
                value = getattr(config, name)
                if name in self.INT_NAMES:
                    self.assertIsInstance(value, int, name)
                else:
                    self.assertIsInstance(value, str, name)
                    self.assertTrue(value, name)

    def test_chunk_settings_are_consistent(self):
        self.assertGreater(config.CHUNK_SIZE, 0)
        self.assertGreaterEqual(config.CHUNK_OVERLAP, 0)
        self.assertLess(config.CHUNK_OVERLAP, config.CHUNK_SIZE)


class RrfFuseTest(unittest.TestCase):
    """rrf_fuse が RRF(k=60) で順位を統合すること。"""

    def test_shared_items_rank_above_single_list_items(self):
        dense = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        keyword = [{"id": "c"}, {"id": "d"}, {"id": "a"}]

        fused = query.rrf_fuse(dense, keyword, top_k=10)
        ids = [hit["id"] for hit in fused]

        self.assertEqual(set(ids), {"a", "b", "c", "d"})
        self.assertLess(ids.index("a"), ids.index("b"))
        self.assertLess(ids.index("a"), ids.index("d"))
        self.assertLess(ids.index("c"), ids.index("b"))
        self.assertLess(ids.index("c"), ids.index("d"))

    def test_scores_follow_k60_formula(self):
        first = [{"id": "x"}]
        second = [{"id": "y"}, {"id": "x"}]

        fused = query.rrf_fuse(first, second, top_k=5)
        scores = {hit["id"]: hit["score"] for hit in fused}

        self.assertAlmostEqual(scores["x"], 1.0 / 61 + 1.0 / 62)
        self.assertAlmostEqual(scores["y"], 1.0 / 61)

    def test_top_k_truncates(self):
        ranking = [{"id": index} for index in range(5)]

        fused = query.rrf_fuse(ranking, top_k=2)

        self.assertEqual(len(fused), 2)


class ExtractJsonTest(unittest.TestCase):
    """_extract_json が素の JSON とコードフェンスを解析すること。"""

    def test_plain_json(self):
        self.assertEqual(
            query._extract_json('{"scores": [0.1, 0.9]}'),
            {"scores": [0.1, 0.9]},
        )

    def test_fenced_json(self):
        text = '```json\n{"scores": [0.2, 0.8]}\n```'
        self.assertEqual(
            query._extract_json(text), {"scores": [0.2, 0.8]}
        )

    def test_json_surrounded_by_text(self):
        text = '結果:\n[{"index":0,"score":0.5}]\n以上'
        self.assertEqual(
            query._extract_json(text), [{"index": 0, "score": 0.5}]
        )

    def test_non_string_returns_none(self):
        self.assertIsNone(query._extract_json(None))
        self.assertIsNone(query._extract_json("not json at all"))


class IterChunksTest(unittest.TestCase):
    """iter_chunks が見出しを保持しサイズ分割すること。"""

    def test_heading_kept_and_body_within_chunk_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample.md"
            heading = "# 見出しテスト"
            body = "あ" * (config.CHUNK_SIZE * 2 + 10)
            path.write_text(f"{heading}\n{body}\n", encoding="utf-8")

            chunks = list(ingest.iter_chunks(path, root))

        self.assertGreater(len(chunks), 1)
        for index, chunk in enumerate(chunks):
            text = chunk["payload"]["text"]
            with self.subTest(index=index):
                self.assertTrue(text.startswith(heading))
                self.assertEqual(chunk["payload"]["chunk_index"], index)
                # 見出し行を除いた本文が CHUNK_SIZE を超えないこと。
                _, _, body_part = text.partition("\n")
                self.assertLessEqual(len(body_part), config.CHUNK_SIZE)

    def test_each_section_keeps_its_heading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sections.md"
            path.write_text(
                "# 一方\n" + "い" * 40 + "\n## 二方\n" + "ろ" * 40 + "\n",
                encoding="utf-8",
            )

            chunks = list(ingest.iter_chunks(path, root))

        texts = [chunk["payload"]["text"] for chunk in chunks]
        self.assertTrue(any(text.startswith("# 一方") for text in texts))
        self.assertTrue(any(text.startswith("## 二方") for text in texts))
        self.assertEqual(chunks[0]["payload"]["title"], "一方")


class EvaluateQueriesTest(unittest.TestCase):
    """評価クエリが 6 件で expected を持つこと。"""

    def test_six_queries_with_expected(self):
        cases = evaluate.EVAL_QUERIES
        self.assertEqual(len(cases), 6)
        seen_ids = set()
        for case in cases:
            with self.subTest(case=case.get("id")):
                self.assertIn("id", case)
                self.assertIn("query", case)
                self.assertIn("expected", case)
                self.assertIsInstance(case["query"], str)
                self.assertTrue(case["query"].strip())
                self.assertIsInstance(case["expected"], list)
                self.assertTrue(case["expected"])
                self.assertTrue(
                    all(
                        isinstance(keyword, str) and keyword
                        for keyword in case["expected"]
                    )
                )
                seen_ids.add(case["id"])
        self.assertEqual(len(seen_ids), 6)


class ParseScoresTest(unittest.TestCase):
    """_parse_scores の入力揺れに対する挙動。"""

    def test_partial_scores_keep_none(self) -> None:
        scores = query._parse_scores({"scores": [{"index": 0, "score": 0.9}]}, 3)
        self.assertEqual(scores[0], 0.9)
        self.assertIsNone(scores[1])
        self.assertIsNone(scores[2])

    def test_one_based_index_is_normalized(self) -> None:
        data = {"scores": [{"index": 1, "score": 0.2}, {"index": 2, "score": 0.8}]}
        scores = query._parse_scores(data, 2)
        self.assertEqual(scores, [0.2, 0.8])

    def test_string_score_is_ignored(self) -> None:
        scores = query._parse_scores({"scores": [{"index": 0, "score": "high"}]}, 2)
        self.assertIsNone(scores[0])

    def test_nan_and_inf_in_score_list_are_not_top_relevance(self) -> None:
        # NaN / inf を min(1.0, ...) で丸めると最高関連度 1.0 に化ける。
        # 非有限値は 1.0 であってはならない（0.0 等に明示されること）。
        scores = query._parse_scores(
            [float("nan"), 0.5, float("inf"), float("-inf")], 4
        )
        self.assertIsNotNone(scores)
        self.assertNotEqual(scores[0], 1.0)
        self.assertNotEqual(scores[2], 1.0)
        self.assertEqual(scores[1], 0.5)

    def test_nan_and_inf_in_score_dict_are_not_top_relevance(self) -> None:
        data = {
            "scores": [
                {"index": 0, "score": float("nan")},
                {"index": 1, "score": float("inf")},
            ]
        }
        scores = query._parse_scores(data, 2)
        self.assertIsNotNone(scores)
        self.assertNotEqual(scores[0], 1.0)
        self.assertNotEqual(scores[1], 1.0)


class HeldOutMetricsTest(unittest.TestCase):
    """held-out 指標を evaluate._evaluate_heldout_one 経由で検証する。

    evaluate.py には単独の _ndcg_at / _recall_at は存在しないため、
    実際に指標を計算している _evaluate_heldout_one をテスト対象とする。
    """

    @staticmethod
    def _record(sources, relevant, retrieval_k=3):
        def fake_search(query, mode, top_k):
            return [{"source": source, "text": ""} for source in sources]

        case = {
            "id": "h-test",
            "query": "テスト質問",
            "relevant_sources": relevant,
        }
        return evaluate._evaluate_heldout_one(
            fake_search, case, "hybrid", retrieval_k
        )

    def test_ndcg_at3_is_one_when_relevant_is_rank1(self):
        record = self._record(["target.md"], ["target.md"])

        self.assertAlmostEqual(record["ndcg_at"]["3"], 1.0, places=6)
        self.assertAlmostEqual(record["ndcg_at"]["10"], 1.0, places=6)

    def test_ndcg_at3_matches_log_discount_at_rank3(self):
        record = self._record(
            ["a.md", "b.md", "target.md"], ["target.md"]
        )

        self.assertAlmostEqual(
            record["ndcg_at"]["3"], 1.0 / math.log2(4), places=5
        )

    def test_ndcg_is_zero_without_relevant_source(self):
        record = self._record(["a.md", "b.md"], ["target.md"])

        self.assertEqual(record["ndcg_at"]["3"], 0.0)
        self.assertEqual(record["ndcg_at"]["10"], 0.0)

    def test_recall_at3_is_one_only_with_a_hit(self):
        hit = self._record(
            ["a.md", "target.md", "c.md"], ["target.md"]
        )
        miss = self._record(
            ["a.md", "b.md", "c.md"], ["target.md"]
        )

        self.assertEqual(float(hit["recall_at"]["3"]), 1.0)
        self.assertEqual(float(miss["recall_at"]["3"]), 0.0)
        self.assertEqual(float(hit["recall_at"]["1"]), 0.0)

    def test_heldout_queries_have_nine_labelled_cases(self):
        cases = evaluate.HELD_OUT_QUERIES
        self.assertEqual(len(cases), 9)
        self.assertEqual(evaluate.HELD_OUT_RECALL_K, (1, 3, 5, 10))
        self.assertEqual(evaluate.HELD_OUT_NDCG_K, (3, 10))

        seen_ids = set()
        for case in cases:
            with self.subTest(case=case.get("id")):
                for key in ("id", "query", "relevant_sources"):
                    self.assertIn(key, case)
                self.assertIsInstance(case["id"], str)
                self.assertTrue(case["id"].strip())
                self.assertIsInstance(case["query"], str)
                self.assertTrue(case["query"].strip())
                relevant = case["relevant_sources"]
                self.assertIsInstance(relevant, list)
                self.assertTrue(relevant)
                self.assertTrue(
                    all(
                        isinstance(name, str) and name.strip()
                        for name in relevant
                    )
                )
                seen_ids.add(case["id"])
        self.assertEqual(len(seen_ids), 9)


class RerankTest(unittest.TestCase):
    """rerank が上位 RERANK_MAX_DOCS 件だけを LLM 評価すること。"""

    def setUp(self):
        self._previous_max_docs = query.RERANK_MAX_DOCS
        query.RERANK_MAX_DOCS = 5

    def tearDown(self):
        query.RERANK_MAX_DOCS = self._previous_max_docs

    @staticmethod
    def _points(count=10):
        return [
            {
                "id": f"p{index}",
                "score": 1.0 - index / 10.0,
                "text": f"本文 {index}",
                "source": f"doc{index}.md",
            }
            for index in range(count)
        ]

    def test_first_five_are_reranked_and_all_ten_are_returned(self):
        content = json.dumps({"scores": [0.9, 0.8, 0.7, 0.6, 0.5]})

        def fake_post(url, payload):
            return {"message": {"content": content}}

        with mock.patch.object(query, "_post", fake_post):
            hits = query.rerank("質問", self._points())

        self.assertEqual(len(hits), 10)
        self.assertTrue(all(h["reranked_by_llm"] for h in hits[:5]))
        self.assertTrue(all(not h["reranked_by_llm"] for h in hits[5:]))
        self.assertEqual(
            [h["id"] for h in hits[5:]],
            [f"p{index}" for index in range(5, 10)],
        )
        self.assertTrue(all("base_score" in h for h in hits))

    def test_post_failure_keeps_original_order_and_scores(self):
        original = self._points()

        def failing_post(url, payload):
            raise RuntimeError("LLM 応答なし")

        with mock.patch.object(query, "_post", failing_post), \
                mock.patch.object(query, "_warn", lambda message: None):
            hits = query.rerank("質問", original)

        self.assertEqual(len(hits), 10)
        self.assertEqual(
            [hit["id"] for hit in hits],
            [point["id"] for point in original],
        )
        self.assertEqual(
            [hit["score"] for hit in hits],
            [point["score"] for point in original],
        )
        self.assertTrue(
            all(hit["reranked_by_llm"] is False for hit in hits)
        )
        self.assertTrue(
            all(
                hit["base_score"] == point["score"]
                for hit, point in zip(hits, original)
            )
        )

    def test_score_fields_are_attached_to_all_hits_and_missing_llm_score_is_zero(self):
        content = json.dumps(
            {
                "scores": [
                    {"index": 0, "score": 0.9},
                    {"index": 1, "score": 0.8},
                    {"index": 2, "score": 0.7},
                    {"index": 3, "score": 0.6},
                ]
            }
        )

        def fake_post(url, payload):
            return {"message": {"content": content}}

        with mock.patch.object(query, "_post", fake_post), \
                mock.patch.object(query, "_warn", lambda message: None):
            hits = query.rerank("質問", self._points())

        required = {"base_score", "llm_score", "final_score", "reranked_by_llm"}
        self.assertTrue(all(required.issubset(hit) for hit in hits))
        evaluated = [hit for hit in hits if hit["reranked_by_llm"]]
        self.assertEqual(len(evaluated), 5)
        missing = [hit for hit in evaluated if hit["llm_score"] is None]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["final_score"], 0.0)
        self.assertGreater(missing[0]["base_score"], 0.0)
        self.assertTrue(all(hit["final_score"] == hit["score"] for hit in hits))


class ConfigValidationTest(unittest.TestCase):
    """validate_settings が不正設定を検出すること。"""

    VALID = {
        "CHUNK_SIZE": 800,
        "CHUNK_OVERLAP": 100,
        "TOP_K": 10,
        "RERANK_MAX_DOCS": 5,
        "VECTOR_SIZE": 2048,
    }

    def _problems_with(self, **overrides):
        values = dict(self.VALID)
        values.update(overrides)
        with mock.patch.multiple(config, **values):
            return config.validate_settings()

    def test_overlap_ge_chunk_size_reports_problem(self):
        problems = self._problems_with(CHUNK_OVERLAP=800)
        self.assertTrue(any("CHUNK_OVERLAP" in text for text in problems), problems)

    def test_top_k_zero_reports_problem(self):
        problems = self._problems_with(TOP_K=0)
        self.assertTrue(any("RAG_TOP_K" in text for text in problems), problems)

    def test_vector_size_zero_reports_problem(self):
        problems = self._problems_with(VECTOR_SIZE=0)
        self.assertTrue(any("RAG_VECTOR_SIZE" in text for text in problems), problems)

    def test_valid_settings_returns_empty(self):
        self.assertEqual(self._problems_with(), [])


class EndpointValidationTest(unittest.TestCase):
    """config.validate_endpoint のオプトイン／許可リスト判定を直接検証する。"""

    @staticmethod
    def _validate(url, *, env_remote=None, allowed_hosts="", allow_remote=None):
        """環境変数と config 定数を一時固定して validate_endpoint を呼ぶ。"""
        env = {
            "RAG_ALLOW_REMOTE_ENDPOINTS": "" if env_remote is None else env_remote,
            "RAG_ALLOWED_ENDPOINT_HOSTS": allowed_hosts,
        }
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(config, "RAG_ALLOWED_ENDPOINT_HOSTS", allowed_hosts):
            return config.validate_endpoint(
                url, "テスト Endpoint", allow_remote=allow_remote
            )

    def test_loopback_http_has_no_problems(self):
        for url in ("http://127.0.0.1:11434", "http://localhost:6333", "http://[::1]:6333"):
            with self.subTest(url=url):
                self.assertEqual(self._validate(url), [])

    def test_remote_http_is_fatal(self):
        problems = self._validate("http://example.test:11434")
        self.assertTrue(problems)
        self.assertTrue(any("ローカル限定" in text for text in problems))
        self.assertTrue(any("平文 HTTP" in text for text in problems))

    def test_remote_https_with_opt_in_only_is_still_fatal(self):
        problems = self._validate("https://example.test:11434", env_remote="1")
        self.assertTrue(problems)
        self.assertTrue(any("ローカル限定" in text for text in problems))
        self.assertFalse(any("平文 HTTP" in text for text in problems))

    def test_remote_https_with_allowlisted_host_is_downgraded(self):
        url = "https://example.test:11434"
        for allowed in ("example.test", " Example.Test "):
            with self.subTest(allowed=allowed):
                self.assertEqual(
                    self._validate(url, env_remote="1", allowed_hosts=allowed), []
                )

    def test_host_port_form_is_allowed(self):
        url = "https://example.test:11434"
        self.assertEqual(
            self._validate(url, env_remote="1", allowed_hosts="EXAMPLE.TEST:11434"), []
        )
        # host 単体の許可リストとは別に host:port 形式も照合されること。
        problems = self._validate(url, env_remote="1", allowed_hosts="example.test:9999")
        self.assertTrue(problems)

    def test_invalid_scheme_is_always_fatal(self):
        for allow_remote, env_remote in ((None, "1"), (True, None), ("1", None)):
            with self.subTest(allow_remote=allow_remote):
                problems = self._validate(
                    "ftp://example.test/file",
                    env_remote=env_remote,
                    allowed_hosts="example.test",
                    allow_remote=allow_remote,
                )
                self.assertEqual(len(problems), 1)
                self.assertTrue(any("http/https" in text for text in problems))

    def test_invalid_port_is_always_fatal(self):
        for url in ("http://127.0.0.1:99999", "https://example.test:70000"):
            with self.subTest(url=url):
                problems = self._validate(
                    url,
                    env_remote="1",
                    allowed_hosts="example.test",
                    allow_remote=True,
                )
                self.assertEqual(len(problems), 1)
                self.assertTrue(any("URL が不正" in text for text in problems))

    def test_allow_remote_argument_overrides_environment(self):
        url = "https://example.test:11434"
        # 環境がオプトイン済みでも明示 False なら fatal が残る。
        problems = self._validate(
            url, env_remote="1", allowed_hosts="example.test", allow_remote=False
        )
        self.assertTrue(problems)
        # 環境が未オプトインでも明示 "1" なら警告へ格下げされる。
        problems = self._validate(
            url, env_remote="", allowed_hosts="example.test", allow_remote="1"
        )
        self.assertEqual(problems, [])


class OllamaClientTest(unittest.TestCase):
    """ollama_client.embed を fake session で直接検証する（実 HTTP なし）。"""

    URL = "http://127.0.0.1:11434"

    @staticmethod
    def _response(data=None, status_code=200, text=""):
        response = mock.Mock()
        response.status_code = status_code
        response.text = text
        if isinstance(data, Exception):
            response.json.side_effect = data
        else:
            response.json.return_value = data
        return response

    @contextlib.contextmanager
    def _fake_session(self, responder):
        """_shared_session を差し替え、呼び出し記録付き fake を返す。"""
        session = mock.Mock()
        session.post.side_effect = responder
        with mock.patch.object(ollama_client, "_shared_session", return_value=session):
            yield session

    def test_embed_success_keeps_input_order(self):
        primary = self._response({"embeddings": [[0.1], [0.2], [0.3]]})

        def responder(url, json, timeout):
            return primary

        with self._fake_session(responder) as session:
            vectors = ollama_client.embed(
                ["a", "b", "c"], model="embed-m", url=self.URL, timeout=1.0
            )

        self.assertEqual(vectors, [[0.1], [0.2], [0.3]])
        self.assertEqual(session.post.call_count, 1)
        call = session.post.call_args_list[0]
        self.assertEqual(call.args[0], self.URL + "/api/embed")
        self.assertEqual(
            call.kwargs["json"], {"model": "embed-m", "input": ["a", "b", "c"]}
        )

    def test_embed_404_falls_back_once_per_prompt_in_order(self):
        legacy_map = {"first": [0.1], "second": [0.2], "third": [0.3]}

        def responder(url, json, timeout):
            if url.endswith("/api/embed"):
                return self._response(status_code=404, text="not found")
            return self._response({"embedding": legacy_map[json["prompt"]]})

        with self._fake_session(responder) as session:
            vectors = ollama_client.embed(
                ["first", "second", "third"], model="embed-m", url=self.URL
            )

        self.assertEqual(vectors, [[0.1], [0.2], [0.3]])
        self.assertEqual(session.post.call_count, 4)  # primary 1 + legacy 3
        legacy = [
            call for call in session.post.call_args_list
            if call.args[0].endswith("/api/embeddings")
        ]
        self.assertEqual(len(legacy), 3)
        self.assertEqual(
            [call.kwargs["json"]["prompt"] for call in legacy],
            ["first", "second", "third"],
        )
        self.assertTrue(all("input" not in call.kwargs["json"] for call in legacy))

    def test_both_apis_fail_raises_masked_composite_error(self):
        secret = "Bearer ZZZTOKENZZZ https://example.test:11434"

        def responder(url, json, timeout):
            if url.endswith("/api/embed"):
                return self._response(status_code=404, text=secret)
            raise RuntimeError(secret)

        with self._fake_session(responder):
            with self.assertRaises(RuntimeError) as ctx:
                ollama_client.embed(["a"], model="embed-m", url=self.URL)

        message = str(ctx.exception)
        self.assertIn("Ollama embed に失敗", message)
        self.assertIn("フォールバック /api/embeddings も失敗", message)
        for needle in ("ZZZTOKENZZZ", "https://example.test:11434"):
            with self.subTest(needle=needle):
                self.assertNotIn(needle, message)
        self.assertIn("<masked>", message)

    def test_validate_dim_mismatch_raises_without_fallback(self):
        primary = self._response({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

        def responder(url, json, timeout):
            if url.endswith("/api/embed"):
                return primary
            raise AssertionError("次元不一致のときフォールバックしてはならない")

        with self._fake_session(responder) as session:
            with self.assertRaisesRegex(RuntimeError, "埋め込み次元"):
                ollama_client.embed(
                    ["a", "b"], model="embed-m", url=self.URL, validate_dim=3
                )

        self.assertEqual(session.post.call_count, 1)
        self.assertTrue(session.post.call_args_list[0].args[0].endswith("/api/embed"))

    def test_flat_array_is_accepted_for_single_input(self):
        primary = self._response({"embeddings": [0.1, 0.2, 0.3]})

        with self._fake_session(lambda url, json, timeout: primary) as session:
            vectors = ollama_client.embed(["single"], model="embed-m", url=self.URL)

        self.assertEqual(vectors, [[0.1, 0.2, 0.3]])
        self.assertEqual(session.post.call_count, 1)

    def test_empty_input_makes_no_http_call(self):
        def forbidden():
            raise AssertionError("空入力で HTTP セッションを生成してはならない")

        with mock.patch.object(ollama_client, "_shared_session", new=forbidden):
            self.assertEqual(ollama_client.embed([]), [])
            self.assertEqual(ollama_client.embed(iter(())), [])

    def test_prefix_is_reflected_in_json_input(self):
        primary = self._response({"embeddings": [[0.1], [0.2]]})
        texts = ["a", "b"]

        with self._fake_session(lambda url, json, timeout: primary) as session:
            ollama_client.embed(
                texts, model="embed-m", url=self.URL, prefix="クエリ: "
            )

        payload = session.post.call_args_list[0].kwargs["json"]
        self.assertEqual(payload["input"], ["クエリ: a", "クエリ: b"])
        self.assertEqual(texts, ["a", "b"])  # 呼び出し側の入力は変更しない

    def test_non_finite_vectors_follow_vector_contract(self):
        # _vector が非有限値を弾く実装なら embed は失敗し、現行（弾かない）なら
        # そのまま返す。どちらになっても契約をテストで固定する。
        try:
            rejects = ollama_client._vector([float("nan")]) is None
        except Exception:
            rejects = True

        def responder(url, json, timeout):
            if url.endswith("/api/embed"):
                return self._response(
                    {"embeddings": [[float("nan"), 0.1], [0.2, float("inf")]]}
                )
            return self._response({"embedding": [float("nan"), float("inf")]})

        with self._fake_session(responder):
            if rejects:
                with self.assertRaises((RuntimeError, ValueError, TypeError)):
                    ollama_client.embed(["a", "b"], model="embed-m", url=self.URL)
            else:
                vectors = ollama_client.embed(
                    ["a", "b"], model="embed-m", url=self.URL
                )
                self.assertTrue(math.isnan(vectors[0][0]))
                self.assertTrue(math.isinf(vectors[1][1]))


class MaskSecretsTest(unittest.TestCase):
    """mask_secrets が各種秘密情報をマスクすること。"""

    def test_masks_url_bearer_and_api_key(self):
        text = (
            "https://example.test:8445/api?token=abc Bearer ZZZTOKENZZZ "
            "api_key=ZZZKEYZZZ http://127.0.0.1:6333/collections"
        )
        masked = config.mask_secrets(text)
        for needle in (
            "https://example.test:8445/api?token=abc",
            "Bearer ZZZTOKENZZZ",
            "api_key=ZZZKEYZZZ",
            "http://127.0.0.1:6333/collections",
        ):
            with self.subTest(needle=needle):
                self.assertNotIn(needle, masked)
        self.assertIn("<masked>", masked)

    def test_masks_ipv4_ipv6_and_path(self):
        text = "ipv4=192.0.2.44 ipv6=[2001:db8::1]:6333 path=/home/user/secret.md"
        masked = config.mask_secrets(text)
        for needle in ("192.0.2.44", "[2001:db8::1]:6333", "/home/user/secret.md"):
            with self.subTest(needle=needle):
                self.assertNotIn(needle, masked)
        self.assertIn("<masked>", masked)


class QdrantCollectionPathTest(unittest.TestCase):
    """qdrant_collection_path が URL 予約文字をエンコードすること。"""

    def test_encodes_special_characters(self):
        encoded = config.qdrant_collection_path("a/b?c# d")
        self.assertEqual(encoded, "a%2Fb%3Fc%23%20d")
        for char in ("/", "?", "#", " "):
            with self.subTest(char=char):
                self.assertNotIn(char, encoded)


class ResolveCorpusDirTest(unittest.TestCase):
    """resolve_corpus_dir が cwd に依存せず既存パスを解決すること。"""

    def test_absolute_path_is_resolved_as_is(self):
        with mock.patch.object(config, "CORPUS_DIR", str(HERE)):
            resolved = config.resolve_corpus_dir()
        self.assertEqual(resolved, HERE.resolve())
        self.assertTrue(resolved.is_absolute())

    def test_relative_path_uses_repo_root(self):
        with mock.patch.object(config, "CORPUS_DIR", "rag_poc"):
            resolved = config.resolve_corpus_dir()
        self.assertEqual(resolved, (config.REPO_ROOT / "rag_poc").resolve())


class PointIdTest(unittest.TestCase):
    """_point_id が 64bit 範囲で衝突しないこと。"""

    def test_ids_are_64bit_unique_and_stable(self):
        ids = [ingest._point_id("doc.md", index) for index in range(64)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(0 <= value < 2**64 for value in ids))
        self.assertEqual(ingest._point_id("doc.md", 0), ids[0])
        self.assertNotEqual(
            ingest._point_id("doc.md", 0), ingest._point_id("other.md", 0)
        )


class MakeSessionTest(unittest.TestCase):
    """_make_session の Retry が POST を含むこと（外部通信なし）。"""

    def test_retry_allows_post(self):
        captured: dict[str, object] = {}

        class FakeRetry:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        class FakeAdapter:
            def __init__(self, max_retries=None):
                self.max_retries = max_retries

        class FakeSession:
            def __init__(self):
                self.mounts = {}

            def mount(self, prefix, adapter):
                self.mounts[prefix] = adapter

        adapters_module = types.ModuleType("requests.adapters")
        adapters_module.HTTPAdapter = FakeAdapter  # type: ignore[attr-defined]
        retry_module = types.ModuleType("urllib3.util.retry")
        retry_module.Retry = FakeRetry  # type: ignore[attr-defined]
        util_module = types.ModuleType("urllib3.util")
        util_module.retry = retry_module  # type: ignore[attr-defined]
        urllib3_module = types.ModuleType("urllib3")
        urllib3_module.util = util_module  # type: ignore[attr-defined]

        fake_requests = types.SimpleNamespace(Session=FakeSession)
        with mock.patch.object(ingest, "requests", fake_requests), \
                mock.patch.dict(
                    sys.modules,
                    {
                        "requests.adapters": adapters_module,
                        "urllib3": urllib3_module,
                        "urllib3.util": util_module,
                        "urllib3.util.retry": retry_module,
                    },
                ):
            session = ingest._make_session()

        methods = captured.get("allowed_methods")
        self.assertIsNotNone(methods)
        self.assertIn("POST", methods)
        self.assertIn("GET", methods)
        self.assertEqual(set(session.mounts), {"http://", "https://"})


class LegacyIdsTest(unittest.TestCase):
    """_check_legacy_ids が旧 CRC32 ID を検出すること。"""

    @staticmethod
    def _session_with_ids(ids):
        response = mock.Mock(status_code=200)
        response.json.return_value = {
            "result": {"points": [{"id": value} for value in ids]}
        }
        session = mock.Mock()
        session.post.return_value = response
        return session

    def test_legacy_crc32_ids_raise(self):
        session = self._session_with_ids([12345, 67890])
        with self.assertRaisesRegex(RuntimeError, "旧 CRC32 ID"):
            ingest._check_legacy_ids(session)
        session.post.assert_called_once()

    def test_64bit_ids_do_not_raise(self):
        session = self._session_with_ids([2**33, 2**40])
        ingest._check_legacy_ids(session)
        session.post.assert_called_once()


class CheckpointTest(unittest.TestCase):
    """_load_checkpoint が embed_model 不一致を無効とすること。"""

    @staticmethod
    def _points():
        return [{"id": 1, "payload": {config.PAYLOAD_TEXT_KEY: "本文"}}]

    @classmethod
    def _write_state(cls, path, embed_model, completed=2):
        points = cls._points()
        state = {
            "collection": ingest.COLLECTION,
            "vector_size": ingest.VECTOR_SIZE,
            "embed_model": embed_model,
            "signature": ingest._checkpoint_signature(points),
            "completed_batches": completed,
        }
        path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        return points

    def test_embed_model_mismatch_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".ingest-checkpoint.json"
            points = self._write_state(path, "another-model")
            with mock.patch.object(ingest, "CHECKPOINT_PATH", path), \
                    mock.patch.dict(os.environ, {"RAG_INGEST_RESUME": "1"}):
                self.assertEqual(ingest._load_checkpoint(points), 0)

    def test_matching_checkpoint_returns_completed_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".ingest-checkpoint.json"
            points = self._write_state(path, ingest.EMBED_MODEL, completed=1)
            with mock.patch.object(ingest, "CHECKPOINT_PATH", path), \
                    mock.patch.dict(os.environ, {"RAG_INGEST_RESUME": "1"}):
                self.assertEqual(ingest._load_checkpoint(points), 1)


class ZeroBasedScoresTest(unittest.TestCase):
    """_parse_scores が 0 混在 index を 0-based として扱うこと。"""

    def test_zero_mixed_index_is_not_shifted(self):
        data = {
            "scores": [
                {"index": 0, "score": 0.1},
                {"index": 2, "score": 0.3},
            ]
        }
        scores = query._parse_scores(data, 3)
        self.assertEqual(scores[0], 0.1)
        self.assertIsNone(scores[1])
        self.assertEqual(scores[2], 0.3)

    def test_full_zero_based_indices_are_kept(self):
        data = {
            "scores": [
                {"index": 0, "score": 0.1},
                {"index": 1, "score": 0.5},
                {"index": 2, "score": 0.9},
            ]
        }
        self.assertEqual(query._parse_scores(data, 3), [0.1, 0.5, 0.9])


class MaskErrorMessageTest(unittest.TestCase):
    """_mask_error_message が Bearer をマスクすること。"""

    def test_bearer_token_is_masked(self):
        masked = query._mask_error_message("LLM 失敗: Bearer ZZZSECRETZZZ")
        self.assertNotIn("ZZZSECRETZZZ", masked)
        self.assertIn("<masked>", masked)


class AggregateLatencyTest(unittest.TestCase):
    """_aggregate_latency が samples から統計を再計算すること。"""

    def test_recomputes_avg_median_and_stdev_from_samples(self):
        rows = [
            {"latency_stats": {"count": 2, "samples": [0.1, 0.3]}},
            {"latency_stats": {"count": 1, "samples": [0.5]}},
        ]
        summary = evaluate._aggregate_latency(rows)
        samples = [0.1, 0.3, 0.5]
        self.assertEqual(summary["count"], 3)
        self.assertAlmostEqual(summary["avg"], round(sum(samples) / 3, 6))
        self.assertAlmostEqual(summary["median"], round(statistics.median(samples), 6))
        self.assertAlmostEqual(summary["stdev"], round(statistics.pstdev(samples), 6))
        self.assertAlmostEqual(summary["min"], 0.1)
        self.assertAlmostEqual(summary["max"], 0.5)


class LegacyLatencyFallbackTest(unittest.TestCase):
    """samples を欠く旧形式 record のフォールバック集計を固定する。"""

    @staticmethod
    def _legacy_stats(count, avg, median, minimum, maximum, stdev):
        return {
            "count": count,
            "avg": avg,
            "median": median,
            "min": minimum,
            "max": maximum,
            "stdev": stdev,
        }

    def test_fallback_uses_record_statistics_when_samples_are_absent(self):
        rows = [
            {"latency_stats": self._legacy_stats(2, 0.1, 0.2, 0.05, 0.6, 0.02)},
            {"latency_stats": self._legacy_stats(3, 0.3, 0.4, 0.05, 0.6, 0.04)},
            # count=0 の行と latency_stats 欠落行は集計対象外。
            {"latency_stats": self._legacy_stats(0, 9.9, 9.9, 9.9, 9.9, 9.9)},
            {"mode": "hybrid"},
        ]

        summary = evaluate._aggregate_latency(rows)

        # count は有効 record の合計（2 + 3 = 5）。
        self.assertEqual(summary["count"], 5)
        # それ以外は record 側の統計値から算出する。avg / stdev は平均、
        # median は record median の中央値、min / max は record 値の最小 /
        # 最大。この fixture ではいずれも record 値の平均と一致する。
        self.assertAlmostEqual(summary["avg"], 0.2, places=6)
        self.assertAlmostEqual(summary["median"], 0.3, places=6)
        self.assertAlmostEqual(summary["min"], 0.05, places=6)
        self.assertAlmostEqual(summary["max"], 0.6, places=6)
        self.assertAlmostEqual(summary["stdev"], 0.03, places=6)
        self.assertNotIn("samples", summary)


class ReuseSimpleResultsTest(unittest.TestCase):
    """_reuse_simple_results が schema_version 欠落を再利用しないこと。"""

    @staticmethod
    def _payload(with_schema):
        payload = {
            "results": [{"mode": "hybrid"}],
            "summary": {"hybrid": {}},
            "collection": "rag_poc_docs",
            "corpus_sha256": "deadbeef",
            "corpus_file_count": 7,
            "embed_model": "embed-m",
            "chat_model": "chat-m",
            "vector_size": 768,
            "top_k": 10,
            "qrels_label": "qrels.json",
            "qrels_schema_version": 1,
        }
        if with_schema:
            payload["schema_version"] = 2
        return payload

    @staticmethod
    def _patches():
        return (
            mock.patch.object(evaluate, "_load_collection", return_value="rag_poc_docs"),
            mock.patch.object(
                evaluate,
                "_eval_metadata",
                return_value={
                    "corpus_sha256": "deadbeef",
                    "corpus_file_count": 7,
                    "embed_model": "embed-m",
                    "chat_model": "chat-m",
                    "vector_size": 768,
                },
            ),
            mock.patch.object(evaluate, "_load_top_k", return_value=10),
        )

    def test_missing_schema_version_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval-results.json"
            path.write_text(
                json.dumps(self._payload(with_schema=False)), encoding="utf-8"
            )
            p1, p2, p3 = self._patches()
            with mock.patch.object(evaluate, "OUTPUT_PATH", str(path)), p1, p2, p3:
                records, notes = evaluate._reuse_simple_results()
        self.assertIsNone(records)
        self.assertTrue(any("schema_version" in note for note in notes))

    def test_matching_metadata_reuses_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval-results.json"
            path.write_text(json.dumps(self._payload(with_schema=True)), encoding="utf-8")
            p1, p2, p3 = self._patches()
            with mock.patch.object(evaluate, "OUTPUT_PATH", str(path)), p1, p2, p3:
                records, notes = evaluate._reuse_simple_results()
        self.assertEqual(records, [{"mode": "hybrid"}])
        self.assertTrue(notes)


class CallSearchTest(unittest.TestCase):
    """_call_search が warmup + 計測回数だけ呼び、全エラーを集約すること。"""

    def test_repeat_two_runs_warmup_plus_two_measurements(self):
        state = {"calls": 0}

        def fake_func(question, mode, top_k):
            state["calls"] += 1
            return [{"source": "doc.md"}]

        def errors_getter():
            return [
                {
                    "stage": "round",
                    "mode": None,
                    "type": "Error",
                    "message": "e%d" % state["calls"],
                }
            ]

        raw, description, stats, errors = evaluate._call_search(
            fake_func, "質問", "hybrid", 3, repeat=2, errors_getter=errors_getter
        )

        self.assertEqual(state["calls"], 3)
        self.assertEqual(raw, [{"source": "doc.md"}])
        self.assertEqual(stats["count"], 2)
        self.assertEqual(len(stats["samples"]), 2)
        self.assertEqual([error["message"] for error in errors], ["e1", "e2", "e3"])
        self.assertIn("mode='hybrid'", description)


class EmbeddingPrefixTest(unittest.TestCase):
    """日本語特化埋め込みの prefix と次元テーブル。"""

    def test_ruri_prefixes_and_dimension(self):
        self.assertEqual(
            config.embedding_prefixes("kun432/cl-nagoya-ruri-base"), ("クエリ: ", "文章: ")
        )
        self.assertEqual(config.expected_vector_size("kun432/cl-nagoya-ruri-base"), 768)

    def test_existing_model_has_no_prefix(self):
        self.assertEqual(config.embedding_prefixes("qwen2.5:3b"), ("", ""))
        self.assertEqual(config.expected_vector_size("qwen2.5:3b"), 2048)

    def test_ingest_passage_prefix_uses_namedtuple_attribute(self):
        # NamedTuple 化後も ingest 側で passage prefix が失われないこと。
        with mock.patch.object(ingest, "EMBED_MODEL", "kun432/cl-nagoya-ruri-base"):
            self.assertEqual(ingest._passage_prefix(), "文章: ")
        with mock.patch.object(ingest, "EMBED_MODEL", "qwen2.5:3b"):
            self.assertEqual(ingest._passage_prefix(), "")

    def test_ruri_requires_768_dimension(self):
        with mock.patch.object(config, "EMBED_MODEL", "kun432/cl-nagoya-ruri-base"):
            with mock.patch.object(config, "VECTOR_SIZE", 2048):
                problems = config.validate_settings()
        self.assertTrue(any("RAG_VECTOR_SIZE" in problem for problem in problems))

    def test_query_prefix_only_on_embed_input(self):
        embed_calls = []
        chat_calls = []

        def fake_embed(texts, **kwargs):
            embed_calls.append((list(texts), kwargs))
            return [[0.1, 0.2, 0.3] for _ in texts]

        def fake_post(url, payload):
            chat_calls.append((url, payload))
            return {"message": {"content": "ok"}}

        with mock.patch.object(query, "_ollama_embed", fake_embed):
            with mock.patch.object(query, "_post", fake_post):
                with mock.patch.object(query, "EMBED_MODEL", "kun432/cl-nagoya-ruri-base"):
                    vector = query.embed_query("パスキー sudo の使い方")
                    query.generate_answer("パスキー sudo の使い方", [{"source": "s", "text": "t"}])
        self.assertEqual(vector, [0.1, 0.2, 0.3])
        self.assertTrue(embed_calls)
        self.assertEqual(embed_calls[0][1].get("prefix"), "クエリ: ")
        chat_payload = [p for u, p in chat_calls if u.endswith("/api/chat")][0]
        self.assertNotIn("クエリ: ", chat_payload["messages"][1]["content"])


class EmbeddingPrefixNamedTupleTest(unittest.TestCase):
    """embedding_prefixes() の NamedTuple 契約と 2 タプル互換を固定する。"""

    def test_known_model_exposes_query_and_passage_attributes(self):
        prefixes = config.embedding_prefixes("kun432/cl-nagoya-ruri-base")

        self.assertTrue(
            hasattr(prefixes, "query"),
            "config.py の NamedTuple 化が未反映の可能性があります",
        )
        self.assertTrue(
            hasattr(prefixes, "passage"),
            "config.py の NamedTuple 化が未反映の可能性があります",
        )
        self.assertEqual(prefixes.query, "クエリ: ")
        self.assertEqual(prefixes.passage, "文章: ")
        self.assertIsInstance(prefixes, tuple)
        self.assertEqual(tuple(prefixes), ("クエリ: ", "文章: "))
        query_prefix, passage_prefix = prefixes
        self.assertEqual((query_prefix, passage_prefix), ("クエリ: ", "文章: "))

    def test_unknown_model_keeps_empty_named_tuple_and_unpacking(self):
        prefixes = config.embedding_prefixes("qwen2.5:3b")

        self.assertTrue(
            hasattr(prefixes, "query"),
            "config.py の NamedTuple 化が未反映の可能性があります",
        )
        self.assertTrue(
            hasattr(prefixes, "passage"),
            "config.py の NamedTuple 化が未反映の可能性があります",
        )
        self.assertEqual(prefixes.query, "")
        self.assertEqual(prefixes.passage, "")
        self.assertEqual(list(prefixes), ["", ""])
        query_prefix, passage_prefix = prefixes
        self.assertEqual((query_prefix, passage_prefix), ("", ""))


class _RustSkipCounter:
    """Rust 同等性テストの skip を集計する mixin。

    ``run()`` 経由で実際に実行されたテストの skip だけを数える。
    ``RustSkipGateTest`` が意図的に送出する SkipTest は ``run()`` を通らない
    ため集計せず、通常実行の skip 数とゲートの挙動を分離する。
    """

    def run(self, result=None):
        self._rust_skip_counting = True
        try:
            return super().run(result)
        finally:
            self._rust_skip_counting = False

    def skipTest(self, reason):
        if getattr(self, "_rust_skip_counting", False):
            _RUST_SKIP_REASONS.append(str(reason))
        super().skipTest(reason)


class RustBridgeTest(_RustSkipCounter, unittest.TestCase):
    """Rust CLI ブリッジのパス解決と同等性。"""

    SKIP_GATE_ENV = "RAG_REQUIRE_RUST"

    @classmethod
    def tearDownClass(cls):
        """skip 理由を検査し、RAG_REQUIRE_RUST=1 なら unittest 経由でも fail にする。

        ``__main__`` の後始末だけでは ``python3 -m unittest`` 等のローダー経由で
        ゲートが発動しないため、クラス単位で skip 理由を集計する。
        """
        super().tearDownClass()
        if os.environ.get(cls.SKIP_GATE_ENV, "") != "1":
            return
        reasons = list(_RUST_SKIP_REASONS)
        if reasons:
            raise AssertionError(
                "Rust 同等性テストが skip されました: %d 件 (%s=1)"
                % (len(reasons), cls.SKIP_GATE_ENV)
            )

    def test_binary_path_default(self):
        import rust_bridge as bridge

        path = bridge.rust_binary_path()
        self.assertTrue(str(path).endswith("rag_poc/ragcore/target/debug/ragcore"), str(path))
        self.assertTrue(path.is_absolute())

    def test_cosine_batch_matches_python(self):
        import compare_embeddings
        import rust_bridge as bridge

        if not bridge.available():
            self.skipTest("Rust binary not built")
        pairs = [([1.0, 0.0], [1.0, 0.0]), ([1.0, 0.0], [0.0, 1.0])]
        rust_scores = bridge.cosine_batch(pairs)
        py_scores = [compare_embeddings._cosine(a, b) for a, b in pairs]
        for rust_score, py_score in zip(rust_scores, py_scores):
            self.assertAlmostEqual(rust_score, py_score, places=9)

    def test_run_op_raises_on_error(self):
        import rust_bridge as bridge

        if not bridge.available():
            self.skipTest("Rust binary not built")
        with self.assertRaises(bridge.RustUnavailable):
            bridge.run_op("cosine_batch", {"pairs": [[[1.0], [1.0, 0.0]]]})


class RustOpEquivalenceTest(_RustSkipCounter, unittest.TestCase):
    """CLI op と Python 実装の同等性（Rust バイナリがある場合のみ検証）。

    RustBridgeTest と同じ流儀で、バイナリ未ビルド／当該 op 未実装の場合は
    理由を明記して skip する（親が cargo build 後に実行する想定）。
    """

    SKIP_GATE_ENV = "RAG_REQUIRE_RUST"

    def _rust_unavailable(self, reason: str) -> None:
        """Rust 不在を通常は skip、RAG_REQUIRE_RUST=1 なら fail にする。

        CI 等で Rust 同等性を必須化する場合は ``RAG_REQUIRE_RUST=1`` を設定する。
        通常環境では従来どおり理由付きで skip し、未ビルドを隠蔽しない。
        """
        if os.environ.get(self.SKIP_GATE_ENV, "") == "1":
            self.fail(
                "Rust 同等性テストが必要です（%s=1）: %s" % (self.SKIP_GATE_ENV, reason)
            )
        self.skipTest(reason)

    def _run_op(self, op, payload):
        """bridge.run_op を呼ぶ。利用不可なら skip（ゲート時は fail）する。"""
        import rust_bridge as bridge

        if not bridge.available():
            self._rust_unavailable(
                "Rust binary not built (or cosine probe failed); op=%r" % op
            )
        try:
            return bridge.run_op(op, payload)
        except bridge.RustUnavailable as exc:
            self._rust_unavailable("Rust op %r unavailable: %s" % (op, exc))

    def test_chunk_matches_python_split_long_text(self):
        # 長いテキストのウィンドウ分割（trim / 空除外 / step）が一致すること。
        text = "あ い う え お か き く け こ " * 6
        size, overlap = 13, 3
        rust = self._run_op(
            "chunk", {"text": text, "chunk_size": size, "overlap": overlap}
        )
        with mock.patch.object(ingest, "CHUNK_SIZE", size), \
                mock.patch.object(ingest, "CHUNK_OVERLAP", overlap):
            expected = ingest._split_long_text(text)
        self.assertEqual(rust, expected)
        self.assertGreater(len(expected), 1)

        # 空白のみのウィンドウはどちらも除外される。
        blank = " " * 24
        rust_blank = self._run_op(
            "chunk", {"text": blank, "chunk_size": 5, "overlap": 1}
        )
        with mock.patch.object(ingest, "CHUNK_SIZE", 5), \
                mock.patch.object(ingest, "CHUNK_OVERLAP", 1):
            expected_blank = ingest._split_long_text(blank)
        self.assertEqual(rust_blank, [])
        self.assertEqual(rust_blank, expected_blank)

    def test_point_id_matches_python_ingest(self):
        cases = [
            ("doc.md", 0),
            ("doc.md", 7),
            ("passkey-gateway/docs/other.md", 12345),
            ("別の名前.md", 99),
        ]
        for source, index in cases:
            with self.subTest(source=source, index=index):
                rust = self._run_op(
                    "point_id", {"source": source, "chunk_index": index}
                )
                self.assertEqual(rust, ingest._point_id(source, index))

    def test_fingerprint_matches_python_and_sorts_by_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, body in (
                ("b.md", "B の本文"),
                ("a.md", "A の本文"),
                ("sub/c.md", "C の本文"),
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body.encode("utf-8"))

            expected_count, expected_sha = evaluate._corpus_fingerprint(
                {"CORPUS_DIR": str(root)}
            )
            files = sorted(
                (path for path in root.rglob("*.md") if path.is_file()),
                key=lambda path: str(path),
            )
            paths = [
                {"path": str(path), "bytes": list(path.read_bytes())}
                for path in files
            ]
            rust = self._run_op("fingerprint", {"paths": paths})
            self.assertEqual(rust["count"], expected_count)
            self.assertEqual(rust["sha256"], expected_sha)

            # 入力順を逆にしてもパス順ソートで同一になる（ソート契約）。
            reversed_result = self._run_op(
                "fingerprint", {"paths": list(reversed(paths))}
            )
            self.assertEqual(reversed_result, rust)

    def test_recall_matches_python_heldout_with_basename(self):
        relevant = ["target.md"]
        # バックスラッシュ + 大小文字違いの basename 照合を含む。
        retrieved = ["other.md", "docs\\sub\\TARGET.MD", "third.md"]
        rust = self._run_op(
            "recall",
            {"relevant": relevant, "retrieved": retrieved, "k": 3},
        )

        def fake_search(query, mode, top_k):
            return [{"source": source, "text": ""} for source in retrieved]

        case = {"id": "h", "query": "q", "relevant_sources": relevant}
        record = evaluate._evaluate_heldout_one(
            fake_search, case, "hybrid", len(retrieved)
        )
        self.assertEqual(float(rust), 1.0)
        self.assertEqual(float(rust), float(record["recall_at"]["3"]))

        miss = self._run_op(
            "recall",
            {"relevant": relevant, "retrieved": ["a.md", "b.md"], "k": 3},
        )
        self.assertEqual(float(miss), 0.0)

    def test_ndcg_matches_python_heldout_formula(self):
        relevant = ["target.md", "second.md"]
        retrieved = ["other.md", "dir/target.md", "second.md", "noise.md"]
        rust = self._run_op(
            "ndcg",
            {"relevant": relevant, "retrieved": retrieved, "k": 3},
        )

        def fake_search(query, mode, top_k):
            return [{"source": source, "text": ""} for source in retrieved]

        case = {"id": "h", "query": "q", "relevant_sources": relevant}
        record = evaluate._evaluate_heldout_one(
            fake_search, case, "hybrid", len(retrieved)
        )
        self.assertAlmostEqual(
            float(rust), float(record["ndcg_at"]["3"]), places=6
        )

    def test_latency_matches_python_aggregate(self):
        samples = [0.1, 0.3, 0.5, 0.9]
        rust = self._run_op("latency", {"samples": samples})
        rows = [
            {"latency_stats": {"count": len(samples), "samples": samples}}
        ]
        expected = evaluate._aggregate_latency(rows)
        self.assertEqual(rust["count"], expected["count"])
        for key in ("avg", "median", "min", "max", "stdev"):
            with self.subTest(key=key):
                self.assertAlmostEqual(rust[key], expected[key], places=6)


class RustSkipGateTest(unittest.TestCase):
    """RAG_REQUIRE_RUST=1 で Rust 不在を skip ではなく fail にするゲート。"""

    @staticmethod
    def _gate_run(require_rust, *, available=False, op_error=None):
        """Rust 不在を mock して _run_op を呼び、その結果（例外）を返す。"""
        import rust_bridge as bridge

        gate = RustOpEquivalenceTest("test_chunk_matches_python_split_long_text")
        with mock.patch.dict(os.environ, {"RAG_REQUIRE_RUST": require_rust}), \
                mock.patch.object(bridge, "available", return_value=available):
            if op_error is not None:
                with mock.patch.object(bridge, "run_op", side_effect=op_error):
                    return gate._run_op("chunk", {"text": ""})
            return gate._run_op("chunk", {"text": ""})

    def test_require_rust_turns_binary_missing_into_failure(self):
        with self.assertRaises(AssertionError) as ctx:
            self._gate_run("1")
        message = str(ctx.exception)
        self.assertNotIsInstance(ctx.exception, unittest.SkipTest)
        self.assertIn("RAG_REQUIRE_RUST", message)
        self.assertIn("Rust", message)

    def test_require_rust_turns_op_unavailable_into_failure(self):
        import rust_bridge as bridge

        with self.assertRaises(AssertionError) as ctx:
            self._gate_run(
                "1", available=True,
                op_error=bridge.RustUnavailable("op missing"),
            )
        message = str(ctx.exception)
        self.assertNotIsInstance(ctx.exception, unittest.SkipTest)
        self.assertIn("RAG_REQUIRE_RUST", message)
        self.assertIn("op missing", message)

    def test_without_require_rust_keeps_skipping(self):
        for value in ("", "0"):
            with self.subTest(value=value):
                with self.assertRaises(unittest.SkipTest) as ctx:
                    self._gate_run(value)
                self.assertIn("Rust binary not built", str(ctx.exception))


class LastSearchErrorsContractTest(unittest.TestCase):
    """last_search_errors() が現行 context の値をコピーして返す契約を固定する。

    query.py が contextvars 化されても module global のままでも成立するよう、
    内部変数へ直接触れず公開 API 経由で検証する（別担当の変更未反映でも壊れない）。
    """

    @staticmethod
    def _clear_via_search() -> None:
        """成功する search でエラー状態をリセットする。"""
        with mock.patch.object(query, "search_dense", return_value=[]), \
                mock.patch.object(query, "search_keyword", return_value=[]):
            query.search("短い質問")

    def test_returns_current_snapshot_as_copy(self) -> None:
        with mock.patch.object(query, "_warn", lambda message: None):
            query.search("あ" * (config.QUESTION_MAX_CHARS + 1))
            first = query.last_search_errors()
            self.assertEqual(len(first), 1)
            self.assertEqual(first[0]["stage"], "input-validation")
            self.assertEqual(first[0]["mode"], "hybrid")
            self.assertIn("type", first[0])
            self.assertIn("message", first[0])

            # 返値を書き換えても現行 context の状態は変わらない（コピー契約）。
            first[0]["stage"] = "mutated"
            first.append({"stage": "extra"})
            second = query.last_search_errors()
            self.assertEqual(len(second), 1)
            self.assertEqual(second[0]["stage"], "input-validation")
            self._clear_via_search()

    def test_search_resets_errors_between_calls(self) -> None:
        with mock.patch.object(query, "_warn", lambda message: None):
            query.search("あ" * (config.QUESTION_MAX_CHARS + 1))
            self.assertTrue(query.last_search_errors())
            self._clear_via_search()
            self.assertEqual(query.last_search_errors(), [])

    def test_bogus_mode_returns_no_hits_and_records_validation_stage(self) -> None:
        # 未知 mode は hybrid へ黙ってフォールバックせず、空ヒットと
        # mode-validation エラーを残す（dense/keyword を呼ばない）。
        with mock.patch.object(
            query, "search_dense", side_effect=AssertionError("dense called")
        ), mock.patch.object(
            query, "search_keyword", side_effect=AssertionError("keyword called")
        ):
            hits = query.search("短い質問", mode="bogus")

        errors = query.last_search_errors()
        self.assertEqual(hits, [])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["stage"], "mode-validation")
        self.assertEqual(errors[0]["mode"], "bogus")
        self.assertIn("type", errors[0])
        self.assertIn("message", errors[0])
        self._clear_via_search()


class EvaluateMainTest(unittest.TestCase):
    """evaluate.main() の分岐を mock で検証する（実検索・実通信なし）。

    evaluate.py は読み取り専用（レイテンシ計測ジョブが使用中）のため、
    OUTPUT_PATH と sys.argv をテスト側で差し替えて main() を直接呼ぶ。
    """

    QRELS_BODY = {
        "schema_version": 1,
        "simple_queries": [
            {"id": "s-main", "query": "質問", "expected": ["キーワード"]}
        ],
        "heldout_queries": [
            {"id": "h-main", "query": "質問", "relevant_sources": ["target.md"]}
        ],
    }
    ALT_QRELS_BODY = {
        "schema_version": 7,
        "simple_queries": [
            {"id": "s-alt", "query": "別ドメイン", "expected": ["別語"]}
        ],
        "heldout_queries": [
            {"id": "h-alt", "query": "別ドメイン", "relevant_sources": ["alt.md"]}
        ],
    }
    METADATA = {
        "embed_model": "embed-main",
        "chat_model": "chat-main",
        "vector_size": 768,
        "corpus_file_count": 2,
        "corpus_sha256": "deadbeef",
    }

    def _invoke_main(self, argv, top_k=10):
        """OUTPUT_PATH / sys.argv を差し替えて main() を呼び成果物を返す。

        戻り値は (payload, heldout_calls, simple_calls)。
        """
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "eval-results.json"
            heldout_calls: list[tuple[int, int]] = []
            simple_calls: list[tuple[int, int]] = []

            def search(question, mode, top_k):
                return []

            def fake_heldout(func, retrieval_k, repeat, errors_getter):
                heldout_calls.append((retrieval_k, repeat))
                return [], []

            def fake_simple(func, k, repeat, errors_getter):
                simple_calls.append((k, repeat))
                return [], []

            fake_query = types.SimpleNamespace(last_search_errors=lambda: [])
            patchers = (
                mock.patch.object(evaluate, "OUTPUT_PATH", str(output)),
                mock.patch.object(sys, "argv", ["evaluate.py", *argv]),
                mock.patch.object(evaluate, "QRELS", self.QRELS_BODY),
                mock.patch.object(evaluate, "EVAL_QUERIES", self.QRELS_BODY["simple_queries"]),
                mock.patch.object(
                    evaluate, "HELD_OUT_QUERIES", self.QRELS_BODY["heldout_queries"]
                ),
                mock.patch.object(evaluate, "QRELS_SCHEMA_VERSION", 1),
                mock.patch.object(
                    evaluate, "_load_search_function",
                    return_value=(search, fake_query),
                ),
                mock.patch.object(evaluate, "_validate_settings", return_value=[]),
                mock.patch.object(evaluate, "_endpoint_problems", return_value=([], [])),
                mock.patch.object(evaluate, "_load_top_k", return_value=top_k),
                mock.patch.object(evaluate, "_load_repeat", return_value=1),
                mock.patch.object(
                    evaluate, "_load_collection", return_value="main-collection"
                ),
                mock.patch.object(evaluate, "_eval_metadata", return_value=self.METADATA),
                mock.patch.object(evaluate, "_corpus_dir", return_value=Path(tmp)),
                mock.patch.object(evaluate, "_run_heldout_eval", fake_heldout),
                mock.patch.object(evaluate, "_run_simple_eval", fake_simple),
                mock.patch.object(evaluate, "_reuse_simple_results", return_value=([], [])),
                mock.patch("sys.stdout", new_callable=io.StringIO),
                mock.patch("sys.stderr", new_callable=io.StringIO),
            )
            with contextlib.ExitStack() as stack:
                for patcher in patchers:
                    stack.enter_context(patcher)
                code = evaluate.main()
            self.assertEqual(code, 0)
            self.assertTrue(output.is_file())
            payload = json.loads(output.read_text(encoding="utf-8"))
            return payload, heldout_calls, simple_calls

    def test_retrieval_k_is_raised_to_ten_with_note(self) -> None:
        payload, heldout_calls, simple_calls = self._invoke_main([], top_k=3)
        self.assertEqual(payload["top_k"], 3)
        self.assertEqual(payload["heldout_retrieval_k"], 10)
        self.assertEqual(heldout_calls, [(10, 1)])
        self.assertEqual(simple_calls, [(3, 1)])
        self.assertTrue(
            any("取得件数を 3 → 10 に引き上げた" in note for note in payload["notes"]),
            payload["notes"],
        )

    def test_qrels_option_switches_label_and_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            alt = Path(tmp) / "alt-qrels.json"
            alt.write_text(
                json.dumps(self.ALT_QRELS_BODY, ensure_ascii=False), encoding="utf-8"
            )
            payload, _, _ = self._invoke_main(["--qrels", str(alt)])
        self.assertEqual(payload["qrels_label"], "alt-qrels.json")
        self.assertEqual(payload["qrels_schema_version"], 7)
        self.assertEqual(
            payload["heldout"]["queries"], self.ALT_QRELS_BODY["heldout_queries"]
        )
        self.assertTrue(
            any(
                "評価クエリは alt-qrels.json（schema_version=7）から読み込み" in note
                for note in payload["notes"]
            ),
            payload["notes"],
        )

    def test_payload_contains_expected_assembly(self) -> None:
        payload, _, _ = self._invoke_main([])
        for key in (
            "generated_at", "schema_version", "qrels_schema_version", "qrels_label",
            "script", "search_function", "collection", "embed_model", "chat_model",
            "vector_size", "corpus_file_count", "corpus_sha256", "repeat", "modes",
            "top_k", "heldout_retrieval_k", "metric_definitions", "notes", "summary",
            "results", "heldout",
        ):
            with self.subTest(key=key):
                self.assertIn(key, payload)
        self.assertEqual(payload["script"], "rag_poc/evaluate.py")
        self.assertEqual(
            payload["search_function"],
            {"module": "query", "name": "search", "preferred_name": "search"},
        )
        self.assertEqual(payload["collection"], "main-collection")
        self.assertEqual(payload["embed_model"], self.METADATA["embed_model"])
        self.assertEqual(payload["vector_size"], self.METADATA["vector_size"])
        self.assertEqual(payload["modes"], list(evaluate.MODES))
        self.assertEqual(payload["top_k"], 10)
        self.assertEqual(payload["heldout_retrieval_k"], 10)
        self.assertEqual(payload["repeat"], 1)
        self.assertEqual(payload["results"], [])
        self.assertEqual(set(payload["summary"]), set(evaluate.MODES))
        self.assertEqual(
            set(payload["heldout"]),
            {"definitions", "recall_k", "ndcg_k", "queries", "summary", "results"},
        )
        self.assertEqual(
            payload["heldout"]["queries"], self.QRELS_BODY["heldout_queries"]
        )
        self.assertTrue(all(isinstance(note, str) for note in payload["notes"]))


class QueryHelpTest(unittest.TestCase):
    """--help が設定検証より前に処理されることを mock で固定する（通信なし）。"""

    def test_help_is_shown_even_when_settings_are_invalid(self):
        invalid = {
            "VECTOR_SIZE": 0,
            "TOP_K": 0,
            "CHUNK_OVERLAP": config.CHUNK_SIZE,
            "OLLAMA_URL": "https://example.test:11434",
        }
        with mock.patch.multiple(config, **invalid):
            # まず不正設定であることを確認してから、sys.argv を差し替えて呼ぶ。
            self.assertTrue(config.validate_settings())
            with mock.patch.object(sys, "argv", ["query.py", "--help"]), \
                    mock.patch("sys.stdout", new_callable=io.StringIO) as stdout, \
                    mock.patch("sys.stderr", new_callable=io.StringIO) as stderr, \
                    mock.patch.multiple(
                        query,
                        TOP_K=0,
                        OLLAMA_URL=invalid["OLLAMA_URL"],
                    ):
                code = None
                try:
                    code = query.main()
                except SystemExit as exc:  # argparse の --help は SystemExit(0)
                    code = exc.code

        self.assertEqual(code, 0)
        self.assertIn("usage:", stdout.getvalue())
        self.assertIn("--help", stdout.getvalue())
        self.assertIn("--mode", stdout.getvalue())
        self.assertNotIn("設定エラー", stdout.getvalue() + stderr.getvalue())


class RecreateGuardTest(unittest.TestCase):
    """--recreate の --yes ガードを mock で検証する（実 DELETE なし）。"""

    def test_recreate_without_yes_returns_two_without_deleting(self):
        stderr = io.StringIO()

        def forbidden(*args, **kwargs):
            raise AssertionError("--yes 無しで削除処理・HTTP セッションを開始してはならない")

        with mock.patch.multiple(
            ingest,
            COLLECTION="rag_poc_docs",
            CORPUS_DIR="passkey-gateway/docs",
            _CLI_CORPUS_DIR=None,
        ), mock.patch.object(ingest, "_make_session", forbidden), \
                mock.patch.object(ingest, "recreate_collection", forbidden), \
                mock.patch("sys.stderr", stderr):
            code = ingest.main(
                ["--recreate", "--corpus-dir", "rag_poc/corpus-subset"]
            )

        self.assertEqual(code, 2)
        message = stderr.getvalue()
        self.assertIn("コレクション rag_poc_docs を削除して再作成します", message)
        self.assertIn("実行するには --yes を指定してください", message)

    def test_parse_args_recreate_and_yes_are_independent_flags(self):
        only = ingest._parse_args(["--recreate"])
        self.assertTrue(only.recreate)
        self.assertFalse(only.yes)

        approved = ingest._parse_args(["--recreate", "--yes"])
        self.assertTrue(approved.recreate)
        self.assertTrue(approved.yes)

        default = ingest._parse_args([])
        self.assertFalse(default.recreate)
        self.assertFalse(default.yes)


class RecreateFlowTest(unittest.TestCase):
    """recreate_collection が DELETE → PUT の順で呼ぶこと（実通信なし）。"""

    def test_delete_then_put_with_expected_vector_config(self):
        class FakeResponse:
            status_code = 200

        session = mock.Mock()
        session.delete.return_value = FakeResponse()
        session.put.return_value = FakeResponse()

        with mock.patch.object(ingest, "COLLECTION", "rag_poc_docs"), \
                mock.patch.object(ingest, "QDRANT_URL", "http://127.0.0.1:6333"), \
                mock.patch.object(ingest, "resolved_vector_size", return_value=768):
            ingest.recreate_collection(session)

        self.assertEqual(session.delete.call_count, 1)
        self.assertEqual(session.put.call_count, 1)
        expected_url = "http://127.0.0.1:6333/collections/rag_poc_docs"
        self.assertEqual(session.delete.call_args[0][0], expected_url)
        self.assertEqual(session.put.call_args[0][0], expected_url)
        self.assertEqual(
            session.put.call_args[1]["json"],
            {"vectors": {"size": 768, "distance": "Cosine"}},
        )


class EvaluateCollectionsTest(unittest.TestCase):
    """--collections のパースと複数コレクション分岐を mock で検証する。"""

    def test_parse_collections_splits_trims_and_deduplicates(self):
        self.assertEqual(
            evaluate._parse_collections(" a, b ,c,a "), ["a", "b", "c"]
        )
        self.assertEqual(evaluate._parse_collections(None), [])
        self.assertEqual(evaluate._parse_collections(""), [])
        self.assertEqual(evaluate._parse_collections(" , "), [])

    def test_parse_args_exposes_collections_option(self):
        args = evaluate._parse_args(["--collections", "a,b,c"])
        self.assertEqual(args.collections, "a,b,c")
        self.assertIsNone(evaluate._parse_args([]).collections)

    def test_collection_wrapper_accepts_both_search_and_answer_signatures(self):
        def fake_search(question, mode, top_k):
            return []

        def fake_answer(question, hits):
            return "ok"

        search_wrapped = evaluate._collection_search(fake_search, query, "col_x")
        answer_wrapped = evaluate._collection_search(fake_answer, query, "col_x")
        self.assertEqual(search_wrapped("q", "hybrid", 3), [])
        self.assertEqual(answer_wrapped("q", [{"source": "s", "text": "t"}]), "ok")
        self.assertEqual(query.COLLECTION, "rag_poc_docs")

    def test_collection_output_path_is_per_collection(self):
        path = evaluate._collection_output_path("alpha")
        self.assertTrue(path.endswith("eval-results-alpha.json"))
        self.assertTrue(os.path.isabs(path))
        special = evaluate._collection_output_path("a/b:c")
        self.assertTrue(special.startswith(os.path.join(evaluate.HERE, "eval-results-a_b_c_")))
        # 安定ハッシュのため同じ入力は同じパス、異なる衝突候補は別パス。
        self.assertEqual(special, evaluate._collection_output_path("a/b:c"))
        self.assertNotEqual(special, evaluate._collection_output_path("a b:c"))

    def test_collection_output_path_boundary_names_use_16bit_suffix(self):
        """___ / ... / 空に近い名前は collection フォールバック + blake2b suffix。"""
        for name in ("___", "...", "", "   ", ".", "_", "._"):
            with self.subTest(name=name):
                path = evaluate._collection_output_path(name)
                basename = os.path.basename(path)
                self.assertTrue(os.path.isabs(path))
                self.assertTrue(basename.startswith("eval-results-collection_"), basename)
                self.assertTrue(basename.endswith(".json"), basename)
                suffix = basename[len("eval-results-collection_"):-len(".json")]
                self.assertEqual(len(suffix), 4, basename)
                self.assertTrue(all(char in "0123456789abcdef" for char in suffix), basename)
                self.assertEqual(
                    suffix,
                    hashlib.blake2b(
                        name.strip().encode("utf-8"), digest_size=2
                    ).hexdigest(),
                )

        # 空白のみは空文字と同じ正規化結果になる。
        self.assertEqual(
            evaluate._collection_output_path(""), evaluate._collection_output_path("   ")
        )
        # 境界値同士も 16bit suffix により衝突しない。
        paths = {
            name: evaluate._collection_output_path(name)
            for name in ("___", "...", "", ".", "_")
        }
        self.assertEqual(len(set(paths.values())), len(paths))

    def test_collection_output_path_sanitized_collisions_are_separated(self):
        """a/b と a:b は同じ _ サニタイズ結果でも 16bit suffix で分離される。"""
        plain = evaluate._collection_output_path("a_b")
        self.assertEqual(os.path.basename(plain), "eval-results-a_b.json")

        paths = {}
        for name in ("a/b", "a:b", "a b"):
            with self.subTest(name=name):
                path = evaluate._collection_output_path(name)
                basename = os.path.basename(path)
                self.assertTrue(basename.startswith("eval-results-a_b_"), basename)
                self.assertTrue(basename.endswith(".json"), basename)
                suffix = basename[len("eval-results-a_b_"):-len(".json")]
                self.assertEqual(len(suffix), 4, basename)
                self.assertTrue(all(char in "0123456789abcdef" for char in suffix), basename)
                self.assertEqual(
                    suffix,
                    hashlib.blake2b(name.encode("utf-8"), digest_size=2).hexdigest(),
                )
                self.assertEqual(path, evaluate._collection_output_path(name))
                paths[name] = path

        self.assertEqual(len(set(paths.values())), len(paths))
        self.assertNotEqual(paths["a/b"], paths["a:b"])
        self.assertNotIn(plain, paths.values())

    def test_multi_collection_branch_writes_each_and_prints_summary(self):
        calls = []

        def fake_evaluate_once(args, output_path, collection_override=None, backup=False):
            calls.append((output_path, collection_override, backup))
            payload = {
                "heldout": {
                    "summary": {
                        mode: {
                            "recall_at": {"10": 0.5},
                            "ndcg_at": {"10": 0.4},
                            "latency_stats": {"avg": 0.01},
                        }
                        for mode in evaluate.MODES
                    }
                }
            }
            Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            def fake_output_path(name):
                return str(Path(tmp) / ("eval-results-%s.json" % name))

            stderr = io.StringIO()
            with mock.patch.object(evaluate, "_evaluate_once", fake_evaluate_once), \
                    mock.patch.object(
                        evaluate, "_collection_output_path", fake_output_path
                    ), \
                    mock.patch.object(
                        sys, "argv", ["evaluate.py", "--collections", "alpha,beta,gamma"]
                    ), \
                    mock.patch("sys.stderr", stderr), \
                    mock.patch("sys.stdout", new_callable=io.StringIO):
                code = evaluate.main()

        self.assertEqual(code, 0)
        self.assertEqual([call[1] for call in calls], ["alpha", "beta", "gamma"])
        self.assertTrue(all(call[2] is False for call in calls))
        table = stderr.getvalue()
        for token in ("collection", "mode", "recall@10", "nDCG@10", "latency_avg_sec"):
            with self.subTest(token=token):
                self.assertIn(token, table)
        for name in ("alpha", "beta", "gamma"):
            with self.subTest(name=name):
                self.assertIn(name, table)
        for mode in evaluate.MODES:
            with self.subTest(mode=mode):
                self.assertIn(mode, table)

    def test_without_collections_keeps_single_collection_behavior(self):
        calls = []

        def fake_evaluate_once(args, output_path, collection_override=None, backup=False):
            calls.append((output_path, collection_override, backup))
            return 0

        with mock.patch.object(evaluate, "_evaluate_once", fake_evaluate_once), \
                mock.patch.object(sys, "argv", ["evaluate.py"]), \
                mock.patch("sys.stdout", new_callable=io.StringIO), \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            code = evaluate.main()

        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0][1])
        self.assertTrue(calls[0][2])


class GeneratedAnswerErrorTest(unittest.TestCase):
    """生成失敗が検索エラーを汚さず answer_error へ分離されること。"""

    def test_generation_failure_stays_out_of_record_error(self):
        record = {"query": "q", "mode": "hybrid", "hits": 1}

        def bad_answer(question, hits):
            raise RuntimeError("boom")

        evaluate._record_generated_answer(
            record, "q", [{"source": "s", "text": "t"}], bad_answer
        )
        self.assertIsNone(record.get("error"))
        self.assertIn("answer_error", record)
        self.assertIn("RuntimeError", record["answer_error"])
        self.assertIsNone(record["answer"])
        self.assertIsNotNone(record["answer_latency_sec"])

    def test_successful_answer_has_no_error_fields(self):
        record = {"query": "q", "mode": "hybrid", "hits": 1}
        evaluate._record_generated_answer(
            record, "q", [{"source": "s", "text": "t"}], lambda q, hits: "回答"
        )
        self.assertEqual(record["answer"], "回答")
        self.assertIsNone(record.get("error"))
        self.assertNotIn("answer_error", record)

    @staticmethod
    def _timeout_class():
        """requests.Timeout を返す（未導入スタブなら同名のローカル例外）。"""
        import requests

        timeout = getattr(requests, "Timeout", None)
        if timeout is None:
            # requests 未導入環境では先頭で stub が入っている。実通信はしないため
            # 例外種別名を揃えたローカル例外でタイムアウト経路を固定する。
            timeout = type("Timeout", (Exception,), {})
        return timeout

    def test_answer_timeout_error_is_masked_and_record_error_stays_none(self):
        record = {"query": "q", "mode": "hybrid", "hits": 1}
        secret = "Bearer ZZZTIMEOUTTOKENZZZ"

        def slow_answer(question, hits):
            raise self._timeout_class()("Ollama 応答タイムアウト: " + secret)

        evaluate._record_generated_answer(
            record, "q", [{"source": "s", "text": "t"}], slow_answer
        )

        self.assertIsNone(record.get("error"))
        self.assertIn("answer_error", record)
        self.assertIn("Timeout", record["answer_error"])
        self.assertIn("<masked>", record["answer_error"])
        self.assertNotIn("ZZZTIMEOUTTOKENZZZ", record["answer_error"])
        self.assertIsNone(record["answer"])
        self.assertIsNotNone(record["answer_latency_sec"])
        self.assertEqual(record["sources"], ["s"])

    def test_answer_timeout_does_not_stop_full_evaluation(self):
        secret = "Bearer ZZZTIMEOUTTOKENZZZ"

        def slow_answer(question, hits):
            raise self._timeout_class()("Ollama 応答タイムアウト: " + secret)

        def fake_search(question, mode, top_k):
            # recall は source 側のキーワード一致で算出されるため、双方に含める。
            return [{"source": "キーワード.md", "text": "キーワード を含む本文"}]

        record = evaluate._evaluate_one(
            fake_search, "質問", ["キーワード"], "hybrid", 3,
            with_answer=True, answer_fn=slow_answer,
        )

        self.assertIsNone(record["error"])
        self.assertEqual(record["hits"], 1)
        self.assertTrue(record["recall"])
        self.assertIn("answer_error", record)
        self.assertIn("Timeout", record["answer_error"])
        self.assertNotIn("ZZZTIMEOUTTOKENZZZ", json.dumps(record, ensure_ascii=False))
        self.assertIsNone(record["answer"])
        self.assertIsNotNone(record["answer_latency_sec"])


class CloudClientTest(unittest.TestCase):
    """cloud_client の v1 契約を requests.Session の fake で検証する（実 HTTP なし）。

    v1-client は requests.Session().request() と _SESSION の遅延生成を使う。
    _SESSION はモジュール共有なのでテストごとに fake へ作り直し、前のテストの
    handler が呼ばれないようにする。_SESSION が非 None の間は HTTPAdapter /
    Retry の import が走らないため、実通信も外部モジュールの準備も不要。
    秘密情報はダミー文字列のみを使う。
    """

    GO_KEY = "OPENCODE_GO_API_KEY"
    LEGACY_KEY = "OPENCODE_API_KEY"

    class FakeRequestException(Exception):
        """requests.RequestException の代わりに使うローカル例外。"""

    class FakeHTTPError(FakeRequestException):
        def __init__(self, status_code, message="http error"):
            super().__init__(message)
            self.response = types.SimpleNamespace(status_code=status_code)

    def setUp(self):
        self._old_session = getattr(cloud_client, "_SESSION", None)
        self._old_usage = getattr(cloud_client, "_LAST_USAGE", None)
        self._handler = None
        self._session = self._make_fake_session()
        cloud_client._SESSION = self._session
        cloud_client._LAST_USAGE = None

    def tearDown(self):
        cloud_client._SESSION = self._old_session
        cloud_client._LAST_USAGE = self._old_usage

    def _make_fake_session(self):
        """config.RETRY_* に従い status code をリトライする fake Session。"""
        test = self

        class FakeSession:
            def __init__(self):
                self.mounts = {}
                self.calls = []

            def mount(self, prefix, adapter):
                self.mounts[prefix] = adapter

            def request(self, method, url, **kwargs):
                handler = test._handler
                if handler is None:
                    raise AssertionError("handler が未設定です")
                total = int(config.RETRY_TOTAL)
                forcelist = set(config.RETRY_STATUS_FORCELIST)
                response = None
                for attempt in range(total + 1):
                    self.calls.append((method, url, kwargs))
                    response = handler(url, **kwargs)
                    if (getattr(response, "status_code", None) in forcelist
                            and attempt < total):
                        continue
                    return response
                return response

        return FakeSession()

    def _fake_requests(self, handler):
        """cloud_client.requests を差し替え、handler を fake Session へ渡す。"""
        self._handler = handler

        class _FakeSession:
            def request(self, method, url, **kwargs):
                return handler(url, **kwargs)

            def mount(self, prefix, adapter):
                return None

        return types.SimpleNamespace(
            post=handler,
            Session=_FakeSession,
            RequestException=self.FakeRequestException,
            HTTPError=self.FakeHTTPError,
            Timeout=self.FakeRequestException,
        )

    @classmethod
    def _response(cls, data, status_code=200, text=""):
        def raise_for_status():
            if status_code >= 400:
                raise cls.FakeHTTPError(status_code, "http %d" % status_code)

        return types.SimpleNamespace(
            status_code=status_code,
            text=text,
            headers={},
            json=lambda: data,
            raise_for_status=raise_for_status,
        )

    @contextlib.contextmanager
    def _cloud_env(self, **overrides):
        values = {
            cloud_client.ENV_ENABLED: "1",
            self.GO_KEY: "key-value-for-test",
        }
        values.update(overrides)
        with _temporary_env(**values):
            yield

    @staticmethod
    def _read_usage():
        """v1-client の last_usage() が返すコピーをそのまま読む。"""
        return cloud_client.last_usage()

    @staticmethod
    def _usage_text(value):
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, default=str, ensure_ascii=False)
        data = getattr(value, "__dict__", None)
        if isinstance(data, dict):
            return json.dumps(data, default=str, ensure_ascii=False)
        return str(value)

    def test_enabled_only_true_for_exact_one(self):
        for value in ("1", "true", "True", "yes", "01", "0", "", " 1 "):
            with self.subTest(value=value):
                with _temporary_env(**{cloud_client.ENV_ENABLED: value}):
                    self.assertIs(cloud_client.enabled(), value == "1")

    def test_enabled_false_when_unset(self):
        with _temporary_env(**{cloud_client.ENV_ENABLED: None}):
            self.assertIs(cloud_client.enabled(), False)

    def test_api_key_name_skips_empty_string(self):
        with _temporary_env(**{self.LEGACY_KEY: "", self.GO_KEY: "go-value"}):
            self.assertEqual(cloud_client.api_key_name(), self.GO_KEY)

    def test_api_key_name_prefers_go_key(self):
        with _temporary_env(
            **{self.LEGACY_KEY: "legacy-value", self.GO_KEY: "go-value"}
        ):
            self.assertEqual(cloud_client.api_key_name(), self.GO_KEY)

    def test_api_key_name_returns_none_without_keys(self):
        with _temporary_env(**{self.LEGACY_KEY: None, self.GO_KEY: None}):
            self.assertIsNone(cloud_client.api_key_name())

    def test_api_key_name_never_returns_the_value(self):
        secret = "ZZZKEYVALUEZZZ"
        with _temporary_env(**{self.GO_KEY: secret}):
            name = cloud_client.api_key_name()
        self.assertEqual(name, self.GO_KEY)
        self.assertNotEqual(name, secret)

    def test_embed_batches_and_preserves_input_order(self):
        texts = ["text-%d" % index for index in range(7)]
        lookup = {text: [float(index)] for index, text in enumerate(texts)}
        seen = []

        def handler(url, **kwargs):
            raw = kwargs["json"]["input"]
            batch = [raw] if isinstance(raw, str) else list(raw)
            seen.append(batch)
            return self._response(
                {"data": [{"embedding": lookup[text]} for text in batch]}
            )

        with self._cloud_env(), \
                mock.patch.object(
                    cloud_client, "requests", self._fake_requests(handler)
                ), \
                mock.patch("time.sleep", lambda *_: None):
            vectors = cloud_client.embed(texts, batch_size=3)

        self.assertEqual(vectors, [[float(index)] for index in range(7)])
        self.assertEqual(seen, [texts[0:3], texts[3:6], texts[6:7]])

    def test_embed_batch_size_zero_is_rejected_without_http(self):
        calls = []

        def handler(url, **kwargs):
            calls.append(kwargs)
            return self._response({"data": [{"embedding": [0.0]}]})

        with self._cloud_env(), \
                mock.patch.object(
                    cloud_client, "requests", self._fake_requests(handler)
                ):
            with self.assertRaisesRegex(Exception, "batch_size"):
                cloud_client.embed(["a"], batch_size=0)

        self.assertEqual(calls, [])

    def test_embed_negative_batch_size_is_rejected(self):
        calls = []

        def handler(url, **kwargs):
            calls.append(kwargs)
            return self._response({"data": [{"embedding": [0.0]}]})

        with self._cloud_env(), \
                mock.patch.object(
                    cloud_client, "requests", self._fake_requests(handler)
                ):
            with self.assertRaisesRegex(Exception, "batch_size"):
                cloud_client.embed(["a"], batch_size=-1)

        self.assertEqual(calls, [])

    def test_embed_true_batch_size_is_cloud_unavailable(self):
        # bool は int の部分クラスだが件数として不正。CloudUnavailable で拒否し、
        # HTTP リクエストを一切送らない（マスク済み例外の契約も維持）。
        calls = []

        def handler(url, **kwargs):
            calls.append(kwargs)
            return self._response({"data": [{"embedding": [0.0]}]})

        with self._cloud_env(), \
                mock.patch.object(
                    cloud_client, "requests", self._fake_requests(handler)
                ):
            with self.assertRaises(cloud_client.CloudUnavailable) as ctx:
                cloud_client.embed(["a"], batch_size=True)

        self.assertIn("batch_size", str(ctx.exception))
        self.assertEqual(calls, [])

    def test_embed_false_batch_size_is_cloud_unavailable(self):
        calls = []

        def handler(url, **kwargs):
            calls.append(kwargs)
            return self._response({"data": [{"embedding": [0.0]}]})

        with self._cloud_env(), \
                mock.patch.object(
                    cloud_client, "requests", self._fake_requests(handler)
                ):
            with self.assertRaises(cloud_client.CloudUnavailable) as ctx:
                cloud_client.embed(["a"], batch_size=False)

        self.assertIn("batch_size", str(ctx.exception))
        self.assertEqual(calls, [])

    def test_embed_retries_429_then_succeeds(self):
        state = {"count": 0}

        def handler(url, **kwargs):
            index = state["count"]
            state["count"] += 1
            if index == 0:
                return self._response(
                    {"error": "too many requests"}, status_code=429
                )
            return self._response({"data": [{"embedding": [0.25]}]})

        with self._cloud_env(), \
                mock.patch.object(
                    cloud_client, "requests", self._fake_requests(handler)
                ), \
                mock.patch("time.sleep", lambda *_: None):
            vectors = cloud_client.embed(["a"], batch_size=1)

        self.assertEqual(vectors, [[0.25]])
        self.assertEqual(state["count"], 2)

    def test_embed_retries_5xx_then_succeeds(self):
        statuses = (500, 503)
        state = {"count": 0}

        def handler(url, **kwargs):
            index = state["count"]
            state["count"] += 1
            if index < len(statuses):
                return self._response(
                    {"error": "server error"}, status_code=statuses[index]
                )
            return self._response({"data": [{"embedding": [0.5]}]})

        with self._cloud_env(), \
                mock.patch.object(
                    cloud_client, "requests", self._fake_requests(handler)
                ), \
                mock.patch("time.sleep", lambda *_: None):
            vectors = cloud_client.embed(["a"], batch_size=1)

        self.assertEqual(vectors, [[0.5]])
        self.assertEqual(state["count"], 3)

    def test_embed_fails_fast_on_4xx(self):
        for status in (400, 401, 404):
            with self.subTest(status=status):
                state = {"count": 0}

                def handler(url, _status=status, **kwargs):
                    state["count"] += 1
                    return self._response(
                        {"error": "client error"}, status_code=_status
                    )

                with self._cloud_env(), \
                        mock.patch.object(
                            cloud_client, "requests", self._fake_requests(handler)
                        ), \
                        mock.patch("time.sleep", lambda *_: None):
                    with self.assertRaises(Exception):
                        cloud_client.embed(["a"], batch_size=1)

                self.assertEqual(state["count"], 1)

    def test_chat_attaches_session_header(self):
        seen = []

        def handler(url, **kwargs):
            seen.append(dict(kwargs.get("headers") or {}))
            return self._response(
                {"choices": [{"message": {"content": "ok"}}]}
            )

        with self._cloud_env(), \
                mock.patch.object(
                    cloud_client, "requests", self._fake_requests(handler)
                ):
            answer = cloud_client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(answer, "ok")
        self.assertEqual(len(seen), 1)
        values = [
            value for key, value in seen[0].items()
            if str(key).lower() == "x-opencode-session"
        ]
        self.assertTrue(values, seen[0])
        self.assertTrue(all(isinstance(value, str) and value for value in values))

    def test_embed_omits_session_header(self):
        seen = []

        def handler(url, **kwargs):
            seen.append(dict(kwargs.get("headers") or {}))
            return self._response({"data": [{"embedding": [0.1]}]})

        with self._cloud_env(), \
                mock.patch.object(
                    cloud_client, "requests", self._fake_requests(handler)
                ):
            cloud_client.embed(["a"], batch_size=1)

        self.assertEqual(len(seen), 1)
        keys = {str(key).lower() for key in seen[0]}
        self.assertNotIn("x-opencode-session", keys)

    def test_usage_is_readable_after_chat(self):
        usage = {"prompt_tokens": 111, "completion_tokens": 7, "total_tokens": 118}

        def handler(url, **kwargs):
            return self._response({
                "choices": [{"message": {"content": "ok"}}],
                "usage": usage,
            })

        with self._cloud_env(), \
                mock.patch.object(
                    cloud_client, "requests", self._fake_requests(handler)
                ):
            self.assertEqual(
                cloud_client.chat([{"role": "user", "content": "hi"}]), "ok"
            )

        observed = self._read_usage()
        self.assertIsNotNone(observed, "usage が last_usage() 等で参照できません")
        text = self._usage_text(observed)
        self.assertIn("111", text)
        self.assertIn("118", text)

    def test_usage_is_readable_after_embed(self):
        usage = {"prompt_tokens": 222, "total_tokens": 229}

        def handler(url, **kwargs):
            return self._response({
                "data": [{"embedding": [0.1]}],
                "usage": usage,
            })

        with self._cloud_env(), \
                mock.patch.object(
                    cloud_client, "requests", self._fake_requests(handler)
                ):
            cloud_client.embed(["a"], batch_size=1)

        observed = self._read_usage()
        self.assertIsNotNone(observed, "usage が last_usage() 等で参照できません")
        text = self._usage_text(observed)
        self.assertIn("222", text)
        self.assertIn("229", text)

    def test_error_messages_mask_key_value_and_endpoint(self):
        secret_value = "ZZZKEYVALUEZZZ"
        secret_url = "https://cloud.example.test/v1?api_key=ZZZURLSECRETZZZ"

        def handler(url, **kwargs):
            raise self.FakeRequestException(
                "failed for " + secret_url + " api_key=" + secret_value
            )

        with _temporary_env(**{
            cloud_client.ENV_ENABLED: "1",
            self.GO_KEY: secret_value,
            cloud_client.ENV_BASE_URL: secret_url,
        }), mock.patch.object(
            cloud_client, "requests", self._fake_requests(handler)
        ), mock.patch("time.sleep", lambda *_: None):
            with self.assertRaises(Exception) as ctx:
                cloud_client.embed(["a"], batch_size=1)

        message = str(ctx.exception)
        for needle in (secret_value, "cloud.example.test", "ZZZURLSECRETZZZ"):
            with self.subTest(needle=needle):
                self.assertNotIn(needle, message)
        self.assertIn("<masked>", message)

    def test_chat_error_masks_secret(self):
        secret = "Bearer ZZZCHATSECRETZZZ"

        def handler(url, **kwargs):
            raise self.FakeRequestException("chat failed: " + secret)

        with self._cloud_env(), \
                mock.patch.object(
                    cloud_client, "requests", self._fake_requests(handler)
                ), \
                mock.patch("time.sleep", lambda *_: None):
            with self.assertRaises(Exception) as ctx:
                cloud_client.chat([{"role": "user", "content": "hi"}])

        message = str(ctx.exception)
        self.assertNotIn("ZZZCHATSECRETZZZ", message)
        self.assertIn("<masked>", message)

    def test_shared_session_mounts_retry_with_config_values(self):
        """_shared_session() が config.RETRY_* を Retry へ渡して mount すること。"""
        captured = {}

        class FakeRetry:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        class FakeAdapter:
            def __init__(self, max_retries=None):
                self.max_retries = max_retries

        class MountSession:
            def __init__(self):
                self.mounts = {}

            def mount(self, prefix, adapter):
                self.mounts[prefix] = adapter

        adapters_module = types.ModuleType("requests.adapters")
        adapters_module.HTTPAdapter = FakeAdapter
        retry_module = types.ModuleType("urllib3.util.retry")
        retry_module.Retry = FakeRetry
        util_module = types.ModuleType("urllib3.util")
        util_module.retry = retry_module
        urllib3_module = types.ModuleType("urllib3")
        urllib3_module.util = util_module

        fake_requests = types.SimpleNamespace(
            Session=MountSession,
            RequestException=self.FakeRequestException,
        )
        with mock.patch.object(cloud_client, "requests", fake_requests), \
                mock.patch.object(cloud_client, "_SESSION", None), \
                mock.patch.dict(sys.modules, {
                    "requests.adapters": adapters_module,
                    "urllib3": urllib3_module,
                    "urllib3.util": util_module,
                    "urllib3.util.retry": retry_module,
                }):
            session = cloud_client._shared_session()

        self.assertEqual(set(session.mounts), {"http://", "https://"})
        self.assertIsInstance(session.mounts["https://"].max_retries, FakeRetry)
        self.assertEqual(captured.get("total"), config.RETRY_TOTAL)
        self.assertEqual(captured.get("backoff_factor"), config.RETRY_BACKOFF)
        self.assertIn(429, captured.get("status_forcelist", ()))
        self.assertEqual(
            frozenset(captured.get("allowed_methods", ())),
            config.RETRY_ALLOWED_METHODS,
        )

    def test_embed_partial_failure_exposes_completed_batches_and_masks(self):
        """途中バッチ失敗で完了済みバッチ数が機械可読かつマスク済みであること。"""
        secret = "Bearer ZZZPARTIALSECRETZZZ"
        state = {"count": 0}

        def handler(url, **kwargs):
            state["count"] += 1
            if state["count"] == 1:
                return self._response({
                    "data": [
                        {"embedding": [0.1]},
                        {"embedding": [0.2]},
                    ]
                })
            raise self.FakeRequestException("embed failed: " + secret)

        with self._cloud_env(), \
                mock.patch.object(
                    cloud_client, "requests", self._fake_requests(handler)
                ), \
                mock.patch("time.sleep", lambda *_: None):
            with self.assertRaises(cloud_client.CloudUnavailable) as ctx:
                cloud_client.embed(["a", "b", "c"], batch_size=2)

        exc = ctx.exception
        self.assertEqual(getattr(exc, "completed_batches", None), 1)
        message = str(exc)
        self.assertIn("completed_batches=1", message)
        self.assertNotIn("ZZZPARTIALSECRETZZZ", message)
        self.assertIn("<masked>", message)

    def test_accumulate_usage_ignores_non_finite_values(self):
        """NaN / inf は使用量として加算せず、有限値だけを保持すること。"""
        cloud_client._reset_usage()
        cloud_client._accumulate_usage({
            "prompt_tokens": 10,
            "completion_tokens": float("nan"),
            "total_tokens": float("inf"),
            "cost": {"usd": float("-inf"), "eur": 0.25},
        })

        usage = self._read_usage() or {}
        self.assertEqual(usage.get("prompt_tokens"), 10)
        self.assertNotIn("completion_tokens", usage)
        self.assertNotIn("total_tokens", usage)
        self.assertNotIn("cost.usd", usage)
        self.assertEqual(usage.get("cost.eur"), 0.25)
        for value in usage.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.assertTrue(math.isfinite(value))


class CloudConfigGuardTest(unittest.TestCase):
    """config.validate_settings() がクラウド設定を fail-closed で検証すること。

    config.py は読み取り専用のため、mock.patch.dict で環境変数を差し替え、
    mock.patch.object で定数を一時固定して validate_settings() を呼ぶ。
    """

    def _problems(self, **values):
        with _temporary_env(**values), \
                mock.patch.object(config, "EMBED_MODEL", "qwen2.5:3b"):
            return config.validate_settings()

    def test_cloud_enabled_without_embed_model_is_rejected(self):
        problems = self._problems(
            **{
                config.ENV_CLOUD_ENABLED: "1",
                config.ENV_CLOUD_CHAT_ENABLED: "0",
                config.ENV_CLOUD_EMBED_MODEL: None,
            }
        )
        self.assertTrue(
            any(config.ENV_CLOUD_EMBED_MODEL in text for text in problems),
            problems,
        )

    def test_cloud_vector_size_mismatch_is_rejected(self):
        with mock.patch.object(config, "VECTOR_SIZE", 2048):
            problems = self._problems(
                **{
                    config.ENV_CLOUD_ENABLED: "1",
                    config.ENV_CLOUD_CHAT_ENABLED: "0",
                    config.ENV_CLOUD_EMBED_MODEL: "text-embedding-3-small",
                }
            )
        self.assertTrue(
            any("RAG_VECTOR_SIZE" in text for text in problems), problems
        )

    def test_chat_enabled_without_cloud_is_rejected(self):
        problems = self._problems(
            **{
                config.ENV_CLOUD_ENABLED: "0",
                config.ENV_CLOUD_CHAT_ENABLED: "1",
                config.ENV_CLOUD_EMBED_MODEL: None,
            }
        )
        self.assertTrue(
            any(config.ENV_CLOUD_CHAT_ENABLED in text for text in problems),
            problems,
        )

    def test_cloud_disabled_has_no_cloud_problems(self):
        problems = self._problems(
            **{
                config.ENV_CLOUD_ENABLED: "0",
                config.ENV_CLOUD_CHAT_ENABLED: "0",
                config.ENV_CLOUD_EMBED_MODEL: None,
            }
        )
        self.assertFalse(any("RAG_CLOUD" in text for text in problems), problems)

    def test_cloud_enabled_with_explicit_matching_model_is_accepted(self):
        with mock.patch.object(config, "VECTOR_SIZE", 1536):
            problems = self._problems(
                **{
                    config.ENV_CLOUD_ENABLED: "1",
                    config.ENV_CLOUD_CHAT_ENABLED: "0",
                    config.ENV_CLOUD_EMBED_MODEL: "text-embedding-3-small",
                }
            )
        self.assertEqual(problems, [], problems)


class ResolveDimensionsTest(unittest.TestCase):
    """resolve_dimensions の優先順と不正値拒否（実通信なし）。"""

    _UNSET = object()
    SINGULAR = getattr(
        config, "ENV_CLOUD_EMBED_DIMENSION", "RAG_CLOUD_EMBED_DIMENSION"
    )
    PLURAL = getattr(cloud_client, "ENV_DIMENSIONS", "RAG_CLOUD_DIMENSIONS")

    def _resolve(self, explicit=_UNSET, **env):
        values = {self.SINGULAR: None, self.PLURAL: None}
        values.update(env)
        with _temporary_env(**values):
            if explicit is self._UNSET:
                return cloud_client.resolve_dimensions()
            return cloud_client.resolve_dimensions(explicit)

    def test_explicit_dimension_overrides_both_envs(self):
        value = self._resolve(
            768, **{self.SINGULAR: "1024", self.PLURAL: "1536"}
        )
        self.assertEqual(value, 768)

    def test_singular_env_overrides_plural_env(self):
        value = self._resolve(**{self.SINGULAR: "1024", self.PLURAL: "1536"})
        self.assertEqual(value, 1024)

    def test_plural_env_used_when_singular_is_empty(self):
        for empty in ("", "   ", None):
            with self.subTest(empty=empty):
                value = self._resolve(
                    **{self.SINGULAR: empty, self.PLURAL: "1536"}
                )
                self.assertEqual(value, 1536)

    def test_no_env_returns_none_for_model_default(self):
        self.assertIsNone(self._resolve())

    def test_empty_envs_return_none_for_model_default(self):
        self.assertIsNone(
            self._resolve(**{self.SINGULAR: "", self.PLURAL: "  "})
        )

    def test_explicit_invalid_values_are_rejected(self):
        for value in (True, False, 0, -1, "abc", "0", "-3", 1.5):
            with self.subTest(value=value):
                with self.assertRaises(cloud_client.CloudUnavailable):
                    self._resolve(value)

    def test_env_invalid_values_are_rejected(self):
        for name in (self.SINGULAR, self.PLURAL):
            for value in ("0", "-1", "abc", "1.5"):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(cloud_client.CloudUnavailable):
                        self._resolve(**{name: value})

    def test_invalid_value_is_not_leaked_in_message(self):
        secret = "ZZZDIMSECRETZZZ"
        with self.assertRaises(cloud_client.CloudUnavailable) as ctx:
            self._resolve(secret)
        message = str(ctx.exception)
        self.assertNotIn(secret, message)
        self.assertIn("positive integer", message)


class CloudIngestDimensionTest(unittest.TestCase):
    """cloud_ingest の次元解決が config の単数形優先・複数形互換に従うこと。

    実通信なしで環境変数だけを差し替える。RAG_VECTOR_SIZE は毎回 unset し、
    既存のグローバル設定に結果が左右されないようにする。
    """

    SINGULAR = config.ENV_CLOUD_EMBED_DIMENSION
    PLURAL = config.ENV_CLOUD_DIMENSIONS

    @contextlib.contextmanager
    def _dimension_env(self, **values):
        settings = {self.SINGULAR: None, self.PLURAL: None, "RAG_VECTOR_SIZE": None}
        settings.update(values)
        with _temporary_env(**settings):
            yield

    def test_expected_dim_prefers_singular_env_over_plural(self):
        with self._dimension_env(**{self.SINGULAR: "1024", self.PLURAL: "1536"}):
            self.assertEqual(
                cloud_ingest._expected_dim("text-embedding-3-small"), 1024
            )

    def test_expected_dim_accepts_plural_env_as_compat(self):
        with self._dimension_env(**{self.PLURAL: "1536"}):
            self.assertEqual(cloud_ingest._expected_dim("qwen3-embedding-8b"), 1536)

    def test_expected_dim_uses_model_table_without_env(self):
        model = "text-embedding-3-small"
        with self._dimension_env():
            self.assertEqual(
                cloud_ingest._expected_dim(model),
                config.CLOUD_EMBED_DIMENSIONS[model],
            )

    def test_resolve_dimensions_prefers_singular_env_over_plural(self):
        with self._dimension_env(**{self.SINGULAR: "1024", self.PLURAL: "1536"}):
            self.assertEqual(
                cloud_ingest._resolve_dimensions("text-embedding-3-small"), 1024
            )

    def test_resolve_dimensions_accepts_plural_env_as_compat(self):
        with self._dimension_env(**{self.PLURAL: "3072"}):
            self.assertEqual(
                cloud_ingest._resolve_dimensions("gemini-embedding-2"), 3072
            )


class GenerationMaxQueriesTest(unittest.TestCase):
    """cloud_bench.main() の --generation-max-queries 境界を fake で検証する。"""

    def _invoke(self, max_queries):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            for name in ("a.md", "b.md", "c.md"):
                (corpus / name).write_text(
                    "# %s\n本文 %s\n" % (name, name), encoding="utf-8"
                )
            qrels = {
                "schema_version": 1,
                "heldout_queries": [
                    {
                        "id": "q%d" % index,
                        "query": "質問%d" % index,
                        "relevant_sources": ["%s.md" % "abc"[index]],
                    }
                    for index in range(3)
                ],
            }
            qrels_path = root / "qrels.json"
            qrels_path.write_text(
                json.dumps(qrels, ensure_ascii=False), encoding="utf-8"
            )
            output = root / "out.json"
            argv = [
                "cloud_bench.py",
                "--corpus", str(corpus),
                "--qrels", str(qrels_path),
                "--output", str(output),
                "--with-generation",
                "--generation-max-queries", str(max_queries),
            ]
            calls = []

            def fake_embed(texts, **kwargs):
                return [[1.0, 0.0] for _ in texts]

            def fake_generation(record, query, candidates, **kwargs):
                calls.append(query["id"])
                record["rerank_ndcg_at_5"] = 1.0
                return 0.01

            patchers = (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(cloud_client, "enabled", return_value=True),
                mock.patch.object(cloud_client, "embed", fake_embed),
                mock.patch.object(cloud_client, "last_usage", return_value=None),
                mock.patch.object(cloud_bench, "_run_generation", fake_generation),
                mock.patch("sys.stdout", new_callable=io.StringIO),
                mock.patch("sys.stderr", new_callable=io.StringIO),
            )
            with contextlib.ExitStack() as stack:
                for patcher in patchers:
                    stack.enter_context(patcher)
                code = cloud_bench.main()
            payload = (
                json.loads(output.read_text(encoding="utf-8"))
                if output.is_file() else None
            )
            return code, calls, payload

    def test_negative_value_is_rejected_before_cloud_calls(self):
        code, calls, payload = self._invoke(-1)
        self.assertEqual(code, 2)
        self.assertEqual(calls, [])
        self.assertIsNone(payload)

    def test_zero_means_all_queries(self):
        code, calls, payload = self._invoke(0)
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["q0", "q1", "q2"])
        self.assertEqual(payload["summary"]["queries"], 3)

    def test_one_limits_to_first_query(self):
        code, calls, _ = self._invoke(1)
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["q0"])

    def test_value_larger_than_query_count_is_clamped(self):
        code, calls, _ = self._invoke(99)
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["q0", "q1", "q2"])


class CompareCloudLocalRowTest(unittest.TestCase):
    """compare_cloud_local._row が evaluate 系と cloud_bench 系の両形式を読むこと。"""

    def test_evaluate_style_uses_heldout_hybrid_metrics(self):
        data = {
            "generated_at": "2026-11-26T00:00:00Z",
            "qrels_label": "qrels.json",
            "corpus_file_count": 7,
            "embed_model": "local-m",
            "repeat": 2,
            "heldout": {
                "queries": [{"id": "h1"}],
                "summary": {
                    "hybrid": {
                        "recall_at": {"1": 0.5, "10": 0.9},
                        "ndcg_at": {"10": 0.8},
                        "latency_stats": {"avg": 0.12},
                    }
                },
            },
            "summary": {"hybrid": {}},
        }

        row = compare_cloud_local._row("local", "def", data)

        self.assertEqual(row["set_recall_at_10"], 0.9)
        self.assertEqual(row["ndcg_at_10"], 0.8)
        self.assertEqual(row["hit_at_1"], 0.5)
        self.assertEqual(row["latency_sec"], 0.12)
        self.assertEqual(row["source"], "qrels.json")
        self.assertEqual(row["embed_model"], "local-m")
        self.assertEqual(row["corpus_file_count"], 7)
        self.assertEqual(row["queries"], [{"id": "h1"}])
        self.assertEqual(row["repeat"], 2)

    def test_cloud_bench_style_uses_top_level_summary(self):
        data = {
            "generated_at": "2026-11-26T00:00:00Z",
            "corpus": "corpus-subset",
            "chunk_count": 12,
            "queries": [{"id": "q1"}],
            "model": "cloud-m",
            "summary": {
                "set_recall_at_10": 0.7,
                "ndcg_at_10": 0.6,
                "hit_at_1": 0.4,
                "total_search_latency_sec": 0.3,
                "rerank_query_count": 3,
            },
        }

        row = compare_cloud_local._row("cloud", "def", data)

        self.assertEqual(row["source"], "corpus-subset")
        self.assertEqual(row["chunk_count"], 12)
        self.assertEqual(row["queries"], [{"id": "q1"}])
        self.assertEqual(row["embed_model"], "cloud-m")
        self.assertEqual(row["set_recall_at_10"], 0.7)
        self.assertEqual(row["ndcg_at_10"], 0.6)
        self.assertEqual(row["hit_at_1"], 0.4)
        self.assertEqual(row["latency_sec"], 0.3)
        self.assertEqual(row["rerank_query_count"], 3)

    def test_evaluate_style_without_hybrid_mode_still_reads_metrics(self):
        data = {
            "heldout": {
                "summary": {
                    "dense": {
                        "recall_at": {"10": 0.75, "1": 0.25},
                        "ndcg_at": {"10": 0.5},
                    }
                }
            },
            "summary": {},
        }

        row = compare_cloud_local._row("dense-only", "def", data)

        self.assertEqual(row["set_recall_at_10"], 0.75)
        self.assertEqual(row["hit_at_1"], 0.25)
        self.assertEqual(row["ndcg_at_10"], 0.5)


class UsageCollectorTest(unittest.TestCase):
    """cloud_bench の usage 集計（per-call / cumulative / 非有限値）を固定する。"""

    @staticmethod
    def _collector(samples):
        iterator = iter(samples)
        with mock.patch.object(
            cloud_client, "last_usage",
            side_effect=lambda: next(iterator, None),
        ):
            collector = cloud_bench._UsageCollector()
            for _ in samples:
                collector.sample()
        return collector

    def test_cumulative_usage_uses_final_values(self):
        collector = self._collector([
            {
                "calls": 1,
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
            {
                "calls": 2,
                "prompt_tokens": 25,
                "completion_tokens": 5,
                "total_tokens": 30,
            },
        ])

        payload = collector.payload()

        self.assertEqual(payload["calls"], 2)
        self.assertEqual(payload["prompt_tokens"], 25)
        self.assertEqual(payload["completion_tokens"], 5)
        self.assertEqual(payload["total_tokens"], 30)

    def test_per_call_usage_is_summed(self):
        collector = self._collector([
            {"calls": 1, "prompt_tokens": 10, "completion_tokens": 2},
            {"calls": 1, "prompt_tokens": 12, "completion_tokens": 3},
        ])

        payload = collector.payload()

        self.assertEqual(payload["calls"], 2)
        self.assertEqual(payload["prompt_tokens"], 22)
        self.assertEqual(payload["completion_tokens"], 5)
        self.assertEqual(payload["total_tokens"], 27)

    def test_non_finite_usage_is_ignored(self):
        collector = self._collector([
            {
                "calls": 1,
                "prompt_tokens": float("nan"),
                "completion_tokens": 2,
                "total_tokens": float("inf"),
            },
            {"calls": 2, "prompt_tokens": 4, "total_tokens": 6},
        ])

        payload = collector.payload()

        self.assertIsNotNone(payload)
        self.assertEqual(payload["calls"], 2)
        self.assertEqual(payload["completion_tokens"], 2)
        self.assertEqual(payload["prompt_tokens"], 4)
        for key, value in payload.items():
            self.assertTrue(math.isfinite(value), key)

    def test_getter_failure_is_noted_and_does_not_stop(self):
        def broken():
            raise RuntimeError("usage unavailable")

        with mock.patch.object(cloud_client, "last_usage", broken):
            collector = cloud_bench._UsageCollector()
            collector.sample()

        self.assertIsNone(collector.payload())
        self.assertTrue(any("last_usage()" in note for note in collector.notes))


class ExtractRankIdsTest(unittest.TestCase):
    """_extract_rank_ids と _run_generation の rank 境界処理。"""

    @staticmethod
    def _run_with_rank(reply):
        candidates = [
            (0.9, "a.md", "A"),
            (0.8, "b.md", "B"),
            (0.7, "c.md", "C"),
        ]
        record = {}
        query = {"query": "q", "relevant_sources": ["b.md"]}
        with mock.patch.object(cloud_bench, "_chat", return_value=reply):
            cloud_bench._run_generation(record, query, candidates)
        return record

    def test_bool_rank_is_rejected(self):
        for reply in ('{"rank": [true]}', '{"rank": [1, false]}'):
            with self.subTest(reply=reply):
                with self.assertRaises(ValueError):
                    cloud_bench._extract_rank_ids(reply)

    def test_non_integer_rank_is_rejected(self):
        for reply in ('{"rank": [1.5]}', '{"rank": ["1"]}', '{"rank": 1}'):
            with self.subTest(reply=reply):
                with self.assertRaises(ValueError):
                    cloud_bench._extract_rank_ids(reply)

    def test_explanation_text_around_json_is_tolerated(self):
        self.assertEqual(
            cloud_bench._extract_rank_ids('結果: {"rank": [1, 3]} 以上'),
            [1, 3],
        )

    def test_out_of_range_rank_is_ignored(self):
        # relevant は b.md の1件。範囲外(0,99)が除かれ rank=2 → b.md が
        # 最上位に来るため、ideal を qrels 全体(1件)に取ると nDCG=1.0。
        record = self._run_with_rank('{"rank": [0, 99, 2]}')

        self.assertNotIn("generation_error", record)
        self.assertTrue(
            any("rank out of range" in w for w in record["rerank_warnings"])
        )
        self.assertAlmostEqual(record["rerank_ndcg_at_5"], 1.0, places=4)

    def test_duplicate_rank_does_not_double_count(self):
        # rank=2 が3回あってもファイル単位で一意化され、b.md が rank1。
        record = self._run_with_rank('{"rank": [2, 2, 2]}')

        self.assertNotIn("generation_error", record)
        self.assertTrue(
            any("duplicate rank" in w for w in record["rerank_warnings"])
        )
        self.assertAlmostEqual(record["rerank_ndcg_at_5"], 1.0, places=4)

    def test_rank_json_embedded_in_explanatory_text_is_accepted(self):
        # rank の無い JSON が先に出ても、rank を持つ方を拾う（説明文混在）。
        reply = (
            "候補を比較した結果、関連度順は {\"comment\": \"判定\"} "
            "のうえで {\"rank\": [2, 1]} です。以上です。"
        )
        self.assertEqual(cloud_bench._extract_rank_ids(reply), [2, 1])

    def test_bool_rank_embedded_in_explanatory_text_is_rejected(self):
        # 説明文に囲まれていても bool rank は受け付けない。
        reply = '判定結果: {"rank": [true]} 以上です。'
        with self.assertRaises(ValueError):
            cloud_bench._extract_rank_ids(reply)

    def test_rank_diagnostics_reports_out_of_range_ranks(self):
        warnings = cloud_bench._rank_diagnostics([0, 99, 2], 3)
        self.assertEqual(
            warnings, [config.mask_secrets("rank out of range: 0, 99")]
        )

    def test_rank_diagnostics_reports_duplicate_ranks(self):
        warnings = cloud_bench._rank_diagnostics([2, 2, 1, 1], 3)
        self.assertEqual(warnings, [config.mask_secrets("duplicate rank: 2, 1")])

    def test_rank_diagnostics_reports_both_kinds(self):
        warnings = cloud_bench._rank_diagnostics([0, 2, 2, 4], 3)
        self.assertEqual(len(warnings), 2)
        self.assertIn("rank out of range", warnings[0])
        self.assertIn("duplicate rank", warnings[1])

    def test_run_generation_records_range_and_duplicate_warnings(self):
        record = self._run_with_rank('{"rank": [0, 2, 2, 99]}')

        self.assertNotIn("generation_error", record)
        warnings = record["rerank_warnings"]
        self.assertTrue(any("rank out of range" in w for w in warnings))
        self.assertTrue(any("duplicate rank" in w for w in warnings))
        self.assertAlmostEqual(record["rerank_ndcg_at_5"], 1.0, places=4)


if __name__ == "__main__":
    _result = unittest.main(exit=False)
    if not getattr(_result, "result", _result).wasSuccessful():
        sys.exit(1)
    if os.environ.get("RAG_REQUIRE_RUST", "") == "1" and _RUST_SKIP_REASONS:
        print(
            "Rust 同等性テストが skip されました: %d 件 (RAG_REQUIRE_RUST=1)"
            % len(_RUST_SKIP_REASONS)
        )
        sys.exit(1)
