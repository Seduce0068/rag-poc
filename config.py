"""RAG PoC の共通設定。"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import NamedTuple
from urllib.parse import quote

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen2.5:3b")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen2.5:3b")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("RAG_COLLECTION", "rag_poc_docs")

# リモート endpoint を明示オプトインしたときに許可するホストのカンマ区切りリスト。
# 既定は空文字（＝オプトインしてもリモート host は許可しない）。
RAG_ALLOWED_ENDPOINT_HOSTS = os.environ.get("RAG_ALLOWED_ENDPOINT_HOSTS", "")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))
TOP_K = int(os.environ.get("RAG_TOP_K", "10"))
RERANK_MAX_DOCS = int(os.environ.get("RAG_RERANK_MAX_DOCS", "5"))

VECTOR_SIZE = int(os.environ.get("RAG_VECTOR_SIZE", "2048"))
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "300"))
QDRANT_TIMEOUT = float(os.environ.get("QDRANT_TIMEOUT", "300"))
# 回答生成 1 回あたりのタイムアウト（既定 60 秒）。evaluate.py の --with-answer が
# query.generate_answer を呼ぶ際に適用し、OLLAMA_TIMEOUT=300 のまま 60 クエリ分
# 放置すると 60 × 300 秒になる最悪ケースを防ぐ。タイムアウトを長くはしない。
RAG_ANSWER_TIMEOUT = float(os.environ.get("RAG_ANSWER_TIMEOUT", "60"))

# 全 HTTP クライアント（ollama_client / query / ingest）で共有する Retry 設定。
# query.py の _make_session() と同じ値を 1 箇所へ集約する（既存挙動は不変）。
RETRY_TOTAL = 3
RETRY_BACKOFF = 0.5
RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)
RETRY_ALLOWED_METHODS = frozenset({"GET", "POST", "PUT"})

CORPUS_DIR = os.environ.get("RAG_CORPUS_DIR", "passkey-gateway/docs")
QUESTION_MAX_CHARS = int(os.environ.get("RAG_QUESTION_MAX_CHARS", "2000"))
REPO_ROOT = Path(__file__).resolve().parent.parent

# 埋め込みモデルの prefix。RAG_EMBED_PREFIX_QUERY / RAG_EMBED_PREFIX_PASSAGE で
# 上書き可能（既定は空文字＝既存モデルの挙動を変えない）。
EMBED_PREFIX_QUERY = os.environ.get("RAG_EMBED_PREFIX_QUERY", "")
EMBED_PREFIX_PASSAGE = os.environ.get("RAG_EMBED_PREFIX_PASSAGE", "")

# 埋め込み prefix の公開型。tuple を継承するため、既存の 2 タプル
# アンパック（``query_prefix, passage_prefix = embedding_prefixes(...)``）
# や == ("a", "b") による比較はそのまま成立する（後方互換）。
class EmbeddingPrefixes(NamedTuple):
    query: str
    passage: str


# モデル名（部分一致）→ EmbeddingPrefixes。将来のモデル追加はここに追記。
_EMBED_PREFIX_TABLE: dict[str, EmbeddingPrefixes] = {
    "ruri": EmbeddingPrefixes("クエリ: ", "文章: "),
}

# モデル名（部分一致）→ 期待ベクトル次元。
_EXPECTED_VECTOR_SIZE_TABLE: dict[str, int] = {
    "ruri": 768,
    "qwen2.5:3b": 2048,
}


def embedding_prefixes(model: str) -> EmbeddingPrefixes:
    """モデルに対応する EmbeddingPrefixes(query, passage) を返す。

    テーブルに一致するモデルが無い場合は EmbeddingPrefixes("", "")。
    RAG_EMBED_PREFIX_QUERY / RAG_EMBED_PREFIX_PASSAGE が非空なら上書き値を優先する。
    戻り値は tuple 継承なので、既存の 2 タプルアンパックもそのまま動く。
    """
    lowered = model.lower()
    default: EmbeddingPrefixes = EmbeddingPrefixes("", "")
    for key, value in _EMBED_PREFIX_TABLE.items():
        if key in lowered:
            default = value
            break
    return EmbeddingPrefixes(
        EMBED_PREFIX_QUERY or default.query,
        EMBED_PREFIX_PASSAGE or default.passage,
    )


def expected_vector_size(model: str) -> int:
    """モデルに対応する期待ベクトル次元を返す（未知のモデルは VECTOR_SIZE）。"""
    lowered = model.lower()
    for key, value in _EXPECTED_VECTOR_SIZE_TABLE.items():
        if key in lowered:
            return value
    return VECTOR_SIZE


def resolve_corpus_dir() -> Path:
    """コーパスディレクトリを ingest/evaluate 共通の手順で絶対パス化する。

    解決後のパスが REPO_ROOT 配下でない場合は ValueError を送出する
    （外部ディレクトリの誤指定を明示的に拒否するため）。
    """
    path = Path(CORPUS_DIR).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        # リポジトリルート基準を優先（cwd 依存を排除）。
        resolved = (REPO_ROOT / path).resolve()
    root = REPO_ROOT.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            "CORPUS_DIR はリポジトリ配下である必要があります: "
            f"{mask_secrets(str(resolved))}"
        )
    return resolved

SCHEMA_VERSION = 2  # eval-results.json のスキーマバージョン（必須キー欠落で再計測）

# 実験定数を config へ集約（現行の各モジュール定義と同一値）。
# query.py はこれらを import して参照する（単一情報源）。ingest.py のローカル定義は
# 後続対応待ちだが値は同一。いずれも current_settings には含めない。
RRF_K = 60                    # query.py: RRF の k
SCROLL_PAGE_SIZE = 256        # query.py / ingest.py: scroll 1 ページの件数
MAX_SCROLL_PAGES = 100        # query.py / ingest.py: scroll の最大ページ数
RERANK_MAX_CHARS = 1200       # query.py: リランク用に渡す本文の最大文字数
# compare_embeddings.py の正規化時切り詰め文字数。リランク用(1200)とは別値の 2000 を
# 維持するため独立定数とし、compare 側からの参照差し替えは呼び出し側の担当。
COMPARE_MAX_CHARS = 2000      # compare_embeddings.py: 先頭 N 文字で切り詰め
EMBED_BATCH_SIZE = 16         # ingest.py / ollama_client.py: 埋め込みリクエスト 1 件あたりの件数
UPSERT_BATCH_SIZE = 64        # ingest.py: upsert 1 件あたりの点数
DEFAULT_MODELS = ["qwen2.5:3b", "qwen2.5:7b"]
COMPARISON_QUERIES = [
    "パスキー sudo の使い方",
    "DSH 起動トークン",
    "Qdrant 常駐構成",
    "OKF メタデータ",
    "RAG 検索モード比較",
]

_MASK_PATTERNS = (
    re.compile(r"https?://[^\s'\"]+", re.I),
    re.compile(r"file://[^\s'\"]+", re.I),
    re.compile(r"\b(?:Bearer|Basic)\s+\S+", re.I),
    re.compile(r"\b(?:token|api[_-]?key|password|secret)\s*[:=]\s*\S+", re.I),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,}[0-9a-fA-F:]*(?::\d+)?\b"),
    re.compile(r"(?:/[A-Za-z0-9._-]+){2,}"),
)

# provenance には生のパス（相対パスを含む）を残さない。ただし README に記載済みの
# 非機微な既定コーパスは比較条件の可読性のため残す（判断：機微情報ではなく、
# 成果物の再現条件として有用なため）。
_SAFE_PROVENANCE_PATHS = frozenset(
    {
        "passkey-gateway/docs",
        "rag_poc/corpus-alt",
        "rag_poc/corpus-subset",
    }
)
# RESOURCE 等の誤検出を避けるため、区切り（_ または文字列端）で一致させる。
_PROVENANCE_PATH_KEY_RE = re.compile(r"(?:^|_)(?:CORPUS_DIR|SOURCE|PATH)(?:$|_)")


def mask_secrets(text: object) -> str:
    """URL・認証情報・IP・パスを汎用プレースホルダへ置換する（共通）。"""
    value = str(text)
    for pattern in _MASK_PATTERNS:
        value = pattern.sub("<masked>", value)
    return value


def mask_for_provenance(name: str, value: object) -> str:
    """設定の出所ノート用に値をマスクする（モデル名やコレクション名は保持）。

    CORPUS_DIR / SOURCE / PATH 系のキーは相対パスでもマスクする。
    ただし README 記載の非機微な既定コーパスは残す。
    """
    text = str(value)
    upper = name.upper()
    if "URL" in upper:
        return mask_secrets(text)
    if any(token in upper for token in ("TOKEN", "KEY", "SECRET", "PASSWORD")):
        return "<masked>"
    if _PROVENANCE_PATH_KEY_RE.search(upper):
        if text in _SAFE_PROVENANCE_PATHS:
            return text
        return "<masked>"
    # 数値・モデル名・コレクション名はそのまま残す。
    return text

PAYLOAD_TEXT_KEY = "text"
PAYLOAD_SOURCE_KEY = "source"
PAYLOAD_TITLE_KEY = "title"
PAYLOAD_CHUNK_INDEX_KEY = "chunk_index"


def qdrant_collection_path(name: str) -> str:
    """コレクション名を URL パスへ安全に埋め込む（ingest/query 共通）。"""
    return quote(name, safe="")


def _allowed_endpoint_hosts() -> set[str]:
    """RAG_ALLOWED_ENDPOINT_HOSTS（カンマ区切り）を小文字の集合へ正規化する。

    モジュール読み込み後に環境変数を差し替えた場合も反映するため、呼び出し時に
    環境変数を再読する（定数 RAG_ALLOWED_ENDPOINT_HOSTS は初期値の保持用）。
    """
    raw = os.environ.get("RAG_ALLOWED_ENDPOINT_HOSTS", RAG_ALLOWED_ENDPOINT_HOSTS)
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


class EndpointProblem(str):
    """validate_endpoint が返す問題文字列（str 互換 + 構造化された致命性）。

    ``str`` を継承しているため、validate_endpoint の ``list[str]`` 契約、
    既存の文字列演算（``in`` / ``len`` / ``join``）、JSON 出力はそのまま維持される。
    ``fatal`` 属性は呼び出し側が問題文の文字列一致で判定するためのものではなく、
    config が検証時に付与した構造化情報である（既定 True = fail-closed）。
    """

    fatal: bool

    def __new__(cls, text: str, *, fatal: bool = True) -> "EndpointProblem":
        instance = super().__new__(cls, text)
        instance.fatal = bool(fatal)
        return instance


def is_fatal_endpoint_problem(problem: str) -> bool:
    """接続先検証の問題が致命（オプトインでも警告へ格下げ不可）かを返す。

    判定は ``EndpointProblem.fatal``（validate_endpoint が付与した構造化情報）
    だけを参照し、問題文の文字列一致には依存しない。未知の文字列や metadata を
    欠くオブジェクトは fail-closed で致命として扱う。したがって caller は
    ``is_fatal_endpoint_problem(problem)`` が真なら環境変数に関係なく停止する。
    """
    flag = getattr(problem, "fatal", None)
    return flag is not False


def validate_endpoint(
    url: str, label: str, allow_remote: bool | str | None = None
) -> list[str]:
    """Ollama/Qdrant の接続先を検証し、致命的な問題だけを返す。

    既定はループバックのみ許可。リモート host は次をすべて満たす場合だけ
    「警告に格下げ」（＝致命問題として返さない）：

    * ``RAG_ALLOW_REMOTE_ENDPOINTS=1`` の明示オプトイン（第三引数の
      ``allow_remote`` が指定された場合はその値を優先。後方互換用）
    * host が ``RAG_ALLOWED_ENDPOINT_HOSTS``（カンマ区切り）に含まれる
    * scheme が https

    URL の解析失敗と不正 scheme はオプトイン／許可リストに関係なく fatal。
    平文 HTTP の外部送信も fatal のまま。返す各問題は ``EndpointProblem``
    （``str``  subclass）で、``fatal=True`` の構造化情報を持つ。呼び出し側は
    ``is_fatal_endpoint_problem()`` で判定し、文字列一致で格下げしてはならない。
    """
    from urllib.parse import urlparse

    problems: list[str] = []
    try:
        parsed = urlparse(url)
        # 不正な port は urlparse ではなく .port で ValueError になる。
        port = parsed.port
    except ValueError:
        return [EndpointProblem(f"{label} の URL が不正です: {mask_secrets(url)}")]

    if parsed.scheme not in ("http", "https"):
        # 不正 scheme はオプトインでも警告へ格下げしない（常に fatal）。
        problems.append(
            EndpointProblem(f"{label} は http/https である必要があります: {parsed.scheme!r}")
        )
        return problems

    if allow_remote is None:
        opt_in = os.environ.get("RAG_ALLOW_REMOTE_ENDPOINTS") == "1"
    elif isinstance(allow_remote, str):
        opt_in = allow_remote == "1"
    else:
        opt_in = bool(allow_remote)

    host = (parsed.hostname or "").lower()
    loopback = host in ("127.0.0.1", "::1", "localhost")
    if not loopback:
        allowed = _allowed_endpoint_hosts()
        host_allowed = host in allowed or (
            port is not None and f"{host}:{port}" in allowed
        )
        downgraded = opt_in and host_allowed and parsed.scheme == "https"
        if not downgraded:
            # ここに到達した時点で (a) オプトイン無し、(b) 許可リスト不一致、
            # (c) 平文 HTTP のいずれか／複数が成立するため、すべて致命として
            # 構造化情報を付与する。caller は文字列を見て格下げしない。
            problems.append(
                EndpointProblem(
                    f"{label} はローカル限定です。リモートを使う場合は明示的なオプトインが必要: {host}"
                )
            )
            if parsed.scheme == "http":
                problems.append(
                    EndpointProblem(f"{label} が平文 HTTP で外部ホストへ送信されます")
                )
    return problems


def current_settings() -> dict[str, object]:
    """環境変数から現在の設定スナップショットを作る（遅延評価）。"""
    def env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return default

    return {
        "CHUNK_SIZE": env_int("CHUNK_SIZE", 800),
        "CHUNK_OVERLAP": env_int("CHUNK_OVERLAP", 100),
        "TOP_K": env_int("RAG_TOP_K", 10),
        "RERANK_MAX_DOCS": env_int("RAG_RERANK_MAX_DOCS", 5),
        "VECTOR_SIZE": env_int("RAG_VECTOR_SIZE", 2048),
        "QUESTION_MAX_CHARS": env_int("RAG_QUESTION_MAX_CHARS", 2000),
        "OLLAMA_URL": os.environ.get("OLLAMA_URL", OLLAMA_URL),
        "QDRANT_URL": os.environ.get("QDRANT_URL", QDRANT_URL),
        "COLLECTION": os.environ.get("RAG_COLLECTION", COLLECTION),
        "CORPUS_DIR": os.environ.get("RAG_CORPUS_DIR", CORPUS_DIR),
    }


def validate_settings() -> list[str]:
    """起動時に検証し、問題を文字列で返す（空なら正常）。"""
    problems: list[str] = []
    if CHUNK_SIZE <= 0:
        problems.append("CHUNK_SIZE は正の整数である必要があります")
    if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        problems.append("CHUNK_OVERLAP は 0 以上 CHUNK_SIZE 未満である必要があります")
    if TOP_K <= 0:
        problems.append("RAG_TOP_K は正の整数である必要があります")
    if RERANK_MAX_DOCS < 0:
        problems.append("RAG_RERANK_MAX_DOCS は 0 以上である必要があります")
    if VECTOR_SIZE <= 0:
        problems.append("RAG_VECTOR_SIZE は正の整数である必要があります")
    if "ruri" in EMBED_MODEL.lower() and VECTOR_SIZE != 768:
        problems.append(
            f"EMBED_MODEL が ruri の場合 RAG_VECTOR_SIZE は 768 である必要があります（現在: {VECTOR_SIZE}）"
        )
    # Ruri は 512 トークン context。既定 800 のまま指定するとエラーになる。
    if "ruri" in EMBED_MODEL.lower() and CHUNK_SIZE > 512:
        problems.append(
            f"EMBED_MODEL が ruri の場合 CHUNK_SIZE は 512 以下である必要があります"
            f"（Ruri は 512 トークン context、現在: {CHUNK_SIZE}）"
        )
    return problems


def validate_question_length(text: str) -> list[str]:
    """質問長を検証し、問題を文字列で返す（空なら正常）。

    query.main() の既存チェックと同じ QUESTION_MAX_CHARS を閾値にする。
    """
    problems: list[str] = []
    length = len(text)
    if length > QUESTION_MAX_CHARS:
        problems.append(f"質問が長すぎます（{length} > {QUESTION_MAX_CHARS} 文字）")
    return problems
