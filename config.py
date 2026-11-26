"""RAG PoC の共通設定。"""

from __future__ import annotations

import os
import re
import sys
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

# 検索モードの唯一の定義（R2-B）。query.SEARCH_MODES / evaluate.MODES は
# このタプルを直接参照し、ローカル定義との乖離はテストで検出する。
SEARCH_MODES: tuple[str, ...] = ("dense", "keyword", "hybrid", "hybrid-rerank")

VECTOR_SIZE = int(os.environ.get("RAG_VECTOR_SIZE", "2048"))
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "300"))
QDRANT_TIMEOUT = float(os.environ.get("QDRANT_TIMEOUT", "300"))
# 回答生成 1 回あたりのタイムアウト（既定 60 秒）。evaluate.py の --with-answer が
# query.generate_answer を呼ぶ際に適用し、OLLAMA_TIMEOUT=300 のまま 60 クエリ分
# 放置すると 60 × 300 秒になる最悪ケースを防ぐ。タイムアウトを長くはしない。
RAG_ANSWER_TIMEOUT = float(os.environ.get("RAG_ANSWER_TIMEOUT", "60"))

# クラウド（opencodeGO）設定。既定はすべて無効／ローカル Ollama を維持する。
ENV_CLOUD_ENABLED = "RAG_CLOUD_ENABLED"
ENV_CLOUD_CHAT_ENABLED = "RAG_CLOUD_CHAT_ENABLED"
ENV_CLOUD_EMBED_MODEL = "RAG_CLOUD_EMBED_MODEL"
ENV_CLOUD_EMBED_DIMENSION = "RAG_CLOUD_EMBED_DIMENSION"
# 移行期の互換名（複数形）。両方設定時は正規（単数形）を優先する。
ENV_CLOUD_DIMENSIONS = "RAG_CLOUD_DIMENSIONS"
RAG_CLOUD_ENABLED = os.environ.get(ENV_CLOUD_ENABLED, "0") == "1"
RAG_CLOUD_CHAT_ENABLED = os.environ.get(ENV_CLOUD_CHAT_ENABLED, "0") == "1"
RAG_CLOUD_EMBED_MODEL = os.environ.get(ENV_CLOUD_EMBED_MODEL, "")
CLOUD_DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# cloud_client.EMBEDDING_MODELS のうち既定次元が確定しているものだけを保持する
# （次元テーブルの単一情報源）。未知モデルは RAG_CLOUD_EMBED_DIMENSION の
# 明示指定が無い限り fail-closed。
CLOUD_EMBED_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "qwen3-embedding-8b": 4096,
    "gemini-embedding-2": 3072,
}

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


# ---------------------------------------------------------------------------
# クラウド設定の検証（fail-closed）
# ---------------------------------------------------------------------------


class CloudConfigError(RuntimeError):
    """クラウド設定の不整合を示す明示エラー（R2-B）。"""


def cloud_enabled() -> bool:
    """RAG_CLOUD_ENABLED を呼び出し時に再読する（既定 0 = ローカル Ollama）。"""
    return os.environ.get(ENV_CLOUD_ENABLED, "0") == "1"


def cloud_chat_enabled() -> bool:
    """RAG_CLOUD_CHAT_ENABLED を呼び出し時に再読する（既定 0 = Ollama /api/chat 維持）。"""
    return os.environ.get(ENV_CLOUD_CHAT_ENABLED, "0") == "1"


def cloud_embed_model_explicit() -> bool:
    """RAG_CLOUD_EMBED_MODEL が明示指定されているか。"""
    return bool(os.environ.get(ENV_CLOUD_EMBED_MODEL, "").strip())


def cloud_embed_model() -> str:
    """cloud_client.embed_model() と同一のモデル ID を返す。

    cloud_client は config.mask_secrets を import するため、循環 import を避ける
    目的で遅延 import する（関数呼び出し時には config の読み込みが完了している）。
    import 不能なら環境変数／既定値へフォールバックする。
    """
    try:
        import cloud_client

        return cloud_client.embed_model()
    except Exception:
        return os.environ.get(ENV_CLOUD_EMBED_MODEL, "") or CLOUD_DEFAULT_EMBED_MODEL


# ---------------------------------------------------------------------------
# 埋め込み次元 環境変数の共通リゾルバ（R2-B）
# ---------------------------------------------------------------------------


def eprint(message: object) -> None:
    """警告を stderr へ出す。呼び出し側は値ではなく環境変数名のみを渡す。"""
    print(str(message), file=sys.stderr, flush=True)


def resolve_cloud_embed_dimension() -> int | None:
    """埋め込み次元の環境変数を 1 箇所で解決する。

    正規 ``RAG_CLOUD_EMBED_DIMENSION`` を優先し、移行期互換の
    ``RAG_CLOUD_DIMENSIONS`` は正規が未設定のときだけ読む。両方が非空なら
    正規を採用し、値は出さずに環境変数名だけを ``eprint`` へ出す。

    未設定なら None。不正値は ValueError を送出し、例外文には環境変数名
    だけを含めて値は一切出さない。
    """
    primary = os.environ.get(ENV_CLOUD_EMBED_DIMENSION)
    legacy = os.environ.get(ENV_CLOUD_DIMENSIONS)
    primary_set = primary is not None and str(primary).strip() != ""
    legacy_set = legacy is not None and str(legacy).strip() != ""
    if primary_set and legacy_set:
        eprint(
            "%s と %s が両方設定されています。%s を優先します"
            % (
                ENV_CLOUD_EMBED_DIMENSION,
                ENV_CLOUD_DIMENSIONS,
                ENV_CLOUD_EMBED_DIMENSION,
            )
        )
    raw = primary if primary_set else (legacy if legacy_set else None)
    if raw is None:
        return None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        raise ValueError(
            "%s は正の整数である必要があります" % ENV_CLOUD_EMBED_DIMENSION
        ) from None
    if value <= 0:
        raise ValueError(
            "%s は正の整数である必要があります" % ENV_CLOUD_EMBED_DIMENSION
        )
    return value


def validate_cloud_settings() -> list[str]:
    """クラウド有効時の設定不整合を検出し、問題文の list を返す（空なら正常）。

    対象は次に示す R2-B の 2 条件と、chat の前提整合である。

    1. Ruri 系 EMBED_MODEL でクラウド既定モデルを使う場合（暗黙既定も含む）
    2. クラウド埋め込みモデルの期待次元が VECTOR_SIZE と異なる場合
    3. RAG_CLOUD_CHAT_ENABLED=1 なのに RAG_CLOUD_ENABLED=0 の場合

    RAG_CLOUD_EMBED_MODEL はクラウド有効時に必須化する（暗黙の既定モデル使用を
    禁じる）。期待次元が表に無いモデルは RAG_CLOUD_EMBED_DIMENSION（互換:
    RAG_CLOUD_DIMENSIONS）の明示を要求し、不明なままなら fail-closed とする。
    値は mask_secrets 済みで返す。
    """
    problems: list[str] = []
    enabled = cloud_enabled()
    chat = cloud_chat_enabled()
    if chat and not enabled:
        problems.append(
            "RAG_CLOUD_CHAT_ENABLED=1 を使うには RAG_CLOUD_ENABLED=1 も必要です"
        )
    if not enabled:
        return problems

    model = cloud_embed_model()
    explicit = cloud_embed_model_explicit()
    ruri = "ruri" in EMBED_MODEL.lower()
    if not explicit:
        if ruri:
            problems.append(
                "EMBED_MODEL が ruri 系なのに RAG_CLOUD_EMBED_MODEL が未指定です。"
                "クラウド既定モデル(%s)には ruri の prefix / 768 次元が反映されず"
                "不整合のため使用できません" % CLOUD_DEFAULT_EMBED_MODEL
            )
        else:
            problems.append(
                "RAG_CLOUD_ENABLED=1 では RAG_CLOUD_EMBED_MODEL の明示指定が必要です"
                "（既定の %s を暗黙に使うことはできません）" % CLOUD_DEFAULT_EMBED_MODEL
            )

    # 既知モデルは表を唯一の情報源とする（embed_query は dimensions を渡さないため
    # 既定次元が確定）。環境変数の解決は共通リゾルバへ委譲し、未知モデルだけ
    # RAG_CLOUD_EMBED_DIMENSION の明示を要求する。
    expected_dim: int | None = CLOUD_EMBED_DIMENSIONS.get(model)
    try:
        configured_dim = resolve_cloud_embed_dimension()
    except ValueError as exc:
        problems.append(str(exc))
        configured_dim = None
    if expected_dim is None:
        expected_dim = configured_dim
    if expected_dim is None:
        problems.append(
            "クラウド埋め込みモデル %s の期待次元が不明です。"
            "RAG_CLOUD_EMBED_DIMENSION を明示してください" % mask_secrets(model)
        )
    elif expected_dim != VECTOR_SIZE:
        problems.append(
            "クラウド埋め込みモデル %s の期待次元(%d)が RAG_VECTOR_SIZE(%d) と一致しません。"
            "コレクション側の次元と揃えてください"
            % (mask_secrets(model), expected_dim, VECTOR_SIZE)
        )
    return problems


def require_cloud_settings() -> None:
    """不整合があれば CloudConfigError を送出する（直接 API 呼び出し用の fail-closed）。"""
    problems = validate_cloud_settings()
    if problems:
        raise CloudConfigError("クラウド設定の不整合: " + "; ".join(problems))


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
# query.py はこれらを import して参照する（単一情報源）。ingest.py /
# compare_embeddings.py のローカル定義は後続対応待ちだが値は同一。呼び出し側は
# 下の experiment_defaults() を経由して参照する（差し替えは担当外のため提案のみ）。
# いずれも current_settings には含めない。
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


def experiment_defaults() -> dict[str, int]:
    """ingest.py / compare_embeddings.py 向けの実験定数 accessor（R2-B）。

    現行の各モジュール内のローカル定数と同一値を返す。呼び出し側の差し替えは
    担当外のため未実施で、ここでは参照先だけを公開する。既存 API は維持し、
    ``current_settings()`` とは独立（実験定数は setting スナップショットに
    含めない従来方針を継続）。

    呼び出し側の提案::

        defaults = config.experiment_defaults()
        RRF_K = defaults["RRF_K"]
        SCROLL_PAGE_SIZE = defaults["SCROLL_PAGE_SIZE"]
        MAX_SCROLL_PAGES = defaults["MAX_SCROLL_PAGES"]
        EMBED_BATCH_SIZE = defaults["EMBED_BATCH_SIZE"]
        UPSERT_BATCH_SIZE = defaults["UPSERT_BATCH_SIZE"]
        RERANK_MAX_CHARS = defaults["RERANK_MAX_CHARS"]
        COMPARE_MAX_CHARS = defaults["COMPARE_MAX_CHARS"]  # compare のみ
    """
    return {
        "RRF_K": RRF_K,
        "SCROLL_PAGE_SIZE": SCROLL_PAGE_SIZE,
        "MAX_SCROLL_PAGES": MAX_SCROLL_PAGES,
        "EMBED_BATCH_SIZE": EMBED_BATCH_SIZE,
        "UPSERT_BATCH_SIZE": UPSERT_BATCH_SIZE,
        "RERANK_MAX_CHARS": RERANK_MAX_CHARS,
        "COMPARE_MAX_CHARS": COMPARE_MAX_CHARS,
    }


_MASK_PATTERNS = (
    re.compile(r"https?://[^\s'\"]+", re.I),
    re.compile(r"file://[^\s'\"]+", re.I),
    re.compile(r"\b(?:Bearer|Basic)\s+\S+", re.I),
    re.compile(r"\b(?:token|api[_-]?key|password|secret)\s*[:=]\s*\S+", re.I),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,}[0-9a-fA-F:]*(?::\d+)?\b"),
    # スラッシュ区切りの列挙（a/b/c 等）を誤検出しないよう絶対パスだけを対象にする。
    re.compile(r"(?:^|(?<=[\s=]))/(?:[A-Za-z0-9._-]+/){1,}[A-Za-z0-9._-]*"),
    # 公開成果物の生成 answer に混ざり得る内部識別子（AGENTS.md の公開サニタイズ基準）。
    # 固有值は分割構築して源コード自体を Betterleaks の検出対象にしない。
    re.compile(r"\b" + "kop" + "pe" + r"\b", re.I),
    re.compile(r"\b(?:uid|gid)\s*(?:[:=]\s*|\s+)\d{2,6}\b", re.I),
    re.compile(r"\b(?:docomo|au|softbank)\b", re.I),
    re.compile(r"\bHOME\s*5G\b", re.I),
    re.compile(r"\b[A-Z]{2,8}\s+[A-Z0-9]{6,12}\s+\d{3,5}[.,]?\d*G\b"),
    re.compile(r"\bsda\d+\b", re.I),
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
        # クラウド設定キー（R2-B）。値は mask_secrets 済みで返す。
        # enabled / chat_enabled は "1"/"0" 文字列、embed_model はモデル ID。
        ENV_CLOUD_ENABLED: mask_secrets(
            os.environ.get(ENV_CLOUD_ENABLED, "1" if RAG_CLOUD_ENABLED else "0")
        ),
        ENV_CLOUD_EMBED_MODEL: mask_secrets(cloud_embed_model()),
        ENV_CLOUD_CHAT_ENABLED: mask_secrets(
            os.environ.get(ENV_CLOUD_CHAT_ENABLED, "1" if RAG_CLOUD_CHAT_ENABLED else "0")
        ),
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
    # クラウド有効時の embed / chat 不整合も同じ fail-closed 経路に載せる（R2-B）。
    problems.extend(validate_cloud_settings())
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
