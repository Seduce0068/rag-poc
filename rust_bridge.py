"""rag_poc/ragcore (Rust) CLI への薄いブリッジ。

stdin JSON で op を渡し stdout JSON を返す CLI を subprocess 経由で呼ぶだけの
薄い層。Rust が無い／失敗した場合は :class:`RustUnavailable` を送出し、
呼び出し側（compare_embeddings.py）が Python 実装へフォールバックする。
例外メッセージには生パスや認証情報を含めない（config.mask_secrets でマスク）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Sequence

import config

# 既定バイナリ（config.REPO_ROOT 基準）
DEFAULT_BINARY = Path("rag_poc/ragcore/target/debug/ragcore")
# 上書き環境変数（絶対／相対いずれも可）
ENV_BINARY = "RAG_RUST_BIN"
# Rust CLI 専用タイムアウト（HTTP 用の OLLAMA_TIMEOUT は流用しない）
ENV_TIMEOUT = "RAG_RUST_TIMEOUT"
DEFAULT_TIMEOUT = 30.0
# available() の probe はさらに軽く済ませる。
_PROBE_TIMEOUT = 5.0

# Rust CLI 出力をメモリへ読み込む上限（文字数）。暴走バイナリの無制限な
# stdout/stderr が OOM・ログ膨張・notes への巨大文字列伝播を起こさないようにする。
MAX_OUTPUT_CHARS = 1_000_000
# UTF-8 の最悪ケース（1 文字 4 バイト）＋切り詰め判定用の 1 バイト。
_MAX_OUTPUT_BYTES = MAX_OUTPUT_CHARS * 4
_TRUNCATED_MARKER = "... (truncated)"


def _timeout() -> float:
    """RAG_RUST_TIMEOUT（秒）を読む。不正値は既定値へフォールバック。"""
    try:
        return max(1.0, float(os.environ.get(ENV_TIMEOUT, str(DEFAULT_TIMEOUT))))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT


class RustUnavailable(Exception):
    """Rust bridge が使えない場合の例外（理由はマスク済みメッセージ）。"""


# 既定パスが未ビルドのときにだけ添える固定のヒント。動的な外部入力は一切含めず、
# config.mask_secrets を通しても書き換わらない安全な定型文として扱う。
_DEFAULT_BUILD_HINT = (
    "Rust バイナリが未ビルド: rag_poc/ragcore/target/debug/ragcore"
    "（cargo build --manifest-path rag_poc/ragcore/Cargo.toml を実行）"
)


def _default_binary_hint(binary: Path) -> str | None:
    """既定パス由来の失敗ならビルド手順のヒントを返す（上書き時は None）。"""
    override = os.environ.get(ENV_BINARY)
    if override:
        return None
    try:
        expected = (config.REPO_ROOT / DEFAULT_BINARY).resolve()
        return _DEFAULT_BUILD_HINT if binary.resolve() == expected else None
    except OSError:
        return None


def _fail(reason: object, hint: str | None = None) -> RustUnavailable:
    """マスク済みの理由を持つ RustUnavailable を組み立てる。

    ``reason`` は必ず ``config.mask_secrets`` を通す（動的なパス・URL・認証情報を
    漏らさない）。``hint`` は固定の安全な文字列だけを渡し、マスク後に連結する
    （マスク前に連結すると相対パスまで ``<masked>`` に潰れてビルド方法が消える）。
    """
    masked = config.mask_secrets(reason)
    if hint:
        masked = f"{masked} {hint}"
    return RustUnavailable(f"Rust bridge unavailable: {masked}")


def _ensure_usable(binary: Path) -> None:
    """実行ビット（および通常ファイル）を確認し、理由付きで拒否する。"""
    hint = _default_binary_hint(binary)
    if not os.access(binary, os.X_OK):
        state = "not found" if not binary.exists() else "not executable"
        raise _fail(f"Rust binary {state}: {binary}", hint)
    if not binary.is_file():
        raise _fail(f"Rust binary is not a regular file: {binary}", hint)


def rust_binary_path() -> Path:
    """Rust バイナリのパスを返す。

    - RAG_RUST_BIN が設定されていれば最優先（絶対パスはそのまま、
      相対パスは config.REPO_ROOT 基準で解決）。REPO_ROOT 配下でなければ拒否。
    - 未設定なら、まず rag_poc リポジトリ直下の
      ragcore/target/debug/ragcore（CI checkout 想定）、次に従来の
      REPO_ROOT/rag_poc/ragcore/target/debug/ragcore（workspace 想定）を試す。
    """
    override = os.environ.get(ENV_BINARY)
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = config.REPO_ROOT / path
        resolved = path.resolve()
        if not resolved.is_relative_to(config.REPO_ROOT.resolve()):
            raise ValueError(
                f"{ENV_BINARY} はリポジトリ配下である必要があります: "
                f"{config.mask_secrets(str(resolved))}"
            )
        return resolved
    # CI（checkout 直下に ragcore）と workspace（rag_poc/ 配下）の両レイアウトを許可。
    candidates = (
        Path(__file__).resolve().parent / "ragcore/target/debug/ragcore",
        (config.REPO_ROOT / DEFAULT_BINARY).resolve(),
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    # 未ビルド時は Diagnostics に分かりやすい方を返す（上書き推奨）。
    return candidates[0].resolve()


def _read_bounded(stream: BinaryIO) -> tuple[str, bool]:
    """一時ファイルの先頭だけを読み、テキストと切り詰め有無を返す。

    全量をメモリへ載せず、最大 MAX_OUTPUT_CHARS 文字だけを保持する。
    """
    stream.seek(0)
    raw = stream.read(_MAX_OUTPUT_BYTES + 1)
    truncated = len(raw) > _MAX_OUTPUT_BYTES
    text = raw[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS]
        truncated = True
    return text, truncated


def run_op(op: str, payload: dict) -> Any:
    """op と payload を Rust CLI へ渡し、結果値を返す。

    stdout/stderr は一時ファイルへリダイレクトし、メモリへは
    MAX_OUTPUT_CHARS 文字までしか読み込まない（暴走出力での OOM と
    ログ膨張を防ぐ）。終了コード 0 なら上限適用済みの stdout を
    JSON 解析して返す。起動失敗・タイムアウト・非 0 終了・JSON
    解析失敗・切り詰めはすべて RustUnavailable に変換する。
    """
    binary = rust_binary_path()
    _ensure_usable(binary)
    # op は予約キー。payload を先に展開して op を最後に置くことで、
    # payload 側に op が混入しても予約キーを上書きできないようにする。
    request = json.dumps({**payload, "op": op})

    # capture_output を外し、巨大出力を一時ファイルへ退避する。
    with tempfile.TemporaryFile(mode="w+b") as stdout_file, \
            tempfile.TemporaryFile(mode="w+b") as stderr_file:
        try:
            completed = subprocess.run(
                [str(binary)],
                input=request.encode("utf-8"),
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=_timeout(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            # マスク済みメッセージを渡しつつ、原因種別はチェーンで残す。
            # TimeoutExpired は stdout/stderr を持たないが理由は作れる。
            # 既定パスが起動できない場合もビルド手順を添える（動的情報はマスク済み）。
            raise _fail(exc, _default_binary_hint(binary)) from exc
        stdout, stdout_truncated = _read_bounded(stdout_file)
        stderr, stderr_truncated = _read_bounded(stderr_file)

    # 上限超過は JSON 解析前に失敗させ、マーカーを理由へ入れて
    # 呼び出し側の notes で気付けるようにする（巨大文字列は渡さない）。
    if stdout_truncated:
        raise _fail(f"stdout exceeded {MAX_OUTPUT_CHARS} chars {_TRUNCATED_MARKER}")
    if stderr_truncated:
        raise _fail(f"stderr exceeded {MAX_OUTPUT_CHARS} chars {_TRUNCATED_MARKER}")

    # CLI はエラー JSON を stdout に出す契約。stderr より先に解析する。
    try:
        value = json.loads(stdout) if stdout.strip() else None
    except (ValueError, TypeError) as exc:
        if completed.returncode == 0:
            raise _fail(exc) from exc
        value = None

    if isinstance(value, dict) and "error" in value:
        raise _fail(value["error"])

    if completed.returncode != 0:
        detail = (stderr or "").strip() or f"exit code {completed.returncode}"
        raise _fail(detail)

    if value is None:
        raise _fail("empty stdout")

    return value


def cosine_batch(
    pairs: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> list[float]:
    """(a, b) ペア全体を 1 回の cosine_batch op で計算し、生値を返す。

    Rust が無い／失敗した場合は RustUnavailable をそのまま送出する
    （呼び出し側がフォールバックする）。丸めは行わない。
    """
    request_pairs = [[list(a), list(b)] for a, b in pairs]
    result = run_op("cosine_batch", {"pairs": request_pairs})
    if not isinstance(result, list) or len(result) != len(request_pairs):
        raise _fail("cosine_batch returned unexpected result")
    try:
        return [float(value) for value in result]
    except (TypeError, ValueError) as exc:
        raise _fail(exc) from exc


def available() -> bool:
    """バイナリ存在と実 op（cosine）の応答を確認する。例外は False。"""
    try:
        binary = rust_binary_path()
    except Exception:  # noqa: BLE001 - 上書きパス不正も探測では False
        return False
    if not binary.is_file():
        return False
    try:
        result = run_op("cosine", {"a": [1.0], "b": [1.0]})
        return result == 1.0
    except Exception:  # noqa: BLE001 - 探測なので何でも False
        return False


def check_eval_results(paths: Sequence[str | Path]) -> dict:
    """eval-results-*.json の検証を Rust op へ委譲し、サマリ dict を返す。

    Rust 側へ渡すのはファイルパスのみ。JSON の内容（値）は一切渡さない。
    タイムアウトと出力上限は :func:`run_op` の既存契約をそのまま使う。
    """
    request_paths = [str(path) for path in paths]
    result = run_op("check_eval_results", {"paths": request_paths})
    if not isinstance(result, dict) or "ok" not in result:
        raise _fail("check_eval_results returned unexpected result")
    return result


def chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Rust op で文字ウィンドウ分割する。失敗時は RustUnavailable を送出する。"""
    result = run_op(
        "chunk", {"text": text, "chunk_size": chunk_size, "overlap": overlap}
    )
    if not isinstance(result, list) or not all(isinstance(item, str) for item in result):
        raise _fail("chunk returned unexpected result")
    return result


def point_id(source: str, chunk_index: int) -> int:
    """安定 ID を Rust op で計算する。失敗時は RustUnavailable を送出する。"""
    result = run_op("point_id", {"source": source, "chunk_index": chunk_index})
    if not isinstance(result, int):
        raise _fail("point_id returned unexpected result")
    return result


def fingerprint(paths: Sequence[tuple[str, str]]) -> dict:
    """(path, content) の組からコーパス fingerprint を Rust op で計算する。"""
    request = [{"path": path, "content": content} for path, content in paths]
    result = run_op("fingerprint", {"paths": request})
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("count"), int)
        or not isinstance(result.get("sha256"), str)
    ):
        raise _fail("fingerprint returned unexpected result")
    return result


def recall(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    """query-level hit@k を Rust op で計算する。"""
    result = run_op(
        "recall",
        {
            "relevant": list(relevant),
            "retrieved": list(retrieved),
            "k": k,
        },
    )
    if not isinstance(result, (int, float)):
        raise _fail("recall returned unexpected result")
    return float(result)


def ndcg(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    """nDCG@k を Rust op で計算する（CLI 側で 6 桁丸め済み）。"""
    result = run_op(
        "ndcg",
        {
            "relevant": list(relevant),
            "retrieved": list(retrieved),
            "k": k,
        },
    )
    if not isinstance(result, (int, float)):
        raise _fail("ndcg returned unexpected result")
    return float(result)


def latency(samples: Sequence[float]) -> dict:
    """レイテンシ統計を Rust op で計算する。"""
    result = run_op("latency", {"samples": list(samples)})
    required = ("count", "avg", "median", "min", "max", "stdev")
    if not isinstance(result, dict) or any(key not in result for key in required):
        raise _fail("latency returned unexpected result")
    return result


def rrf_fuse(rankings: Sequence[Sequence[dict]], top_k: int) -> list[dict]:
    """RRF(k=60) で順位を統合する。失敗時は RustUnavailable を送出する。"""
    result = run_op("rrf_fuse", {"rankings": rankings, "top_k": top_k})
    if not isinstance(result, list) or not all(isinstance(item, dict) for item in result):
        raise _fail("rrf_fuse returned unexpected result")
    return result


def _discover_eval_results(root: Path) -> list[Path]:
    """指定ツリーの公開対象 eval-results*.json をファイル名順に列挙する。

    - eval-results.json（主成果物）も対象に含める。
    - eval-results-cloud-answer.json はローカル検証用で answer を含むため除外。
    """
    excluded = {"eval-results-cloud-answer.json"}
    return sorted(
        path
        for path in root.rglob("eval-results*.json")
        if path.name not in excluded
    )


def main(argv: Sequence[str] | None = None) -> int:
    """`python3 rust_bridge.py check_eval_results [paths...]` の入口。"""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "check_eval_results":
        print(
            "usage: python3 rust_bridge.py check_eval_results [paths...]",
            file=sys.stderr,
        )
        return 2

    paths = [Path(value) for value in args[1:]] or _discover_eval_results(Path("."))
    try:
        result = check_eval_results(paths)
    except RustUnavailable as exc:
        # マスク済みの理由のみ。JSON の値は出さない。
        print("JSON gate failed")
        print(str(exc), file=sys.stderr)
        return 1

    diagnostics = result.get("diagnostics", [])
    if isinstance(diagnostics, list):
        for line in diagnostics:
            if isinstance(line, str):
                # Rust 側が組み立てるのはファイル名とキー名だけ。
                print(line, file=sys.stderr)
    ok = bool(result.get("ok"))
    print("JSON gate OK" if ok else "JSON gate failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
