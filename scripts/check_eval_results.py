"""後方互換用の薄いラッパー。検証ロジックは Rust (ragcore) 側にある。

    python3 scripts/check_eval_results.py [paths...]

パスが省略された場合はカレントツリーの eval-results-*.json を列挙する。
"""
from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rust_bridge  # noqa: E402

if __name__ == "__main__":
    sys.exit(rust_bridge.main(["check_eval_results", *sys.argv[1:]]))
