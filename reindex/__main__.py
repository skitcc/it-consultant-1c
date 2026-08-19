"""python -m reindex"""

from __future__ import annotations

import argparse

from reindex.service import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index documents from WATCH_PATH into Qdrant.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single reindex pass and exit (no filesystem watcher).",
    )
    args = parser.parse_args()
    run(once=args.once)


if __name__ == "__main__":
    main()
