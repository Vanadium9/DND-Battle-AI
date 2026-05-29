from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> int:
    try:
        from ui.app import run_app
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            print(
                "PySide6 is not installed. Install project dependencies with: "
                "pip install -r requirements.txt",
                file=sys.stderr,
            )
            return 1
        raise

    return run_app(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
