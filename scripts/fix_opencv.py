"""Leave exactly one OpenCV distribution installed.

ultralytics hard-requires ``opencv-python``, the GUI build, which pulls Qt and
breaks headless Docker and CI. This project requires ``opencv-python-headless``.
Both ship the same ``cv2`` package and overlap on 52 files, so a clean
``pip install -e ".[dev,vision]"`` installs both and whichever unpacked last
wins those files — a silent coin flip that changes with install order.

pip has no dependency-override mechanism readable from ``pyproject.toml``, so
this cannot be fixed declaratively for pip users. Run this afterwards instead.
It uninstalls both (order matters: removing one first would delete files the
other still needs) and reinstalls headless alone.

``pip check`` will afterwards report ultralytics' ``opencv-python`` requirement
as unmet. That is metadata-only — ``import cv2`` works and the API is identical.

**This is a mitigation, not a repair.** It settles which OpenCV distribution is
installed and settles nothing about the larger problem, which is that importing
ultralytics mutates global state in libraries this pipeline shares: it rebinds
``cv2.imread``/``imwrite``/``imshow`` on Windows, calls ``cv2.setNumThreads(0)``
process-wide, and replaces ``torch.save``. Two live bugs have come out of that
so far. See ``docs/ultralytics-patches.md`` for the audit and for the cleaner
long-term answer, which is running inference out of process.

Usage::

    python scripts/fix_opencv.py           # fix
    python scripts/fix_opencv.py --check   # report only, exit 1 if wrong
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import subprocess
import sys

GUI = "opencv-python"
HEADLESS = "opencv-python-headless"


def installed(name: str) -> str | None:
    """Return the installed version of a distribution, or ``None``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def report() -> tuple[str | None, str | None]:
    """Print which OpenCV distributions are present."""
    gui, headless = installed(GUI), installed(HEADLESS)
    print(f"  {GUI:24s} {gui or '(not installed)'}")
    print(f"  {HEADLESS:24s} {headless or '(not installed)'}")
    return gui, headless


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="report only; exit 1 if the state is wrong"
    )
    args = parser.parse_args(argv)

    print("OpenCV distributions before:")
    gui, headless = report()

    if args.check:
        if gui is not None:
            print(f"\nFAIL: {GUI} is installed. Run this script without --check.")
            return 1
        if headless is None:
            print(f"\nFAIL: {HEADLESS} is not installed.")
            return 1
        print("\nOK: exactly one OpenCV distribution, and it is the headless build.")
        return 0

    if gui is None and headless is not None:
        print("\nAlready correct; nothing to do.")
        return 0

    # Both share 52 file paths. Removing one alone would delete files the other
    # still lists, so remove both and reinstall from scratch.
    print("\nuninstalling both...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", GUI, HEADLESS],
        check=False,
    )
    print("installing headless only...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", f"{HEADLESS}>=4.9"],
        check=False,
    )
    if result.returncode != 0:
        print("FAILED to reinstall headless; cv2 is now missing entirely.")
        return result.returncode

    print("\nOpenCV distributions after:")
    report()
    try:
        import cv2

        print(f"\nimport cv2 -> {cv2.__version__}")
    except ImportError as error:  # pragma: no cover - would mean a broken install
        print(f"\nimport cv2 FAILED: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
