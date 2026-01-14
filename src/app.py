"""
Module entry point for the DRHP Analysis System.

Usage:
  - python -m src.app           # launches the Streamlit UI
  - streamlit run src/ui/app.py # equivalent explicit UI launch
"""

import subprocess
import sys
import os


def main() -> int:
    # Ensure working directory is project root when invoked as module
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    try:
        cmd = [sys.executable, '-m', 'streamlit', 'run', 'src/ui/app.py']
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Failed to start UI: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
