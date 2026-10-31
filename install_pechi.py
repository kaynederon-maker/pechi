"""Install the `pechi` command so it runs from any PowerShell window.

    .venv\\Scripts\\python.exe install_pechi.py

Windows only. It writes one small launcher into the WindowsApps folder, which is
already on PATH, so nothing about PATH changes. Delete pechi.cmd there to uninstall.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def launcher(python: str, app: Path) -> str:
    return (
        "@echo off\r\n"
        "set COLORTERM=truecolor\r\n"
        "set PYTHONUTF8=1\r\n"
        f'"{python}" "{app}" %*\r\n'
    )


def main() -> int:
    if os.name != "nt":
        print("This installer is for Windows.", file=sys.stderr)
        return 1
    target = Path(os.environ["LOCALAPPDATA"]) / "Microsoft/WindowsApps/pechi.cmd"
    text = launcher(sys.executable, HERE / "pechi_tui.py")
    target.write_text(text, encoding="mbcs", newline="")
    print(f"Installed {target}\nOpen a new PowerShell window and type: pechi")
    return 0


if __name__ == "__main__":
    sys.exit(main())
