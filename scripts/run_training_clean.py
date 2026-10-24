"""Launch rPPG-Toolbox with clean ASCII progress output for PowerShell logging."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


TOOLBOX_ROOT = Path("D:/rppg/rPPG-Toolbox")
MAIN_PATH = TOOLBOX_ROOT / "main.py"


def main() -> None:
    if not MAIN_PATH.is_file():
        raise FileNotFoundError(f"Toolbox main.py not found: {MAIN_PATH}")

    # tqdm normally writes to stderr. Windows PowerShell presents redirected
    # native stderr as NativeCommandError. Send it to stdout and use ASCII.
    sys.stderr = sys.stdout
    import tqdm as tqdm_module

    tqdm_class = tqdm_module.tqdm
    original_init = tqdm_class.__init__

    def clean_init(self, *args, **kwargs):
        kwargs.setdefault("file", sys.stdout)
        kwargs.setdefault("ascii", True)
        original_init(self, *args, **kwargs)

    # Keep tqdm as a class so annotations such as `tqdm | None` remain valid.
    tqdm_class.__init__ = clean_init

    os.chdir(TOOLBOX_ROOT)
    sys.path.insert(0, str(TOOLBOX_ROOT))
    sys.argv = [str(MAIN_PATH), *sys.argv[1:]]
    runpy.run_path(str(MAIN_PATH), run_name="__main__")


if __name__ == "__main__":
    main()
