"""Resolve portable paths for the research scripts using a small TOML file."""

from pathlib import Path
import os
import tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = REPOSITORY_ROOT / "config" / "paths.local.toml"
if not CONFIG_FILE.exists():
    CONFIG_FILE = REPOSITORY_ROOT / "config" / "paths.example.toml"
with CONFIG_FILE.open("rb") as stream:
    SETTINGS = tomllib.load(stream)


def configured_path(name: str) -> Path:
    """Resolve a path relative to the repository, with optional env expansion."""
    path = Path(os.path.expandvars(SETTINGS["paths"][name])).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


STUDY_ROOT = configured_path("study_root")
DATA_ROOT = configured_path("data_root")
BOUNDARY_PATH = configured_path("boundary")
CLCD_PATH = configured_path("land_cover")


def font_path(language: str) -> str:
    """Find an installed font, or use an explicitly configured font file."""
    from matplotlib import font_manager

    configured = SETTINGS.get("fonts", {}).get(language, "")
    if configured:
        path = Path(os.path.expandvars(configured)).expanduser()
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Configured font does not exist: {path}")
        return str(path)

    families = (
        ["Noto Sans CJK SC", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
        if language == "chinese"
        else ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next(name for name in families if name in available)
    return font_manager.findfont(font_manager.FontProperties(family=family))

