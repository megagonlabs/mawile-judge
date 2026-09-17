from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "data/demo/configs/mock_judge.yaml"


def discover_config_files(root: Path = PROJECT_ROOT) -> list[Path]:
    paths = [
        path
        for pattern in ("data/**/configs/*.yaml",)
        for path in root.glob(pattern)
        if path.is_file()
    ]
    return sorted(dict.fromkeys(paths))


def discover_run_artifacts(runs_dir: Path = PROJECT_ROOT / "runs") -> list[Path]:
    if not runs_dir.exists():
        return []
    return sorted(runs_dir.glob("*/run.json"), reverse=True)


def resolve_path(path: Path | str, base_dir: Path) -> Path:
    parsed = Path(path).expanduser()
    if parsed.is_absolute():
        return parsed.resolve()
    return (base_dir / parsed).resolve()

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "PROJECT_ROOT",
    "discover_config_files",
    "discover_run_artifacts",
    "resolve_path",
]
