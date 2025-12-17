from __future__ import annotations

import configparser
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    tomllib = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILENAMES: tuple[str, ...] = ("tagger.toml", "tagger.ini")


@dataclass(frozen=True)
class Settings:
    task: str = "postwita"
    data_root: Path = PROJECT_ROOT / "data"
    w2v_small_path: Optional[Path] = None
    w2v_big_path: Optional[Path] = None
    logging_level: str = "INFO"
    lstm_units: int = 1024
    dropout: float = 0.1
    epochs: int = 20
    batch_size: int = 32
    validation_split: float = 0.1
    sequence_length: Optional[int] = None
    output_dir: Path = PROJECT_ROOT / "artifacts"
    seed: int = 0
    env_prefix: str = "TAGGER_"
    config_file: Optional[Path] = None

    def resolve_paths(self) -> "Settings":
        return replace(
            self,
            data_root=self._expand(self.data_root),
            output_dir=self._expand(self.output_dir),
            w2v_small_path=self._expand(self.w2v_small_path),
            w2v_big_path=self._expand(self.w2v_big_path),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "data_root": str(self.data_root),
            "output_dir": str(self.output_dir),
            "w2v_small_path": str(self.w2v_small_path) if self.w2v_small_path else None,
            "w2v_big_path": str(self.w2v_big_path) if self.w2v_big_path else None,
        }

    @staticmethod
    def _expand(value: Optional[Path]) -> Optional[Path]:
        if value is None:
            return None
        return Path(value).expanduser().resolve()


def load_settings(
    config_file: Optional[str | Path] = None,
    env: Mapping[str, str] = os.environ,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Settings:
    base = Settings()
    config_path = _find_config_file(config_file)

    merged: Dict[str, Any] = {}
    if config_path:
        merged.update(_read_config_file(config_path))
        merged["config_file"] = Path(config_path)
    merged.update(_read_env(env, base.env_prefix))
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})

    settings = Settings(**{**asdict(base), **merged})
    return settings.resolve_paths()


def _find_config_file(explicit: Optional[str | Path]) -> Optional[Path]:
    if explicit:
        explicit_path = Path(explicit)
        return explicit_path if explicit_path.exists() else None

    for fname in DEFAULT_CONFIG_FILENAMES:
        candidate = PROJECT_ROOT / fname
        if candidate.exists():
            return candidate
    return None


def _read_env(env: Mapping[str, str], prefix: str) -> Dict[str, Any]:
    mapping = {
        "task": env.get(f"{prefix}TASK"),
        "data_root": env.get(f"{prefix}DATA_ROOT"),
        "w2v_small_path": env.get(f"{prefix}W2V_SMALL"),
        "w2v_big_path": env.get(f"{prefix}W2V_BIG"),
        "logging_level": env.get(f"{prefix}LOG_LEVEL"),
        "lstm_units": env.get(f"{prefix}LSTM_UNITS"),
        "dropout": env.get(f"{prefix}DROPOUT"),
        "epochs": env.get(f"{prefix}EPOCHS"),
        "batch_size": env.get(f"{prefix}BATCH_SIZE"),
        "validation_split": env.get(f"{prefix}VALIDATION_SPLIT"),
        "sequence_length": env.get(f"{prefix}SEQUENCE_LENGTH"),
        "output_dir": env.get(f"{prefix}OUTPUT_DIR"),
        "seed": env.get(f"{prefix}SEED"),
    }
    return _coerce_values(mapping)


def _read_config_file(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".toml", ".tml"} and tomllib:
        content = tomllib.loads(path.read_text())
        section = content.get("tagger", content)
        data = _coerce_values(section)
        task = data.get("task")
        if task and task in content:
            data.update(_coerce_values(content[task]))
        return data

    parser = configparser.ConfigParser()
    parser.read(path)
    base = parser["tagger"] if "tagger" in parser else parser["DEFAULT"]
    data = _coerce_values(base)
    task = data.get("task") or base.get("task")
    if task and parser.has_section(task):
        data.update(_coerce_values(parser[task]))
    return data


def _coerce_values(values: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, raw in values.items():
        if raw in (None, ""):
            continue
        normalized_key = {
            "w2v_small_floc": "w2v_small_path",
            "w2v_big_floc": "w2v_big_path",
            "log_level": "logging_level",
        }.get(key, key)
        if key in {"data_root", "output_dir", "w2v_small_path", "w2v_big_path"}:
            out[normalized_key] = Path(str(raw))
        elif key in {"lstm_units", "epochs", "batch_size", "seed", "sequence_length"}:
            out[normalized_key] = int(raw)
        elif key in {"dropout", "validation_split"}:
            out[normalized_key] = float(raw)
        else:
            out[normalized_key] = raw
    return out
