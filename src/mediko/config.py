"""Load pipeline settings from YAML, with environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(RuntimeError):
    """The pipeline configuration is missing or invalid."""


def config_path() -> Path:
    override = os.environ.get("MEDIKO_CONFIG")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass(frozen=True)
class Settings:
    max_upload_bytes: int
    min_embedded_chars: int
    ocr_lang: str
    ocr_device: str
    render_dpi: int


def load_settings(path: Path | None = None) -> Settings:
    config_file = path or config_path()
    if not config_file.is_file():
        raise ConfigError(f"Config file not found: {config_file}")
    loaded = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config file must be a mapping: {config_file}")

    input_cfg = _mapping(loaded, "input")
    ocr_cfg = _mapping(loaded, "ocr")
    max_upload_bytes = _positive_int(input_cfg, "max_upload_bytes")
    min_embedded_chars = _non_negative_int(input_cfg, "min_embedded_chars")
    render_dpi = _positive_int(ocr_cfg, "render_dpi")
    ocr_lang = _text(ocr_cfg, "lang")
    ocr_device = _text(ocr_cfg, "device")

    if "MEDIKO_MAX_UPLOAD_BYTES" in os.environ:
        max_upload_bytes = _env_positive_int("MEDIKO_MAX_UPLOAD_BYTES")
    if "MEDIKO_OCR_LANG" in os.environ:
        ocr_lang = os.environ["MEDIKO_OCR_LANG"].strip()
    if "MEDIKO_OCR_DEVICE" in os.environ:
        ocr_device = os.environ["MEDIKO_OCR_DEVICE"].strip()
    if not ocr_lang or not ocr_device:
        raise ConfigError("OCR lang and device must be non-empty.")

    return Settings(
        max_upload_bytes=max_upload_bytes,
        min_embedded_chars=min_embedded_chars,
        ocr_lang=ocr_lang,
        ocr_device=ocr_device,
        render_dpi=render_dpi,
    )


def _mapping(loaded: dict, key: str) -> dict:
    value = loaded.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Config is missing the {key} section.")
    return value


def _positive_int(section: dict, key: str) -> int:
    value = _int_value(section, key)
    if value <= 0:
        raise ConfigError(f"{key} must be greater than zero.")
    return value


def _non_negative_int(section: dict, key: str) -> int:
    value = _int_value(section, key)
    if value < 0:
        raise ConfigError(f"{key} must be zero or greater.")
    return value


def _int_value(section: dict, key: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer.")
    return value


def _text(section: dict, key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string.")
    return value.strip()


def _env_positive_int(name: str) -> int:
    raw = os.environ[name].strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer.") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be greater than zero.")
    return value
