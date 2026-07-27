"""Typed configuration loader.

Loads `config.yaml`, then applies environment-variable overrides of the form
`ARAG_<SECTION>__<KEY>[__<SUBKEY>]` (double underscore = nesting). Values are
coerced to match the YAML default's type so `ARAG_RETRIEVAL__K_DENSE=20` becomes
an int, `ARAG_CACHE__ENABLED=true` becomes a bool.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_PREFIX = "ARAG_"


def _coerce(value: str, template: Any) -> Any:
    if isinstance(template, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)
    if isinstance(template, float):
        return float(value)
    if template is None:
        # Best-effort: leave as string unless it's an int/float literal.
        for caster in (int, float):
            try:
                return caster(value)
            except ValueError:
                continue
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if value.lower() in {"none", "null", ""}:
            return None
    return value


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    for env_key, raw in os.environ.items():
        if not env_key.startswith(_ENV_PREFIX):
            continue
        path = env_key[len(_ENV_PREFIX):].lower().split("__")
        node = data
        for part in path[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        leaf = path[-1]
        template = node.get(leaf)
        node[leaf] = _coerce(raw, template)
    return data


@dataclass
class Config:
    """Thin dot-accessible wrapper over the nested config dict.

    Access sections as attributes (`cfg.retrieval`) which return `Section`
    objects, or use `cfg.get("retrieval.k_dense", default)`.
    """

    _data: dict[str, Any] = field(default_factory=dict)
    path: str | None = None

    def __getattr__(self, item: str) -> Any:
        # Only called for names not found normally (avoids clobbering _data/path).
        data = object.__getattribute__(self, "_data")
        if item in data:
            val = data[item]
            return Section(val) if isinstance(val, dict) else val
        raise AttributeError(item)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def as_dict(self) -> dict[str, Any]:
        import copy

        return copy.deepcopy(self._data)

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        """Return a deep-copied config with dotted-key overrides applied.

        Used heavily by the ablation harness to spin up pipeline variants.
        """
        import copy

        data = copy.deepcopy(self._data)
        for dotted, value in overrides.items():
            parts = dotted.split(".")
            node = data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return Config(_data=data, path=self.path)


class Section:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, item: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if item in data:
            val = data[item]
            return Section(val) if isinstance(val, dict) else val
        raise AttributeError(item)

    def get(self, key: str, default: Any = None) -> Any:
        val = self._data.get(key, default)
        return Section(val) if isinstance(val, dict) else val

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)


def load_config(path: str | Path = "config/config.yaml") -> Config:
    path = Path(path)
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
    else:
        data = {}
    data = _apply_env_overrides(data)
    return Config(_data=data, path=str(path))
