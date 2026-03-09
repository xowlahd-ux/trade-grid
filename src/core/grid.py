from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Any


def _is_scalar_option_list(obj: list[Any]) -> bool:
    return len(obj) > 0 and not any(isinstance(x, (dict, list)) for x in obj)


def _is_dict_option_list(obj: list[Any]) -> bool:
    return len(obj) > 0 and all(isinstance(x, dict) for x in obj)


def _collect_paths(obj: Any, path: tuple[str, ...] = ()):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.extend(_collect_paths(v, path + (str(k),)))
    elif isinstance(obj, list):
        if _is_scalar_option_list(obj) or _is_dict_option_list(obj):
            found.append((path, obj))
    return found


def _set_path(d: dict, path: tuple[str, ...], value: Any) -> None:
    cur = d
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = value


def flatten_params(cfg: dict) -> list[tuple[str, ...]]:
    return [p for p, _ in _collect_paths(cfg)]


def expand_grid(cfg: dict) -> list[tuple[dict, dict[str, Any]]]:
    items = _collect_paths(cfg)
    if not items:
        return [(deepcopy(cfg), {})]
    paths = [p for p, _ in items]
    value_lists = [vals for _, vals in items]
    out = []
    for combo in itertools.product(*value_lists):
        new_cfg = deepcopy(cfg)
        params = {}
        for path, val in zip(paths, combo):
            _set_path(new_cfg, path, deepcopy(val))
            params[".".join(path)] = deepcopy(val)
        out.append((new_cfg, params))
    return out
