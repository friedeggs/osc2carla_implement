"""Decorator-based MethodRegistry.

Maps abstract OSC2 action names (e.g. ``vehicle.drive``) to factories that
return ``py_trees.behaviour.Behaviour`` instances bound against a live
CARLA actor.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional


_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register(action_name: str):
    def deco(fn):
        _REGISTRY[action_name] = fn
        return fn
    return deco


class MethodRegistry:
    @staticmethod
    def get(action_name: str) -> Optional[Callable[..., Any]]:
        return _REGISTRY.get(action_name)

    @staticmethod
    def items():
        return _REGISTRY.items()

    @staticmethod
    def names():
        return list(_REGISTRY)

    @staticmethod
    def has(action_name: str) -> bool:
        return action_name in _REGISTRY

    @staticmethod
    def reset() -> None:
        _REGISTRY.clear()
