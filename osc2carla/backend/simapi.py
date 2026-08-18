"""Which simulator the backend actuates against.

The atomic behaviours, the initializer, the ego-policy hand-off and the
metrics collector are all written against CARLA's Python API.  They used to
reach it with a module-level ``import carla`` guarded by a ``try``, which
made "no simulator" the only alternative to CARLA.

There are now two simulators.  This module is the single place that decides
which one is live, and it hands back an object that *looks* like the ``carla``
module::

    from .simapi import sim as carla
    ...
    control = carla.VehicleControl(throttle=1.0)

so the call sites are unchanged.  The one thing that had to change is the
"is a simulator available?" test: the proxy is never ``None``, so those guards
read ``if not carla:`` instead of ``if carla is None:``.

Binding is late and global.  Backend modules capture the proxy at import
time; :func:`bind` re-points it afterwards, which is what lets
``osc2carla.cli`` choose a backend after parsing its arguments.
"""
from __future__ import annotations

from typing import Any, Optional

#: backend name -> human-readable description, for CLI help and errors
BACKENDS = {
    "carla": "CARLA 0.9.x server over RPC (needs the carla Python API)",
    "pygame": "bundled standalone simulator + pygame bird's-eye renderer",
}


class SimulatorNotBound(RuntimeError):
    pass


class _SimProxy:
    """Attribute-forwarding stand-in for whichever simulator module is bound."""

    def __init__(self) -> None:
        self.__dict__["_module"] = None
        self.__dict__["_name"] = None

    # -- binding ----------------------------------------------------------

    def bind(self, module: Any, name: str) -> None:
        self.__dict__["_module"] = module
        self.__dict__["_name"] = name

    def unbind(self) -> None:
        self.__dict__["_module"] = None
        self.__dict__["_name"] = None

    @property
    def module(self) -> Optional[Any]:
        return self.__dict__["_module"]

    @property
    def name(self) -> Optional[str]:
        return self.__dict__["_name"]

    @property
    def is_local(self) -> bool:
        return self.__dict__["_name"] == "pygame"

    # -- module-like behaviour -------------------------------------------

    def __bool__(self) -> bool:
        return self.__dict__["_module"] is not None

    def __getattr__(self, attr: str) -> Any:
        module = self.__dict__.get("_module")
        if module is None:
            raise SimulatorNotBound(
                f"no simulator is bound, so {attr!r} is unavailable. "
                f"Install the CARLA Python API, or run with "
                f"--backend pygame for the bundled simulator.")
        return getattr(module, attr)

    def __repr__(self) -> str:
        name = self.__dict__["_name"]
        return f"<simapi {name or 'unbound'}>"


#: the object every backend module imports as ``carla``
sim = _SimProxy()


def bind(backend: str) -> str:
    """Bind ``carla`` or ``pygame``; returns the name actually bound."""
    if backend == "carla":
        import carla  # type: ignore  # noqa: F401
        sim.bind(carla, "carla")
        return "carla"
    if backend == "pygame":
        from ..localsim import api as local_api
        sim.bind(local_api, "pygame")
        return "pygame"
    raise ValueError(f"unknown backend {backend!r}; expected one of "
                     f"{sorted(BACKENDS)}")


def autobind() -> Optional[str]:
    """Bind CARLA if its Python API happens to be importable.

    Keeps the historical behaviour: with CARLA installed the backend modules
    are live the moment they are imported, and without it they stay inert
    (which is what ``--dry-run`` relies on).
    """
    if sim:
        return sim.name
    try:
        return bind("carla")
    except Exception:  # noqa: BLE001 - absence of carla is the normal case
        return None


def describe() -> str:
    if not sim:
        return "no simulator bound"
    return f"{sim.name} ({BACKENDS.get(sim.name or '', 'unknown')})"


autobind()
