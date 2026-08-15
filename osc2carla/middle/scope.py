"""Symbol-table primitives for the osc2carla semantic middle-end."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Symbol:
    name: str
    kind: str  # 'type' | 'actor' | 'action' | 'modifier' | 'var' | 'field' | 'unit' | 'enum' | 'binding'
    decl: Any = None
    type_ref: Optional[str] = None
    resolved_type: Optional["Symbol"] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class SemanticError(RuntimeError):
    """Raised when the analyser detects an unresolved or inconsistent symbol."""


class Scope:
    """A lexical scope; supports parent chain lookup."""

    _PRIMARY_KINDS = {"type", "actor", "scenario", "var", "binding", "field",
                       "enum", "enum_member", "unit"}

    def __init__(self, name: str, parent: Optional["Scope"] = None):
        self.name = name
        self.parent = parent
        self.symbols: Dict[str, Symbol] = {}
        self._secondary: Dict[str, Dict[str, Symbol]] = {}
        self.children: List[Scope] = []
        if parent is not None:
            parent.children.append(self)

    def define(self, sym: Symbol, *, overwrite: bool = False) -> Symbol:
        if sym.kind in self._PRIMARY_KINDS:
            if not overwrite and sym.name in self.symbols:
                existing = self.symbols[sym.name]
                if existing.kind != sym.kind:
                    raise SemanticError(
                        f"Symbol '{sym.name}' redeclared as {sym.kind} in scope {self.name!r} "
                        f"(previously {existing.kind})"
                    )
            self.symbols[sym.name] = sym
        else:
            ns = self._secondary.setdefault(sym.kind, {})
            ns[sym.name] = sym
        return sym

    def lookup_local(self, name: str, kind: Optional[str] = None) -> Optional[Symbol]:
        if kind is not None and kind not in self._PRIMARY_KINDS:
            ns = self._secondary.get(kind, {})
            return ns.get(name)
        return self.symbols.get(name)

    def lookup(self, name: str, kind: Optional[str] = None) -> Optional[Symbol]:
        sc: Optional[Scope] = self
        while sc is not None:
            sym = sc.lookup_local(name, kind=kind)
            if sym is not None:
                return sym
            sc = sc.parent
        return None

    def lookup_kind(self, name: str, kind: str) -> Optional[Symbol]:
        return self.lookup(name, kind=kind)

    def __repr__(self) -> str:
        return f"Scope({self.name!r}, symbols={list(self.symbols)})"


class GlobalScope(Scope):
    def __init__(self):
        super().__init__(name="<global>")
        for prim in ("int", "uint", "float", "bool", "string"):
            self.define(Symbol(name=prim, kind="type"))
