"""Typed AST dataclasses for the subset of OSC2 used by osc2carla."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class Expr:
    """Marker base class for all expressions."""


@dataclass
class NumLit(Expr):
    value: float
    is_int: bool = False


@dataclass
class BoolLit(Expr):
    value: bool


@dataclass
class StringLit(Expr):
    value: str


@dataclass
class PhysicalLiteral(Expr):
    value: float
    unit: str


@dataclass
class Identifier(Expr):
    name: str


@dataclass
class QualifiedIdentifier(Expr):
    parts: List[str]

    @property
    def name(self) -> str:
        return ".".join(self.parts)


@dataclass
class MemberAccess(Expr):
    target: Expr
    field: str


@dataclass
class Call(Expr):
    target: Expr
    args: "ArgList"


@dataclass
class BinaryExpr(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass
class UnaryExpr(Expr):
    op: str
    operand: Expr


@dataclass
class Rise(Expr):
    expr: Expr


@dataclass
class Fall(Expr):
    expr: Expr


@dataclass
class Elapsed(Expr):
    duration: Expr


@dataclass
class EventRef(Expr):
    name: str


@dataclass
class ArgList:
    positional: List[Expr] = field(default_factory=list)
    named: Dict[str, Expr] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.positional) + len(self.named)


@dataclass
class KeepConstraint:
    expr: Expr


@dataclass
class VarDecl:
    name: str
    type_name: str
    initializer: Optional[Expr] = None


@dataclass
class ActorFieldDecl:
    name: str
    type_name: str
    initializer: Optional[Expr] = None


@dataclass
class ActorDecl:
    name: str
    inherits: Optional[str] = None
    fields: List[ActorFieldDecl] = field(default_factory=list)


@dataclass
class ScenarioActorBinding:
    name: str
    type_name: str
    constraints: List[KeepConstraint] = field(default_factory=list)


@dataclass
class ActionDecl:
    name: str
    inherits: Optional[str] = None


@dataclass
class ModifierDecl:
    name: str


@dataclass
class StructDecl:
    name: str
    inherits: Optional[str] = None
    fields: List[ActorFieldDecl] = field(default_factory=list)


@dataclass
class PhysicalTypeDecl:
    name: str
    si: Dict[str, int] = field(default_factory=dict)


@dataclass
class UnitDecl:
    name: str
    type_name: str
    factor: float = 1.0
    offset: float = 0.0
    si: Dict[str, int] = field(default_factory=dict)


@dataclass
class EnumDecl:
    name: str
    members: List[str] = field(default_factory=list)


@dataclass
class ImportStmt:
    target: str


@dataclass
class Modifier:
    name: str
    args: ArgList = field(default_factory=ArgList)
    at: Optional[str] = None


@dataclass
class WithBlock:
    modifiers: List[Modifier] = field(default_factory=list)
    constraints: List[KeepConstraint] = field(default_factory=list)


@dataclass
class ActionCall:
    actor: Optional[str]
    behaviour: str
    args: ArgList = field(default_factory=ArgList)
    with_block: Optional[WithBlock] = None


@dataclass
class Composition:
    op: str  # 'serial' | 'parallel' | 'one_of'
    members: List["DoMember"] = field(default_factory=list)


@dataclass
class Wait:
    condition: Expr


@dataclass
class Emit:
    event: str
    args: ArgList = field(default_factory=ArgList)


DoMember = Union[Composition, ActionCall, Wait, Emit]


@dataclass
class DoDirective:
    body: DoMember


@dataclass
class ScenarioDecl:
    name: str
    inherits: Optional[str] = None
    actors: List[ScenarioActorBinding] = field(default_factory=list)
    variables: List[VarDecl] = field(default_factory=list)
    do: Optional[DoDirective] = None


@dataclass
class OscFile:
    imports: List[ImportStmt] = field(default_factory=list)
    declarations: List[Any] = field(default_factory=list)

    @property
    def scenarios(self) -> List[ScenarioDecl]:
        return [d for d in self.declarations if isinstance(d, ScenarioDecl)]

    @property
    def actors(self) -> List[ActorDecl]:
        return [d for d in self.declarations if isinstance(d, ActorDecl)]

    @property
    def structs(self) -> List[StructDecl]:
        return [d for d in self.declarations if isinstance(d, StructDecl)]

    @property
    def actions(self) -> List[ActionDecl]:
        return [d for d in self.declarations if isinstance(d, ActionDecl)]

    @property
    def modifiers(self) -> List[ModifierDecl]:
        return [d for d in self.declarations if isinstance(d, ModifierDecl)]

    @property
    def units(self) -> List[UnitDecl]:
        return [d for d in self.declarations if isinstance(d, UnitDecl)]

    @property
    def physical_types(self) -> List[PhysicalTypeDecl]:
        return [d for d in self.declarations if isinstance(d, PhysicalTypeDecl)]

    @property
    def enums(self) -> List[EnumDecl]:
        return [d for d in self.declarations if isinstance(d, EnumDecl)]
