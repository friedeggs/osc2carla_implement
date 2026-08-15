"""Runtime ExecutionContext."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict

from ..frontend import nodes
from ..middle import AnnotatedScenario


@dataclass
class Quantity:
    value: float
    type_name: str = ""

    def __float__(self) -> float:
        return float(self.value)

    def __add__(self, other):
        return _qty_arith(self, other, lambda a, b: a + b)

    def __sub__(self, other):
        return _qty_arith(self, other, lambda a, b: a - b)

    def __mul__(self, other):
        return _qty_arith(self, other, lambda a, b: a * b, allow_dimless=True)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return _qty_arith(self, other, lambda a, b: a / b, allow_dimless=True)

    def __neg__(self):
        return Quantity(-self.value, self.type_name)

    def __lt__(self, other):
        return self.value < _to_value(other)

    def __le__(self, other):
        return self.value <= _to_value(other)

    def __gt__(self, other):
        return self.value > _to_value(other)

    def __ge__(self, other):
        return self.value >= _to_value(other)

    def __eq__(self, other):
        return self.value == _to_value(other)

    def __ne__(self, other):
        return self.value != _to_value(other)


def _to_value(x) -> float:
    if isinstance(x, Quantity):
        return x.value
    return float(x)


def _qty_arith(a: Quantity, b, op: Callable[[float, float], float],
               allow_dimless: bool = False) -> Quantity:
    if isinstance(b, Quantity):
        result_type = a.type_name or b.type_name
    else:
        result_type = a.type_name
    return Quantity(op(a.value, _to_value(b)), result_type)


class EvalError(RuntimeError):
    pass


class ExecutionContext:
    def __init__(self, annotated: AnnotatedScenario, world: Any = None, carla_map: Any = None):
        self.annotated = annotated
        self.world = world
        self.carla_map = carla_map
        self._actor_cache: Dict[str, Any] = {}
        self._role_names: Dict[str, str] = {}
        self._vars: Dict[str, Quantity] = {}
        self.blackboard: Dict[str, bool] = {}
        self._edge_prev: Dict[int, bool] = {}
        self._tick_time: float = 0.0
        self.tick_count: int = 0

        for binding in annotated.scenario.actors:
            attrs = getattr(binding, "attributes", {}) or {}
            self._role_names[binding.name] = attrs.get("name", binding.name)

        self._resolve_initial_variables()

    def _resolve_initial_variables(self) -> None:
        for var in self.annotated.scenario.variables:
            if var.initializer is not None:
                self._vars[var.name] = self._coerce_quantity(
                    self.eval(var.initializer), var.type_name
                )

    def _coerce_quantity(self, value: Any, type_name: str) -> Quantity:
        if isinstance(value, Quantity):
            value.type_name = value.type_name or type_name
            return value
        return Quantity(float(value), type_name)

    def get_variable(self, name: str) -> Quantity:
        if name not in self._vars:
            raise KeyError(f"Variable '{name}' is not bound")
        return self._vars[name]

    def bind_actor(self, binding_name: str, role_name: str, actor: Any) -> None:
        self._role_names[binding_name] = role_name
        self._actor_cache[role_name] = actor

    def actor(self, binding_name: str) -> Any:
        role = self._role_names.get(binding_name, binding_name)
        actor = self._actor_cache.get(role)
        if actor is None and self.world is not None:
            actor = self._lookup_actor_by_role(role)
            if actor is not None:
                self._actor_cache[role] = actor
        return actor

    def _lookup_actor_by_role(self, role_name: str) -> Any:
        if self.world is None:
            return None
        for a in self.world.get_actors():
            attrs = getattr(a, "attributes", None)
            if attrs and attrs.get("role_name") == role_name:
                return a
        return None

    def advance_tick(self, sim_time: float) -> None:
        self._tick_time = sim_time
        self.tick_count += 1

    @property
    def sim_time(self) -> float:
        return self._tick_time

    def eval(self, expr: nodes.Expr) -> Any:
        method = f"_eval_{type(expr).__name__}"
        fn = getattr(self, method, None)
        if fn is None:
            raise EvalError(f"No evaluator for {type(expr).__name__}")
        return fn(expr)

    def _eval_NumLit(self, e: nodes.NumLit) -> Quantity:
        return Quantity(float(e.value))

    def _eval_BoolLit(self, e: nodes.BoolLit) -> bool:
        return bool(e.value)

    def _eval_StringLit(self, e: nodes.StringLit) -> str:
        return e.value

    def _eval_PhysicalLiteral(self, e: nodes.PhysicalLiteral) -> Quantity:
        factor, type_name = self.annotated.unit_table.get(e.unit, (1.0, ""))
        return Quantity(value=e.value * factor, type_name=type_name)

    def _eval_Identifier(self, e: nodes.Identifier) -> Any:
        if e.name in self._vars:
            return self._vars[e.name]
        if e.name in self._role_names:
            return _ActorHandle(self, e.name)
        return e.name

    def _eval_QualifiedIdentifier(self, e: nodes.QualifiedIdentifier) -> Any:
        return e.name

    def _eval_MemberAccess(self, e: nodes.MemberAccess) -> Any:
        target = self.eval(e.target)
        return _resolve_member(target, e.field, self)

    def _eval_Call(self, e: nodes.Call) -> Any:
        callee = self.eval(e.target)
        if not callable(callee):
            raise EvalError(f"Not callable: {e.target}")
        args = [self.eval(a) for a in e.args.positional]
        kwargs = {k: self.eval(v) for k, v in e.args.named.items()}
        return callee(*args, **kwargs)

    def _eval_BinaryExpr(self, e: nodes.BinaryExpr) -> Any:
        left = self.eval(e.left)
        right = self.eval(e.right)
        return _apply_binop(e.op, left, right)

    def _eval_UnaryExpr(self, e: nodes.UnaryExpr) -> Any:
        v = self.eval(e.operand)
        if e.op == "-":
            return -v if not isinstance(v, Quantity) else Quantity(-v.value, v.type_name)
        if e.op == "not":
            return not bool(v)
        raise EvalError(f"Unknown unary op {e.op}")

    def _eval_Rise(self, e: nodes.Rise) -> bool:
        cur = bool(self.eval(e.expr))
        prev = self._edge_prev.get(id(e), False)
        self._edge_prev[id(e)] = cur
        return (not prev) and cur

    def _eval_Fall(self, e: nodes.Fall) -> bool:
        cur = bool(self.eval(e.expr))
        prev = self._edge_prev.get(id(e), True)
        self._edge_prev[id(e)] = cur
        return prev and (not cur)

    def _eval_Elapsed(self, e: nodes.Elapsed) -> bool:
        duration = self.eval(e.duration)
        if isinstance(duration, Quantity):
            duration = duration.value
        start = self._edge_prev.setdefault(id(e), self._tick_time)
        return (self._tick_time - start) >= float(duration)

    def _eval_EventRef(self, e: nodes.EventRef) -> bool:
        return bool(self.blackboard.get(e.name, False))


class _ActorHandle:
    def __init__(self, ctx: ExecutionContext, binding_name: str):
        self._ctx = ctx
        self._binding = binding_name

    @property
    def carla_actor(self):
        return self._ctx.actor(self._binding)

    @property
    def speed(self) -> Quantity:
        a = self.carla_actor
        if a is None:
            return Quantity(0.0, "speed")
        v = a.get_velocity()
        return Quantity(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z), "speed")

    @property
    def position(self) -> "_Position":
        return _Position(self)

    def object_distance(self, reference, direction: str = "euclidean") -> Quantity:
        ref = reference.carla_actor if isinstance(reference, _ActorHandle) else None
        me = self.carla_actor
        if ref is None or me is None:
            return Quantity(float("inf"), "length")
        a = me.get_location()
        b = ref.get_location()
        if direction == "euclidean":
            return Quantity(math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2), "length")
        m = self._ctx.carla_map
        if m is None:
            return Quantity(float("inf"), "length")
        wp_a = m.get_waypoint(a, project_to_road=True)
        wp_b = m.get_waypoint(b, project_to_road=True)
        if wp_a is None or wp_b is None:
            return Quantity(float("inf"), "length")
        return Quantity(abs(wp_a.s - wp_b.s), "length")


class _Position:
    def __init__(self, owner: _ActorHandle):
        self._owner = owner
        self._ctx = owner._ctx

    def ahead_of(self, other: _ActorHandle) -> Quantity:
        m = self._ctx.carla_map
        me = self._owner.carla_actor
        them = other.carla_actor if isinstance(other, _ActorHandle) else None
        if me is None or them is None or m is None:
            return Quantity(0.0, "length")
        wp_me = m.get_waypoint(me.get_location(), project_to_road=True)
        wp_th = m.get_waypoint(them.get_location(), project_to_road=True)
        if wp_me is None or wp_th is None:
            return Quantity(0.0, "length")
        if wp_me.road_id == wp_th.road_id:
            return Quantity(wp_me.s - wp_th.s, "length")
        loc_me = me.get_location()
        loc_th = them.get_location()
        fwd = me.get_transform().get_forward_vector()
        dx, dy = loc_me.x - loc_th.x, loc_me.y - loc_th.y
        return Quantity(dx * fwd.x + dy * fwd.y, "length")


def _apply_binop(op: str, a, b):
    if op == "+":
        return a + b if isinstance(a, (Quantity, float, int)) else Quantity(float(a) + float(b))
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b
    if op == "%":
        return Quantity(_to_value(a) % _to_value(b))
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    if op == "and":
        return bool(a) and bool(b)
    if op == "or":
        return bool(a) or bool(b)
    if op == "=>":
        return (not bool(a)) or bool(b)
    raise EvalError(f"Unsupported binary op {op}")


def _resolve_member(target: Any, field: str, ctx: ExecutionContext) -> Any:
    if isinstance(target, _ActorHandle):
        attr = getattr(target, field, None)
        if attr is not None:
            return attr
        carla_actor = target.carla_actor
        if carla_actor is not None and hasattr(carla_actor, field):
            return getattr(carla_actor, field)
    if isinstance(target, _Position):
        attr = getattr(target, field, None)
        if attr is not None:
            return attr
    if hasattr(target, field):
        return getattr(target, field)
    raise EvalError(f"Unresolved member: {field} on {type(target).__name__}")
