"""Translate the annotated AST into a py_trees behaviour tree."""
from __future__ import annotations

from typing import Any, List

import py_trees

from ..frontend import nodes
from ..middle import AnnotatedScenario
from .context import ExecutionContext, _ActorHandle
from .method_registry import MethodRegistry


class _ConditionLeaf(py_trees.behaviour.Behaviour):
    def __init__(self, expr: nodes.Expr, ctx: ExecutionContext, name: str = "Wait"):
        super().__init__(name=name)
        self._expr = expr
        self._ctx = ctx

    def update(self):
        try:
            ok = bool(self._ctx.eval(self._expr))
        except Exception:  # noqa: BLE001
            ok = False
        return py_trees.common.Status.SUCCESS if ok else py_trees.common.Status.RUNNING


class _EmitLeaf(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, ctx: ExecutionContext):
        super().__init__(name=f"Emit[{name}]")
        self._event = name
        self._ctx = ctx

    def update(self):
        self._ctx.blackboard[self._event] = True
        return py_trees.common.Status.SUCCESS


#: Actions that actuate a vehicle. When a binding is handed to an external
#: controller these are the ones the tree must stop issuing; everything else
#: (assign_position, set_lights, emit, wait) stays, so the scenario's phases,
#: events and monitors are unaffected.
EXTERNALLY_CONTROLLED_ACTIONS = {
    "drive": "running",        # open-ended in the tree -> stay RUNNING
    "ram": "running",
    "change_speed": "success",  # terminates in the tree -> succeed at once so
    "change_lane": "success",   # the surrounding serial block still advances
}


class _ExternalControlLeaf(py_trees.behaviour.Behaviour):
    """Placeholder for an action now issued by an external ego policy."""

    def __init__(self, name: str, mode: str):
        super().__init__(name=name)
        self._mode = mode

    def update(self):
        if self._mode == "success":
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class BehaviorTreeBuilder:
    def __init__(self, annotated: AnnotatedScenario, ctx: ExecutionContext,
                 external_actors=None):
        self.annotated = annotated
        self.ctx = ctx
        #: binding names whose actuation is delegated to an external policy
        self.external_actors = set(external_actors or ())

    def build(self) -> py_trees.behaviour.Behaviour:
        do = self.annotated.scenario.do
        if do is None:
            return py_trees.behaviours.Success(name="Empty")
        return self._build_member(do.body)

    def _build_member(self, member) -> py_trees.behaviour.Behaviour:
        if isinstance(member, nodes.Composition):
            return self._build_composition(member)
        if isinstance(member, nodes.ActionCall):
            return self._build_action(member)
        if isinstance(member, nodes.Wait):
            return _ConditionLeaf(member.condition, self.ctx,
                                  name=f"Wait[{_summarise_expr(member.condition)}]")
        if isinstance(member, nodes.Emit):
            return _EmitLeaf(member.event, self.ctx)
        return py_trees.behaviours.Success(name="Noop")

    def _build_composition(self, comp: nodes.Composition) -> py_trees.composites.Composite:
        children = [self._build_member(m) for m in comp.members]
        op = comp.op
        if op == "serial":
            seq = py_trees.composites.Sequence(name="serial", memory=True)
            seq.add_children(children)
            return seq
        if op == "parallel":
            par = py_trees.composites.Parallel(
                name="parallel",
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
            )
            par.add_children(children)
            return par
        if op == "one_of":
            par = py_trees.composites.Parallel(
                name="one_of",
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
            )
            par.add_children(children)
            return par
        seq = py_trees.composites.Sequence(name=op or "serial", memory=True)
        seq.add_children(children)
        return seq

    def _build_action(self, action: nodes.ActionCall) -> py_trees.behaviour.Behaviour:
        actor_binding = action.actor
        if actor_binding in self.external_actors:
            base = action.behaviour.rsplit(".", 1)[-1]
            mode = EXTERNALLY_CONTROLLED_ACTIONS.get(base)
            if mode is not None:
                return _ExternalControlLeaf(
                    name=f"External[{actor_binding}.{base}]", mode=mode)
        actor_handle = None
        if actor_binding is not None:
            actor_handle = _ActorHandle(self.ctx, actor_binding)
        action_full_name = action.behaviour
        if actor_binding is not None and "." not in action_full_name:
            binding_sym = self.annotated.scope.lookup(actor_binding)
            if binding_sym is not None and binding_sym.type_ref:
                action_full_name = f"{binding_sym.type_ref}.{action.behaviour}"

        builder = MethodRegistry.get(action_full_name)
        if builder is None:
            for fallback in [action.behaviour, f"vehicle.{action.behaviour}"]:
                builder = MethodRegistry.get(fallback)
                if builder is not None:
                    action_full_name = fallback
                    break
        if builder is None:
            return py_trees.behaviours.Success(name=f"Unmapped[{action_full_name}]")

        modifiers: List[nodes.Modifier] = []
        if action.with_block is not None:
            modifiers = action.with_block.modifiers
        return builder(actor_handle, action.args, modifiers, self.ctx)


def _summarise_expr(expr: nodes.Expr) -> str:
    if isinstance(expr, nodes.Rise):
        return f"rise({_summarise_expr(expr.expr)})"
    if isinstance(expr, nodes.Fall):
        return f"fall({_summarise_expr(expr.expr)})"
    if isinstance(expr, nodes.Elapsed):
        return "elapsed"
    if isinstance(expr, nodes.EventRef):
        return f"@{expr.name}"
    if isinstance(expr, nodes.BinaryExpr):
        return f"{_summarise_expr(expr.left)}{expr.op}{_summarise_expr(expr.right)}"
    if isinstance(expr, nodes.Identifier):
        return expr.name
    if isinstance(expr, nodes.MemberAccess):
        return f"{_summarise_expr(expr.target)}.{expr.field}"
    return type(expr).__name__
