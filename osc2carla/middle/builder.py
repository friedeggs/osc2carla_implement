"""Two-pass semantic analyser (middle-end)."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

from ..frontend import nodes
from ..frontend.transformer import parse_file
from .scope import GlobalScope, Scope, SemanticError, Symbol


DEFAULT_STDLIB_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "stdlib")
)


class AnnotatedScenario:
    def __init__(self, scenario, scope, global_scope, unit_table, enum_table):
        self.scenario = scenario
        self.scope = scope
        self.global_scope = global_scope
        self.unit_table = unit_table
        self.enum_table = enum_table
        self.variables: Dict[str, nodes.VarDecl] = {v.name: v for v in scenario.variables}
        self.actor_bindings: Dict[str, nodes.ScenarioActorBinding] = {
            a.name: a for a in scenario.actors
        }


def analyse(source_path: str,
            stdlib_dir: Optional[str] = None,
            preload: Sequence[str] = ("types.osc", "domain.osc")) -> AnnotatedScenario:
    builder = ModelBuilder(stdlib_dir or DEFAULT_STDLIB_DIR)
    for name in preload:
        builder.load_file(os.path.join(builder.stdlib_dir, name))
    user_ast = parse_file(source_path)
    builder.load_ast(user_ast)
    builder.definition_pass()
    builder.resolution_pass()
    return builder.finalise(user_ast)


class ModelBuilder:
    def __init__(self, stdlib_dir: str):
        self.stdlib_dir = stdlib_dir
        self.global_scope = GlobalScope()
        self.asts: List[nodes.OscFile] = []
        self.scenario_scopes: Dict[str, Scope] = {}
        self.unit_table: Dict[str, Tuple[float, str]] = {}
        self.enum_table: Dict[str, List[str]] = {}

    def load_file(self, path: str) -> None:
        self.asts.append(parse_file(path))

    def load_ast(self, ast: nodes.OscFile) -> None:
        self.asts.append(ast)

    def definition_pass(self) -> None:
        for ast in self.asts:
            self._define_top_level(ast)

    def _define_top_level(self, ast: nodes.OscFile) -> None:
        for pt in ast.physical_types:
            self.global_scope.define(Symbol(name=pt.name, kind="type", decl=pt,
                                             extra={"si": pt.si}))

        for unit in ast.units:
            self.global_scope.define(Symbol(name=unit.name, kind="unit", decl=unit,
                                             extra={"type": unit.type_name,
                                                    "factor": unit.factor,
                                                    "offset": unit.offset}))
            self.unit_table[unit.name] = (unit.factor, unit.type_name)

        for enum in ast.enums:
            self.global_scope.define(Symbol(name=enum.name, kind="enum", decl=enum,
                                             extra={"members": enum.members}))
            self.enum_table[enum.name] = list(enum.members)
            for member in enum.members:
                if member not in self.global_scope.symbols:
                    self.global_scope.define(
                        Symbol(name=member, kind="enum_member", decl=enum,
                               extra={"enum": enum.name})
                    )

        for struct in ast.structs:
            actor_scope = Scope(struct.name, parent=self.global_scope)
            sym = Symbol(name=struct.name, kind="type", decl=struct,
                         extra={"scope": actor_scope,
                                "inherits": struct.inherits,
                                "fields": {f.name: f.type_name for f in struct.fields}})
            self.global_scope.define(sym)
            for f in struct.fields:
                actor_scope.define(Symbol(name=f.name, kind="field", decl=f,
                                          type_ref=f.type_name))

        for actor in ast.actors:
            actor_scope = Scope(actor.name, parent=self.global_scope)
            sym = Symbol(name=actor.name, kind="actor", decl=actor,
                         extra={"scope": actor_scope,
                                "inherits": actor.inherits,
                                "fields": {f.name: f.type_name for f in actor.fields}})
            self.global_scope.define(sym)
            for f in actor.fields:
                actor_scope.define(Symbol(name=f.name, kind="field", decl=f,
                                          type_ref=f.type_name))

        for action in ast.actions:
            self.global_scope.define(Symbol(name=action.name, kind="action", decl=action))

        for modifier in ast.modifiers:
            self.global_scope.define(Symbol(name=modifier.name, kind="modifier", decl=modifier))

        for scenario in ast.scenarios:
            scenario_scope = Scope(scenario.name, parent=self.global_scope)
            self.scenario_scopes[scenario.name] = scenario_scope
            self.global_scope.define(Symbol(name=scenario.name, kind="scenario",
                                             decl=scenario,
                                             extra={"scope": scenario_scope}))
            for binding in scenario.actors:
                scenario_scope.define(Symbol(name=binding.name, kind="binding",
                                              decl=binding, type_ref=binding.type_name))
            for var in scenario.variables:
                scenario_scope.define(Symbol(name=var.name, kind="var",
                                              decl=var, type_ref=var.type_name))

    def resolution_pass(self) -> None:
        for sym in list(self.global_scope.symbols.values()):
            if sym.kind in ("actor", "type") and sym.extra.get("inherits"):
                parent_name = sym.extra["inherits"]
                parent_sym = self.global_scope.lookup(parent_name)
                if parent_sym is None:
                    raise SemanticError(
                        f"Actor '{sym.name}' inherits from unknown '{parent_name}'"
                    )
                sym.resolved_type = parent_sym
                inherited_fields = parent_sym.extra.get("fields", {})
                actor_scope: Scope = sym.extra["scope"]
                for fname, ftype in inherited_fields.items():
                    if actor_scope.lookup_local(fname) is None:
                        actor_scope.define(Symbol(name=fname, kind="field",
                                                   type_ref=ftype,
                                                   extra={"inherited_from": parent_name}))
                merged = dict(inherited_fields)
                merged.update(sym.extra.get("fields", {}))
                sym.extra["fields"] = merged

        for scenario_name, scope in self.scenario_scopes.items():
            scenario_sym = self.global_scope.lookup(scenario_name)
            scenario_decl: nodes.ScenarioDecl = scenario_sym.decl
            for binding in scenario_decl.actors:
                type_sym = self.global_scope.lookup(binding.type_name)
                if type_sym is None or type_sym.kind not in ("actor", "type"):
                    raise SemanticError(
                        f"Scenario '{scenario_name}' binds '{binding.name}' to "
                        f"unknown type '{binding.type_name}'"
                    )
                binding.attributes = self._resolve_keep_constraints(  # type: ignore[attr-defined]
                    binding.constraints, type_sym, binding.name
                )
            for var in scenario_decl.variables:
                if var.initializer is not None:
                    self._check_expr(var.initializer, scope)

    def _resolve_keep_constraints(self, constraints, actor_sym, binding_name):
        attrs: Dict[str, object] = {}
        fields = actor_sym.extra.get("fields", {})
        for c in constraints:
            expr = c.expr
            if not (isinstance(expr, nodes.BinaryExpr) and expr.op == "=="):
                continue
            field_name = _extract_it_field(expr.left)
            if field_name is None:
                field_name = _extract_it_field(expr.right)
                value_expr = expr.left
            else:
                value_expr = expr.right
            if field_name is None:
                continue
            if field_name not in fields:
                raise SemanticError(
                    f"keep(it.{field_name} == ...) refers to unknown field on "
                    f"'{actor_sym.name}' (actor binding '{binding_name}')"
                )
            attrs[field_name] = _literal_value(value_expr)
        return attrs

    def _check_expr(self, expr: nodes.Expr, scope: Scope) -> None:
        if isinstance(expr, nodes.Identifier):
            if expr.name == "it":
                return
            if scope.lookup(expr.name) is None:
                return
        elif isinstance(expr, nodes.BinaryExpr):
            self._check_expr(expr.left, scope)
            self._check_expr(expr.right, scope)
        elif isinstance(expr, nodes.UnaryExpr):
            self._check_expr(expr.operand, scope)
        elif isinstance(expr, nodes.Call):
            self._check_expr(expr.target, scope)
            for a in expr.args.positional:
                self._check_expr(a, scope)
            for a in expr.args.named.values():
                self._check_expr(a, scope)
        elif isinstance(expr, nodes.MemberAccess):
            self._check_expr(expr.target, scope)

    def finalise(self, user_ast: nodes.OscFile) -> AnnotatedScenario:
        scenarios = user_ast.scenarios
        if not scenarios:
            raise SemanticError("No scenario declaration in user script")
        scenario = scenarios[0]
        scope = self.scenario_scopes[scenario.name]
        return AnnotatedScenario(scenario=scenario,
                                 scope=scope,
                                 global_scope=self.global_scope,
                                 unit_table=self.unit_table,
                                 enum_table=self.enum_table)


def _extract_it_field(expr: nodes.Expr) -> Optional[str]:
    if isinstance(expr, nodes.MemberAccess) and isinstance(expr.target, nodes.Identifier):
        if expr.target.name == "it":
            return expr.field
    return None


def _literal_value(expr: nodes.Expr) -> object:
    if isinstance(expr, nodes.StringLit):
        return expr.value
    if isinstance(expr, nodes.NumLit):
        return int(expr.value) if expr.is_int else expr.value
    if isinstance(expr, nodes.BoolLit):
        return expr.value
    if isinstance(expr, nodes.PhysicalLiteral):
        return (expr.value, expr.unit)
    if isinstance(expr, nodes.Identifier):
        return expr.name
    if isinstance(expr, nodes.QualifiedIdentifier):
        return expr.name
    return expr
