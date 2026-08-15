"""AST transformer - applies the Visitor pattern to the ANTLR parse tree."""
from __future__ import annotations

import re
from typing import List, Optional

from antlr4 import CommonTokenStream, FileStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from . import nodes
from .generated.openscenario2Lexer import openscenario2Lexer
from .generated.openscenario2Parser import openscenario2Parser
from .generated.openscenario2Visitor import openscenario2Visitor


class _CollectingErrorListener(ErrorListener):
    def __init__(self):
        super().__init__()
        self.errors: List[str] = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(f"line {line}:{column} {msg}")


class ParseError(RuntimeError):
    pass


_PHYSICAL_RE = re.compile(
    r"^(?P<num>[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?|[+-]?\d+)"
    r"(?P<unit>[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)?)$"
)


class ASTTransformer(openscenario2Visitor):
    """Builds an :class:`nodes.OscFile` from a parse tree."""

    def visitOsc_file(self, ctx):
        oscfile = nodes.OscFile()
        for child in ctx.prelude_statement():
            imp = self.visit(child)
            if imp is not None:
                oscfile.imports.append(imp)
        for child in ctx.main_statement():
            decl = self.visit(child)
            if decl is None:
                continue
            if isinstance(decl, list):
                oscfile.declarations.extend(decl)
            else:
                oscfile.declarations.append(decl)
        return oscfile

    def visitPrelude_statement(self, ctx):
        return self.visit(ctx.import_statement())

    def visitImport_statement(self, ctx):
        ref = ctx.import_reference()
        if ref.string_literal() is not None:
            return nodes.ImportStmt(_strip_string(ref.string_literal().getText()))
        return nodes.ImportStmt(ref.structured_identifier().getText())

    def visitMain_statement(self, ctx):
        if ctx.osc_declaration() is not None:
            return self.visit(ctx.osc_declaration())
        return None

    def visitOsc_declaration(self, ctx):
        for getter in (
            ctx.physical_type_declaration,
            ctx.unit_declaration,
            ctx.enum_declaration,
            ctx.struct_declaration,
            ctx.actor_declaration,
            ctx.action_declaration,
            ctx.scenario_declaration,
            ctx.modifier_declaration,
        ):
            child = getter()
            if child is not None:
                return self.visit(child)
        return None

    def visitPhysical_type_declaration(self, ctx):
        name = ctx.declared_type_name().getText()
        si = _parse_si(ctx.base_unit_specifier().si_base_unit_specifier())
        return nodes.PhysicalTypeDecl(name=name, si=si)

    def visitUnit_declaration(self, ctx):
        name = ctx.unit_name().getText()
        type_name = ctx.declared_type_name().getText()
        spec = ctx.unit_specifier().si_unit_specifier()
        si = _parse_si(spec)
        factor = 1.0
        offset = 0.0
        if spec.si_factor() is not None:
            f = spec.si_factor()
            if f.float_literal() is not None:
                factor = float(f.float_literal().getText())
            else:
                factor = float(f.integer_literal().getText())
        if spec.si_offset() is not None:
            o = spec.si_offset()
            if o.float_literal() is not None:
                offset = float(o.float_literal().getText())
            else:
                offset = float(o.integer_literal().getText())
        return nodes.UnitDecl(name=name, type_name=type_name, factor=factor, offset=offset, si=si)

    def visitEnum_declaration(self, ctx):
        name = ctx.enum_name().getText()
        members = [m.enum_member_name().getText() for m in ctx.enum_member_decl()]
        return nodes.EnumDecl(name=name, members=members)

    def visitStruct_declaration(self, ctx):
        name = ctx.struct_name(0).getText()
        inherits = ctx.struct_name(1).getText() if len(ctx.struct_name()) > 1 else None
        fields = self._collect_fields(ctx.struct_member_decl())
        return nodes.StructDecl(name=name, inherits=inherits, fields=fields)

    def visitActor_declaration(self, ctx):
        name = ctx.actor_name(0).getText()
        inherits = ctx.actor_name(1).getText() if len(ctx.actor_name()) > 1 else None
        fields = self._collect_fields(ctx.actor_member_decl())
        return nodes.ActorDecl(name=name, inherits=inherits, fields=fields)

    def visitAction_declaration(self, ctx):
        names = ctx.qualified_behavior_name()
        return nodes.ActionDecl(
            name=names[0].getText(),
            inherits=names[1].getText() if len(names) > 1 else None,
        )

    def visitModifier_declaration(self, ctx):
        name = ctx.modifier_name().getText()
        return nodes.ModifierDecl(name=name)

    def _collect_fields(self, members):
        fields = []
        for m in members:
            fd = m.field_declaration() if hasattr(m, "field_declaration") else None
            if fd is None:
                continue
            field_node = self._field_decl_to_node(fd)
            if field_node is not None:
                if isinstance(field_node, list):
                    fields.extend(field_node)
                else:
                    fields.append(field_node)
        return fields

    def _field_decl_to_node(self, fd_ctx):
        if fd_ctx.parameter_declaration() is not None:
            pd = fd_ctx.parameter_declaration()
            type_name = pd.type_declarator().getText()
            init = self.visit(pd.default_value().expression()) if pd.default_value() is not None else None
            return [
                nodes.ActorFieldDecl(name=n.getText(), type_name=type_name, initializer=init)
                for n in pd.field_name()
            ]
        if fd_ctx.variable_declaration() is not None:
            vd = fd_ctx.variable_declaration()
            type_name = vd.type_declarator().getText()
            init = None
            if vd.default_value() is not None:
                init = self.visit(vd.default_value().expression())
            return [
                nodes.ActorFieldDecl(name=n.getText(), type_name=type_name, initializer=init)
                for n in vd.field_name()
            ]
        return None

    def visitScenario_declaration(self, ctx):
        name = ctx.qualified_behavior_name(0).getText()
        inherits = ctx.qualified_behavior_name(1).getText() if len(ctx.qualified_behavior_name()) > 1 else None
        scenario = nodes.ScenarioDecl(name=name, inherits=inherits)

        for member in ctx.scenario_member_decl():
            fd = member.field_declaration() if hasattr(member, "field_declaration") else None
            if fd is None:
                continue
            self._add_scenario_field(scenario, fd)

        for bs in ctx.behavior_specification():
            if bs.do_directive() is not None:
                scenario.do = nodes.DoDirective(self._visit_do_member(bs.do_directive().do_member()))
        return scenario

    def _add_scenario_field(self, scenario: nodes.ScenarioDecl, fd_ctx):
        if fd_ctx.parameter_declaration() is not None:
            pd = fd_ctx.parameter_declaration()
            type_name = pd.type_declarator().getText()
            constraints: List[nodes.KeepConstraint] = []
            if pd.parameter_with_declaration() is not None:
                for w in pd.parameter_with_declaration().parameter_with_member():
                    kc = w.constraint_declaration().keep_constraint_declaration()
                    if kc is not None:
                        constraints.append(
                            nodes.KeepConstraint(self.visit(kc.constraint_expression().expression()))
                        )
            for fn in pd.field_name():
                scenario.actors.append(
                    nodes.ScenarioActorBinding(
                        name=fn.getText(),
                        type_name=type_name,
                        constraints=list(constraints),
                    )
                )
            return
        if fd_ctx.variable_declaration() is not None:
            vd = fd_ctx.variable_declaration()
            type_name = vd.type_declarator().getText()
            init = self.visit(vd.default_value().expression()) if vd.default_value() is not None else None
            for fn in vd.field_name():
                scenario.variables.append(
                    nodes.VarDecl(name=fn.getText(), type_name=type_name, initializer=init)
                )

    def _visit_do_member(self, dm_ctx) -> nodes.DoMember:
        if dm_ctx.composition() is not None:
            return self._visit_composition(dm_ctx.composition())
        if dm_ctx.behavior_invocation() is not None:
            return self._visit_behavior_invocation(dm_ctx.behavior_invocation())
        if dm_ctx.wait_directive() is not None:
            return nodes.Wait(self._visit_event_spec(dm_ctx.wait_directive().event_specification()))
        if dm_ctx.emit_directive() is not None:
            ed = dm_ctx.emit_directive()
            args = self._visit_argument_list(ed.argument_list()) if ed.argument_list() is not None else nodes.ArgList()
            return nodes.Emit(event=ed.event_name().getText(), args=args)
        return nodes.Wait(nodes.BoolLit(True))

    def _visit_composition(self, c_ctx) -> nodes.Composition:
        op = c_ctx.composition_operator().getText()
        members = [self._visit_do_member(m) for m in c_ctx.do_member()]
        return nodes.Composition(op=op, members=members)

    def _visit_behavior_invocation(self, bi_ctx) -> nodes.ActionCall:
        actor = bi_ctx.actor_expression().getText() if bi_ctx.actor_expression() is not None else None
        behaviour = bi_ctx.behavior_name().getText()
        args = self._visit_argument_list(bi_ctx.argument_list()) if bi_ctx.argument_list() is not None else nodes.ArgList()
        with_block: Optional[nodes.WithBlock] = None
        if bi_ctx.behavior_with_declaration() is not None:
            with_block = self._visit_with_block(bi_ctx.behavior_with_declaration())
        return nodes.ActionCall(actor=actor, behaviour=behaviour, args=args, with_block=with_block)

    def _visit_with_block(self, w_ctx) -> nodes.WithBlock:
        wb = nodes.WithBlock()
        for member in w_ctx.behavior_with_member():
            ma = member.modifier_application()
            if ma is not None:
                wb.modifiers.append(self._visit_modifier_application(ma))
                continue
            cd = member.constraint_declaration()
            if cd is not None and cd.keep_constraint_declaration() is not None:
                kc = cd.keep_constraint_declaration()
                wb.constraints.append(
                    nodes.KeepConstraint(self.visit(kc.constraint_expression().expression()))
                )
        return wb

    def _visit_modifier_application(self, ma_ctx) -> nodes.Modifier:
        name = ma_ctx.modifier_name().getText()
        args = self._visit_argument_list(ma_ctx.argument_list()) if ma_ctx.argument_list() is not None else nodes.ArgList()
        at = None
        if "at" in args.named:
            at = _identifier_text(args.named.pop("at"))
        return nodes.Modifier(name=name, args=args, at=at)

    def _visit_event_spec(self, es_ctx) -> nodes.Expr:
        if es_ctx.event_reference() is not None:
            return nodes.EventRef(es_ctx.event_reference().event_path().getText())
        if es_ctx.event_condition() is not None:
            ec = es_ctx.event_condition()
            if ec.rise_expression() is not None:
                return nodes.Rise(self.visit(ec.rise_expression().bool_expression().expression()))
            if ec.fall_expression() is not None:
                return nodes.Fall(self.visit(ec.fall_expression().bool_expression().expression()))
            if ec.elapsed_expression() is not None:
                return nodes.Elapsed(self.visit(ec.elapsed_expression().duration_expression().expression()))
            if ec.bool_expression() is not None:
                return self.visit(ec.bool_expression().expression())
        return nodes.BoolLit(True)

    def _visit_argument_list(self, al_ctx) -> nodes.ArgList:
        out = nodes.ArgList()
        if al_ctx is None:
            return out
        for pa in al_ctx.positional_argument():
            out.positional.append(self.visit(pa.expression()))
        for na in al_ctx.named_argument():
            out.named[na.argument_name().getText()] = self.visit(na.expression())
        return out

    # ----- expressions -----

    def visitExpression(self, ctx):
        if ctx.ternary_op_exp() is not None:
            return self.visit(ctx.ternary_op_exp())
        return self.visit(ctx.implication())

    def visitTernary_op_exp(self, ctx):
        cond = self.visit(ctx.implication())
        then = self.visit(ctx.expression(0))
        else_ = self.visit(ctx.expression(1))
        return nodes.BinaryExpr(op="?:", left=cond, right=nodes.BinaryExpr(op="branch", left=then, right=else_))

    def visitImplication(self, ctx):
        return self._left_assoc_binop(ctx, ctx.disjunction, "=>")

    def visitDisjunction(self, ctx):
        return self._left_assoc_binop(ctx, ctx.conjunction, "or")

    def visitConjunction(self, ctx):
        return self._left_assoc_binop(ctx, ctx.inversion, "and")

    def visitInversion(self, ctx):
        if ctx.inversion() is not None:
            return nodes.UnaryExpr(op="not", operand=self.visit(ctx.inversion()))
        return self.visit(ctx.relation())

    def visitRelation(self, ctx):
        if ctx.relational_op() is None:
            return self.visit(ctx.sum_exp())
        left = self.visit(ctx.relation())
        right = self.visit(ctx.sum_exp())
        return nodes.BinaryExpr(op=ctx.relational_op().getText(), left=left, right=right)

    def visitSum_exp(self, ctx):
        if ctx.additive_op() is None:
            return self.visit(ctx.term())
        left = self.visit(ctx.sum_exp())
        right = self.visit(ctx.term())
        return nodes.BinaryExpr(op=ctx.additive_op().getText(), left=left, right=right)

    def visitTerm(self, ctx):
        if ctx.multiplicative_op() is None:
            return self.visit(ctx.factor())
        left = self.visit(ctx.term())
        right = self.visit(ctx.factor())
        return nodes.BinaryExpr(op=ctx.multiplicative_op().getText(), left=left, right=right)

    def visitFactor(self, ctx):
        if ctx.factor() is not None:
            return nodes.UnaryExpr(op="-", operand=self.visit(ctx.factor()))
        return self.visit(ctx.postfix_exp())

    def visitPrimary_exp_pe(self, ctx):
        return self.visit(ctx.primary_exp())

    def visitField_access_pe(self, ctx):
        target = self.visit(ctx.postfix_exp())
        return nodes.MemberAccess(target=target, field=ctx.field_name().getText())

    def visitFunction_application_pe(self, ctx):
        target = self.visit(ctx.postfix_exp())
        args = self._visit_argument_list(ctx.argument_list()) if ctx.argument_list() is not None else nodes.ArgList()
        return nodes.Call(target=target, args=args)

    def visitElement_access_pe(self, ctx):
        target = self.visit(ctx.postfix_exp())
        index = self.visit(ctx.expression())
        return nodes.Call(
            target=nodes.MemberAccess(target=target, field="__getitem__"),
            args=nodes.ArgList(positional=[index]),
        )

    def visitCast_exp_pe(self, ctx):
        return self.visit(ctx.postfix_exp())

    def visitType_test_exp_pe(self, ctx):
        return self.visit(ctx.postfix_exp())

    def visitPrimary_exp(self, ctx):
        if ctx.value_exp() is not None:
            return self.visit(ctx.value_exp())
        if ctx.qualified_identifier() is not None:
            qi = ctx.qualified_identifier()
            text = qi.getText()
            if "." in text or "::" in text:
                return nodes.QualifiedIdentifier(parts=text.replace("::", ".").split("."))
            return nodes.Identifier(name=text)
        if ctx.expression() is not None:
            return self.visit(ctx.expression())
        if ctx.getText() == "it":
            return nodes.Identifier(name="it")
        return nodes.Identifier(name=ctx.getText())

    def visitValue_exp(self, ctx):
        if ctx.integer_literal() is not None:
            return nodes.NumLit(value=float(ctx.integer_literal().getText()), is_int=True)
        if ctx.float_literal() is not None:
            return nodes.NumLit(value=float(ctx.float_literal().getText()), is_int=False)
        if ctx.physical_literal() is not None:
            return self._parse_physical(ctx.physical_literal().getText())
        if ctx.bool_literal() is not None:
            return nodes.BoolLit(ctx.bool_literal().getText() == "true")
        if ctx.string_literal() is not None:
            return nodes.StringLit(_strip_string(ctx.string_literal().getText()))
        if ctx.enum_value_reference() is not None:
            text = ctx.enum_value_reference().getText()
            if "!" in text:
                _, member = text.split("!", 1)
                return nodes.Identifier(name=member)
            return nodes.Identifier(name=text)
        return nodes.StringLit(ctx.getText())

    def _left_assoc_binop(self, ctx, child_getter, op):
        children = child_getter()
        if len(children) == 1:
            return self.visit(children[0])
        result = self.visit(children[0])
        for c in children[1:]:
            result = nodes.BinaryExpr(op=op, left=result, right=self.visit(c))
        return result

    def _parse_physical(self, text: str) -> nodes.PhysicalLiteral:
        m = _PHYSICAL_RE.match(text)
        if not m:
            raise ParseError(f"Cannot parse physical literal: {text!r}")
        return nodes.PhysicalLiteral(value=float(m.group("num")), unit=m.group("unit"))


def _strip_string(text: str) -> str:
    if text.startswith('"""') or text.startswith("'''"):
        return text[3:-3]
    return text[1:-1]


def _identifier_text(expr: nodes.Expr) -> str:
    if isinstance(expr, nodes.Identifier):
        return expr.name
    if isinstance(expr, nodes.QualifiedIdentifier):
        return expr.name
    if isinstance(expr, nodes.StringLit):
        return expr.value
    return str(expr)


def _parse_si(ctx) -> dict:
    si: dict = {}
    base = ctx.si_base_exponent_list() if hasattr(ctx, "si_base_exponent_list") else None
    if base is None:
        return si
    for e in base.si_base_exponent():
        si[e.si_base_unit_name().getText()] = int(e.integer_literal().getText())
    return si


def _build_parser(stream):
    lexer = openscenario2Lexer(stream)
    listener = _CollectingErrorListener()
    lexer.removeErrorListeners()
    lexer.addErrorListener(listener)
    tokens = CommonTokenStream(lexer)
    parser = openscenario2Parser(tokens)
    parser.removeErrorListeners()
    parser.addErrorListener(listener)
    return parser, listener


def parse_string(source: str) -> nodes.OscFile:
    stream = InputStream(source)
    parser, listener = _build_parser(stream)
    tree = parser.osc_file()
    if listener.errors:
        raise ParseError("; ".join(listener.errors))
    return ASTTransformer().visit(tree)


def parse_file(path: str) -> nodes.OscFile:
    stream = FileStream(path, encoding="utf-8")
    parser, listener = _build_parser(stream)
    tree = parser.osc_file()
    if listener.errors:
        raise ParseError(f"{path}: {'; '.join(listener.errors)}")
    return ASTTransformer().visit(tree)
