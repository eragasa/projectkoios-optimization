"""Regression gates for eliminated production-code smells."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROTECTED_NORMALIZATION_FILES = {
    "projectkoios/optimization/clustering/gmm.py",
    "projectkoios/optimization/engines/loss.py",
    "projectkoios/optimization/engines/records.py",
}


def _type_parameter_name(parameter: ast.type_param) -> str:
    if isinstance(parameter, ast.TypeVar):
        return parameter.name
    if isinstance(parameter, ast.ParamSpec):
        return parameter.name
    if isinstance(parameter, ast.TypeVarTuple):
        return parameter.name
    raise AssertionError("unsupported AST type parameter")


def _annotation_contains(annotation: ast.expr, name: str) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id == name for node in ast.walk(annotation)
    )


def test_production_annotations_avoid_any_and_unprotected_object() -> None:
    source_root = Path(__file__).parents[1] / "src" / "python"
    violations: list[str] = []

    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root).as_posix()
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(module):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            annotations = [
                argument.annotation
                for argument in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
                if argument.annotation is not None
            ]
            if node.args.vararg is not None and node.args.vararg.annotation is not None:
                annotations.append(node.args.vararg.annotation)
            if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
                annotations.append(node.args.kwarg.annotation)
            if node.returns is not None:
                annotations.append(node.returns)
            for annotation in annotations:
                if _annotation_contains(annotation, "Any"):
                    violations.append(f"{relative}:{node.lineno}:Any")
                if (
                    relative not in PROTECTED_NORMALIZATION_FILES
                    and _annotation_contains(annotation, "object")
                ):
                    violations.append(f"{relative}:{node.lineno}:object")

    assert violations == []


def test_production_returns_do_not_expose_callable_type_parameters() -> None:
    source_root = Path(__file__).parents[1] / "src" / "python"
    violations: list[str] = []

    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root).as_posix()
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = {
            child: parent
            for parent in ast.walk(module)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(module):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.returns is None:
                continue
            parameter_names = {
                _type_parameter_name(parameter) for parameter in node.type_params
            }
            parent = parents.get(node)
            while parent is not None:
                if isinstance(parent, ast.ClassDef):
                    parameter_names.update(
                        _type_parameter_name(parameter)
                        for parameter in parent.type_params
                    )
                parent = parents.get(parent)
            for parameter_name in parameter_names:
                if _annotation_contains(node.returns, parameter_name):
                    violations.append(
                        f"{relative}:{node.lineno}:return {parameter_name}"
                    )

    assert violations == []


def test_production_avoids_assertions_reflection_and_frozen_mutation() -> None:
    source_root = Path(__file__).parents[1] / "src" / "python"
    violations: list[str] = []

    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root).as_posix()
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(module):
            if isinstance(node, ast.Assert):
                violations.append(f"{relative}:{node.lineno}:assert")
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "__getattr__"
            ):
                violations.append(f"{relative}:{node.lineno}:__getattr__")
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "getattr":
                violations.append(f"{relative}:{node.lineno}:getattr")
            if (
                relative not in PROTECTED_NORMALIZATION_FILES
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "__setattr__"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "object"
            ):
                violations.append(f"{relative}:{node.lineno}:object.__setattr__")

    assert violations == []
