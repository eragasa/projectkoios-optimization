"""Regression gate for tests that consume production modules."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_tests_do_not_import_private_production_symbols() -> None:
    """Keep tests coupled to public behavior rather than private decomposition."""
    violations: list[str] = []
    tests_root = Path(__file__).parent

    for path in sorted(tests_root.rglob("*.py")):
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(module):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module is None or not node.module.startswith("projectkoios"):
                continue
            for imported in node.names:
                if imported.name.startswith("_"):
                    violations.append(
                        f"{path.relative_to(tests_root)}:{node.lineno}:{imported.name}"
                    )

    assert violations == []


def test_tests_do_not_manufacture_frozen_record_state() -> None:
    """Require explicit constructors or typed probes for invalid-state tests."""
    violations: list[str] = []
    tests_root = Path(__file__).parent

    for path in sorted(tests_root.rglob("*.py")):
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom) and node.module == "dataclasses":
                for imported in node.names:
                    if imported.name == "replace":
                        violations.append(
                            f"{path.relative_to(tests_root)}:{node.lineno}:replace"
                        )
            if not isinstance(node, ast.Call) or not isinstance(
                node.func,
                ast.Attribute,
            ):
                continue
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "object"
                and node.func.attr in {"__new__", "__setattr__"}
            ):
                violations.append(
                    f"{path.relative_to(tests_root)}:{node.lineno}:"
                    f"object.{node.func.attr}"
                )

    assert violations == []
