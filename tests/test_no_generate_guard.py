from __future__ import annotations

import ast
from pathlib import Path

CORE_MODULES = (
    Path("neural_native/app"),
    Path("neural_native/bridge"),
    Path("neural_native/llm"),
    Path("neural_native/cli.py"),
    Path("scripts/extract_features.py"),
    Path("scripts/run_scripted_demo.py"),
)


class GenerateCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.matches: list[tuple[int, int]] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "generate":
            self.matches.append((node.lineno, node.col_offset))
        self.generic_visit(node)


def test_core_routing_path_never_calls_model_generate() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    files: list[Path] = []
    for module_path in CORE_MODULES:
        absolute = repo_root / module_path
        if absolute.is_dir():
            files.extend(absolute.rglob("*.py"))
        else:
            files.append(absolute)

    offenders: list[str] = []
    for path in files:
        visitor = GenerateCallVisitor()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        offenders.extend(
            f"{path.relative_to(repo_root)}:{line}" for line, _column in visitor.matches
        )

    assert offenders == []


def test_core_router_does_not_use_keyword_or_regex_text_parser() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    core_files = [
        repo_root / "neural_native" / "app" / "vector_port.py",
        repo_root / "neural_native" / "bridge" / "router.py",
    ]

    offenders: list[str] = []
    for path in core_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == "re" for alias in node.names):
                    offenders.append(f"{path.relative_to(repo_root)} imports re")
            if isinstance(node, ast.ImportFrom) and node.module == "re":
                offenders.append(f"{path.relative_to(repo_root)} imports from re")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"startswith", "endswith", "find", "lower", "split"}:
                    offenders.append(
                        f"{path.relative_to(repo_root)}:{node.lineno} uses text parser method"
                    )

    assert offenders == []
