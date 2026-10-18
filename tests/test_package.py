# The package imports, reports the version declared in pyproject.toml, carries the files it
# reads at runtime, so it works installed and not only from a source tree, and its product
# modules never import the evaluation or the packages only the eval extra installs.

import ast
import tomllib
from pathlib import Path

import needtoknow
from needtoknow.bird import POLICIES
from needtoknow.provision import READER_SCHEMA, SCHEMA

ROOT = Path(__file__).parent.parent
EVAL_ONLY = {"scipy", "thefuzz", "pyarrow"}


def test_version_matches_pyproject() -> None:
    pyproject = ROOT / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert needtoknow.__version__ == declared


def test_the_sql_the_package_runs_ships_with_it() -> None:
    for resource in (SCHEMA, READER_SCHEMA, POLICIES):
        assert resource.read_text(encoding="utf-8").strip()


def test_the_product_never_imports_the_evaluation() -> None:
    imported: set[str] = set()
    for path in (ROOT / "src" / "needtoknow").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                for alias in node.names:
                    imported.add(f"{node.module}.{alias.name}")

    assert "needtoknow.agent" in imported
    for name in imported:
        assert name.split(".")[0] not in EVAL_ONLY, name
        assert not name.startswith("needtoknow.evaluation"), name
