# The package reports the version declared in pyproject.toml, and every module in it imports.

import importlib
import pkgutil
import tomllib
from pathlib import Path

import needtoknow

ROOT = Path(__file__).parent.parent


def test_version_matches_pyproject() -> None:
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert needtoknow.__version__ == declared


def test_every_module_imports() -> None:
    names = [module.name for module in pkgutil.iter_modules(needtoknow.__path__)]

    assert {"auth", "audit", "config", "graph", "jev", "providers", "quota"} <= set(names)
    for name in names:
        importlib.import_module(f"needtoknow.{name}")
