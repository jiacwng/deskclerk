# The package imports, reports the version declared in pyproject.toml, and carries the files it
# reads at runtime, so it works installed and not only from a source tree.

import tomllib
from pathlib import Path

import needtoknow
from needtoknow.bird import POLICIES
from needtoknow.provision import SCHEMA


def test_version_matches_pyproject() -> None:
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert needtoknow.__version__ == declared


def test_the_sql_the_package_runs_ships_with_it() -> None:
    for resource in (SCHEMA, POLICIES):
        assert resource.read_text(encoding="utf-8").strip()
