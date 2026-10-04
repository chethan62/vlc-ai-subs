"""The dialog's engine list, the registry and the files on disk agree.

An engine lives in four places at once — the lua table the dialog draws, the
`_ENGINES` registry, its backend module and its installer — and removing one by
hand is exactly the edit that leaves a dead entry in a working dialog (or a
registry pointing at a module that is gone). The lua UI harness that used to catch
a half-removed engine went with the local venvs, so this is the cheap check.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backends import _ENGINES, _INSTALL_HINT                            # noqa: E402


def _dialog_engines() -> set:
    """The ids in aisubs.lua's ENGINES table ('auto' is a dialog-only pseudo-engine)"""
    lua = open(os.path.join(ROOT, "aisubs.lua"), encoding="utf-8").read()
    block = re.search(r"^local ENGINES = \{(.*?)^\}", lua, re.S | re.M).group(1)
    return set(re.findall(r'\{\s*"([a-z_]+)"', block)) - {"auto"}


def test_the_dialog_offers_exactly_the_registry():
    assert _dialog_engines() == set(_ENGINES)


def test_every_registered_engine_has_its_module_on_disk():
    for name, (module, _cls, _label) in _ENGINES.items():
        path = os.path.join(ROOT, module.replace(".", os.sep) + ".py")
        assert os.path.isfile(path), f"{name} → {module} is registered but missing"


def test_every_registered_engine_has_an_installer():
    assert set(_INSTALL_HINT) >= set(_ENGINES), "an engine with no install hint cannot be fixed by a user"
    for name, hint in _INSTALL_HINT.items():
        for script in re.findall(r"\./(install-[\w-]+\.sh)", hint):
            assert os.path.isfile(os.path.join(ROOT, script)), f"{name}'s installer {script} is gone"
