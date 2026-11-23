"""pytest plugin: run tests/ against the installed emeans_rocm package.

The tests import ``faiss.contrib.e_means``, as upstream's do. With this plugin every
``faiss.contrib.e_means[.sub]`` import returns the module object of ``emeans_rocm[.sub]``,
so each file is loaded once and ``mock.patch("faiss.contrib.e_means.<sub>.<name>")``
patches the code the package runs. From the repository root, with the package installed:

    cd tests
    PYTHONPATH=../tools python -m pytest -q -p pkg_alias $(ls *e_means*.py | grep -v -e utils -e conditions)

Child interpreters (``torch.multiprocessing.spawn`` in the distributed tests, ``subprocess``)
get the same alias: the plugin puts a ``sitecustomize.py`` that imports it on ``PYTHONPATH``.
This shadows any other ``sitecustomize`` module for the test run.

Three tests start a separate interpreter and add the module's folder to
``faiss.contrib.__path__``. They check the faiss layout itself, which this plugin does not
reach, so they are skipped here and still run in the normal (symlinked) setup.
"""

import atexit
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

OLD = "faiss.contrib.e_means"
NEW = "emeans_rocm"

FAISS_LAYOUT_ONLY = {
    "test_e_means.py::TestEmeans::test_numpy_training_does_not_import_torch",
    "torch_test_e_means.py::TestTorchEmeans::test_optional_triton_dependency_isolation",
    "torch_test_e_means_cute.py::TestEmeansCute::test_optional_stack_and_device_scope",
}


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name != OLD and not name.startswith(OLD + "."):
            return None
        real = importlib.util.find_spec(NEW + name[len(OLD):])
        if real is None:
            return None
        spec = importlib.util.spec_from_loader(
            name, self, origin=real.origin,
            is_package=real.submodule_search_locations is not None)
        spec.has_location = real.has_location
        return spec

    def create_module(self, spec):
        module = importlib.import_module(NEW + spec.name[len(OLD):])
        # importlib stamps the alias spec on the module after this returns; keep the real
        # one so exec_module can put it back (find_spec would return the stamped one).
        spec.loader_state = module.__spec__
        return module

    def exec_module(self, module):
        module.__spec__ = module.__spec__.loader_state


if any(n == OLD or n.startswith(OLD + ".") for n in sys.modules):
    raise RuntimeError(f"{OLD} was imported before the pkg_alias plugin loaded")
sys.meta_path.insert(0, _Alias())

# Child interpreters inherit the environment: a sitecustomize there imports this module first.
if "PKG_ALIAS_SITE" not in os.environ:
    _site = tempfile.mkdtemp(prefix="pkg_alias_")
    Path(_site, "sitecustomize.py").write_text(
        "import sys\nsys.path.insert(0, %r)\nimport pkg_alias  # noqa: F401\n"
        % str(Path(__file__).resolve().parent))
    os.environ["PKG_ALIAS_SITE"] = _site
    atexit.register(shutil.rmtree, _site, True)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in (_site, os.environ.get("PYTHONPATH")) if p)


def pytest_report_header(config):
    package = importlib.import_module(NEW)
    try:
        version = importlib.metadata.version("emeans-rocm")
    except importlib.metadata.PackageNotFoundError:
        version = "not installed (found on sys.path)"
    return f"pkg_alias: {OLD} -> {NEW} {version} at {os.path.dirname(package.__file__)}"


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="checks the faiss.contrib layout; not run against the package")
    for item in items:
        if item.nodeid.split("/")[-1] in FAISS_LAYOUT_ONLY:
            item.add_marker(skip)
