"""Check the built wheel and sdist of emeans-rocm before they are published.

    python tools/check_dist.py dist/

Run from the repository root after building (``uv build -o dist .``). Fails when:

* the wheel's ``emeans_rocm/`` files differ from the git-tracked ``contrib/e_means``
  files minus ``figure/`` (README images stay out of the package);
* the sdist holds anything beyond ``contrib/e_means`` (minus ``figure/``), ``LICENSE``,
  ``README.md``, ``pyproject.toml``, ``PKG-INFO`` and ``.gitignore`` (so no local files
  such as ``.planning/`` or ``docs/`` are published);
* the version in either file differs from ``pyproject.toml``, or there is not exactly
  one wheel and one sdist.
"""

import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

PACKAGE_SOURCE = "contrib/e_means/"
LEFT_OUT = "figure/"
SDIST_FILES = {"LICENSE", "README.md", "pyproject.toml", "PKG-INFO", ".gitignore"}


def _project_version(root):
    match = re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.M)
    if match is None:
        raise SystemExit("no version in pyproject.toml")
    return match.group(1)


def _tracked(root):
    out = subprocess.run(["git", "ls-files", PACKAGE_SOURCE], cwd=root, check=True,
                         capture_output=True, text=True).stdout.split()
    files = {name[len(PACKAGE_SOURCE):] for name in out}
    return {name for name in files if not name.startswith(LEFT_OUT)}


def _one(dist, pattern):
    found = sorted(dist.glob(pattern))
    if len(found) != 1:
        raise SystemExit(f"expected one {pattern} in {dist}, found {[p.name for p in found]}")
    return found[0]


def check(dist, root):
    errors = []
    version = _project_version(root)
    tracked = _tracked(root)

    wheel = _one(dist, "*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
        text = archive.read(metadata).decode() if metadata else ""
    package = {n[len("emeans_rocm/"):] for n in names if n.startswith("emeans_rocm/")}
    other = sorted(n for n in names if not n.startswith("emeans_rocm/") and ".dist-info/" not in n)
    if package != tracked:
        errors.append(f"wheel: missing {sorted(tracked - package)}, extra {sorted(package - tracked)}")
    if other:
        errors.append(f"wheel: files outside emeans_rocm/: {other}")
    if f"\nVersion: {version}\n" not in text:
        errors.append(f"wheel: METADATA version is not {version}")

    sdist = _one(dist, "*.tar.gz")
    with tarfile.open(sdist) as archive:
        members = [m.name.split("/", 1)[1] for m in archive.getmembers()
                   if m.isfile() and "/" in m.name]
        pkg_info = archive.extractfile(next(m for m in archive.getmembers()
                                            if m.name.endswith("/PKG-INFO"))).read().decode()
    unexpected = sorted(m for m in members
                        if m not in SDIST_FILES and not m.startswith(PACKAGE_SOURCE))
    source = {m[len(PACKAGE_SOURCE):] for m in members if m.startswith(PACKAGE_SOURCE)}
    if unexpected:
        errors.append(f"sdist: unexpected files {unexpected}")
    if source != tracked:
        errors.append(f"sdist: missing {sorted(tracked - source)}, extra {sorted(source - tracked)}")
    if f"\nVersion: {version}\n" not in pkg_info:
        errors.append(f"sdist: PKG-INFO version is not {version}")

    for error in errors:
        print("FAIL", error)
    if not errors:
        print(f"ok: {wheel.name} and {sdist.name} hold the {len(tracked)} tracked package files, "
              f"version {version}")
    return not errors


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(0 if check(Path(sys.argv[1]), Path(__file__).resolve().parents[1]) else 1)
