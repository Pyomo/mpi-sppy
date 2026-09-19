###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""The shipped OOTB policy files must survive packaging.

The policies are JSON data, not code, so the package declarations do not pick
them up on their own: before [tool.setuptools.package-data] was added, the
wheel contained mpisppy/generic/out_of_the_box.py but no ootb_policies/, and a
pip-installed ``--out-of-the-box`` died in load_policy() with FileNotFoundError
-- the first command the docs tell a new user to run.

No test that imports the checkout can see this, because there the JSON is
simply present. So these tests build a wheel and look inside it, then load a
policy with ONLY the wheel's contents importable. Its own file because it
builds a wheel (about a second) rather than exercising mpi-sppy.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

# repo root: .../mpisppy/tests/this_file.py -> up two
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# What setuptools needs to build the wheel, per pyproject.toml.
_SOURCES = ["pyproject.toml", "README.md", "LICENSE.md", "mpi_one_sided_test.py"]


def _staged_source(dest):
    """Copy the minimum source needed to build, into a clean directory.

    Building the checkout in place would reuse its build/lib/, where setuptools
    leaves the files it copied on a previous build and never prunes ones that
    are no longer included. A stale build/lib/ makes this test pass even with
    the package-data declaration deleted -- it did, until the build was staged.
    """
    for name in _SOURCES:
        src = os.path.join(_REPO, name)
        if os.path.exists(src):        # only pyproject.toml is truly required
            shutil.copy2(src, os.path.join(dest, name))
    shutil.copytree(
        os.path.join(_REPO, "mpisppy"), os.path.join(dest, "mpisppy"),
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return dest


def _build_wheel(outdir):
    """Build a wheel from a staged copy of the checkout and return its path.

    --no-cache-dir because pip would otherwise hand back a wheel built from an
    earlier version of pyproject.toml, which is exactly the file under test.
    """
    with tempfile.TemporaryDirectory() as staged:
        _staged_source(staged)
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps",
             "--no-build-isolation", "--no-cache-dir", "-w", outdir, staged],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"could not build the wheel (rc={proc.returncode}):\n"
                f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
            )
    wheels = [f for f in os.listdir(outdir) if f.endswith(".whl")]
    if len(wheels) != 1:
        raise AssertionError(f"expected one wheel in {outdir}, got {wheels}")
    return os.path.join(outdir, wheels[0])


class TestPolicyIsPackaged(unittest.TestCase):
    """A built wheel carries the policies, and they load from it."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.wheel = _build_wheel(cls._tmp.name)
        with zipfile.ZipFile(cls.wheel) as zf:
            cls.names = zf.namelist()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_policy_json_in_wheel(self):
        policies = [n for n in self.names
                    if n.startswith("mpisppy/generic/ootb_policies/")
                    and n.endswith(".json")]
        self.assertTrue(
            policies,
            "the built wheel contains no mpisppy/generic/ootb_policies/*.json, "
            "so --out-of-the-box cannot work for a pip-installed user; see "
            "[tool.setuptools.package-data] in pyproject.toml",
        )

    def test_every_checkout_policy_is_shipped(self):
        """A policy added to the checkout later must ship too."""
        src = os.path.join(_REPO, "mpisppy", "generic", "ootb_policies")
        on_disk = {f for f in os.listdir(src) if f.endswith(".json")}
        in_wheel = {os.path.basename(n) for n in self.names
                    if n.startswith("mpisppy/generic/ootb_policies/")
                    and n.endswith(".json")}
        self.assertEqual(
            on_disk - in_wheel, set(),
            "policy files in the checkout that the wheel does not ship",
        )

    def test_load_policy_from_installed_layout(self):
        """load_policy() works with only the wheel's contents importable.

        Run in a subprocess whose cwd and sys.path exclude the checkout, so an
        accidental fallback to the source directory cannot make this pass.
        """
        with tempfile.TemporaryDirectory() as unpacked:
            with zipfile.ZipFile(self.wheel) as zf:
                zf.extractall(unpacked)
            env = dict(os.environ)
            env["PYTHONPATH"] = unpacked
            proc = subprocess.run(
                [sys.executable, "-c",
                 "import json, mpisppy.generic.out_of_the_box as o;"
                 "p = o.load_policy();"
                 "print('OOTB_POLICY_KEYS=' + json.dumps(sorted(p)))"],
                capture_output=True, text=True, cwd=unpacked, env=env,
            )
            self.assertEqual(
                proc.returncode, 0,
                f"load_policy() failed against the installed layout:\n"
                f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}",
            )
            # the module really came from the unpacked wheel, not the checkout
            marker = "OOTB_POLICY_KEYS="
            lines = [ln for ln in proc.stdout.splitlines()
                     if ln.startswith(marker)]
            self.assertEqual(
                len(lines), 1,
                f"expected one {marker} line, got {len(lines)}; stdout was:\n"
                f"{proc.stdout[-3000:]}",
            )
            keys = json.loads(lines[0][len(marker):])
            for expected in ("bundle_sizing", "effort_scaling"):
                self.assertIn(expected, keys)


if __name__ == "__main__":
    unittest.main()
