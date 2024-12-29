###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

from pathlib import Path

import pytest

from addheader.add import FileFinder, detect_files
import mpisppy

yaml = pytest.importorskip("yaml")
addheader = pytest.importorskip("addheader")

def test_headers():
    root = Path(mpisppy.__file__).parent.parent

    conf = root / "addheader.yml"
    with open(conf) as f:
        conf = yaml.safe_load(f)
    conf = conf["patterns"]

    has_header, missing_header = detect_files(FileFinder(root, glob_patterns=conf))
    nonempty_missing_header = []
    for p in missing_header:
        if p.stat().st_size == 0:
            continue
        nonempty_missing_header.append(p)
    print(f"{nonempty_missing_header=}")
    assert not nonempty_missing_header
