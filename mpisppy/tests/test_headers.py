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
from addheader.add import FileFinder, TextFileModifier, detect_files

import mpisppy

yaml = pytest.importorskip("yaml")
addheader = pytest.importorskip("addheader")

def _patterns(root, conf_name):
    with open(root / conf_name) as f:
        return yaml.safe_load(f)["patterns"]


def _nonempty(paths):
    return [p for p in paths if p.stat().st_size != 0]


def test_headers():
    root = Path(mpisppy.__file__).parent.parent

    conf = _patterns(root, "addheader.yml")
    has_header, missing_header = detect_files(FileFinder(root, glob_patterns=conf))
    nonempty_missing_header = _nonempty(missing_header)
    print(f"{nonempty_missing_header=}")
    assert not nonempty_missing_header


def test_cpp_headers():
    # detect_files only knows # comments, so the C++ files (// comments,
    # addheader-cpp.yml) are checked with a modifier of their own.
    root = Path(mpisppy.__file__).parent.parent

    conf = _patterns(root, "addheader-cpp.yml")
    modifier = TextFileModifier(comment_prefix="//", delim_char="/")
    finder = FileFinder(root, glob_patterns=conf)
    assert finder.files, "no C++ files found"
    missing_header = [p for p in finder.files if not modifier.detect(p)]
    nonempty_missing_header = _nonempty(missing_header)
    print(f"{nonempty_missing_header=}")
    assert not nonempty_missing_header
