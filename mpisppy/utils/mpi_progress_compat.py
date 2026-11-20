###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

"""Compare a native progress extension's build signature with mpi4py."""


def check_native_mpi_compatibility(extension, mpi):
    """Reject known MPI mismatches before passing a communicator to C.

    Matching values are a guard against mismatches, not proof that two loaded
    copies of an MPI library share the same runtime state.
    """
    sizeof = getattr(mpi, "_sizeof", None)
    if not callable(sizeof):
        raise RuntimeError(
            "mpi4py does not expose MPI_Comm size; cannot check the native "
            "MPI progress extension"
        )

    expected = (
        ("BUILD_MPI_COMM_SIZE", sizeof(mpi.Comm)),
        ("BUILD_MPI_ANY_SOURCE", mpi.ANY_SOURCE),
        ("BUILD_MPI_ANY_TAG", mpi.ANY_TAG),
        ("BUILD_MPI_LIBRARY_VERSION", mpi.Get_library_version().rstrip("\x00")),
    )
    for attribute, runtime_value in expected:
        if getattr(extension, attribute, None) != runtime_value:
            raise RuntimeError(
                f"{extension.__name__} has a different {attribute} than "
                "mpi4py's MPI runtime; rebuild with the matching MPI compiler"
            )
