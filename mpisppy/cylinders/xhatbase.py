###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import abc
import os
import math

import mpisppy.cylinders.spoke as spoke
from mpisppy.utils.xhat_eval import Xhat_Eval

class XhatInnerBoundBase(spoke.InnerBoundNonantSpoke):

    @abc.abstractmethod
    def xhat_extension(self):
        raise NotImplementedError


    def xhat_prep(self):
        ## for later
        self.verbose = self.opt.options["verbose"] # typing aid

        if not isinstance(self.opt, Xhat_Eval):
            raise RuntimeError(f"{self.__class__.__name__} must be used with Xhat_Eval.")

        xhatter = self.xhat_extension()

        ### begin iter0 stuff
        xhatter.pre_iter0()
        if self.opt.extensions is not None:
            self.opt.extobject.pre_iter0()  # for an extension
        self.opt._save_original_nonants()

        self.opt._lazy_create_solvers()  # no iter0 loop, but we need the solvers

        self.opt._update_E1()
        if abs(1 - self.opt.E1) > self.opt.E1_tolerance:
            raise ValueError(f"Total probability of scenarios was {self.opt.E1} "+\
                                 f"(E1_tolerance is {self.opt.E1_tolerance})")
        ### end iter0 stuff (but note: no need for iter 0 solves in an xhatter)

        xhatter.post_iter0()
        if self.opt.extensions is not None:
            self.opt.extobject.post_iter0()  # for an extension

        self.opt._save_nonants() # make the cache

        # Optional: try an xhat loaded from a file before the normal
        # xhatter main loop. See doc/src/xhat_from_file.rst.
        self._try_file_xhat()

        return xhatter

    def _try_file_xhat(self):
        """Evaluate a file-supplied xhat once, before the main loop.

        Gated on ``options['xhat_from_file']`` being a path. Two-stage
        only for V1 (matches ``ciutils.read_xhat``). Hard-fails on
        missing file, length mismatch, or multi-stage. Restores
        nonants afterwards so the spoke's main loop sees clean state.
        """
        path = self.opt.options.get("xhat_from_file", None)
        if not path:
            return
        # Lazy import to avoid any startup-time coupling and to keep
        # numpy out of the non-feature path.
        from mpisppy.confidence_intervals import ciutils

        if self.opt.multistage:
            raise RuntimeError(
                "--xhat-from-file is two-stage only; multi-stage support "
                "is planned as a follow-up. See "
                "doc/xhat_from_file_design.md."
            )
        if not os.path.exists(path):
            raise RuntimeError(
                f"--xhat-from-file={path!r} does not exist."
            )

        nonant_cache = ciutils.read_xhat(path, num_stages=2)
        # Length check against the root-node nonant count of an
        # arbitrary local scenario (all local scenarios share the
        # same nonant count by PH invariant).
        any_s = next(iter(self.opt.local_scenarios.values()))
        expected = len(any_s._mpisppy_data.nonant_indices)
        got = len(nonant_cache["ROOT"])
        if got != expected:
            raise RuntimeError(
                f"--xhat-from-file vector length {got} does not match the "
                f"problem's root-node nonant count {expected} (file={path!r})."
            )

        if self.cylinder_rank == 0:
            print(f"[xhat-from-file] evaluating {path!r} "
                  f"({expected} nonants)")
        Eobj = self.opt.evaluate(nonant_cache)
        # Restore nonants so the main loop starts from clean state.
        self.opt._restore_nonants()
        if Eobj is not None and math.isfinite(Eobj):
            self.update_if_improving(Eobj)
        elif self.cylinder_rank == 0:
            print(f"[xhat-from-file] candidate gave Eobj={Eobj!r}; not "
                  f"updating inner bound")
