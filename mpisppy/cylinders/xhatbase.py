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
from mpisppy.cylinders.spwindow import Field
from mpisppy.utils.xhat_eval import Xhat_Eval

class XhatInnerBoundBase(spoke.InnerBoundNonantSpoke):

    # Advertise the optional feasibility-cut send buffer. When
    # cfg.xhat_feasibility_cuts_count == 0 the buffer is size 1 (just the
    # trailing count slot) and never written; see spwindow.FieldLengths
    # and XhatBase._try_one.
    send_fields = (*spoke.InnerBoundNonantSpoke.send_fields,
                   Field.XHAT_FEASIBILITY_CUT,)

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

        Gated on ``options['xhat_from_file']`` being a path. The file may be

        * a ``.csv`` written by ``sputils.write_nonant_tree_csv``
          (``node_name, variable_name, value``; node-local names) -- any
          number of stages, matched to the model by variable name, or
        * a ``.npy`` holding a bare ROOT vector (``ciutils.read_xhat``) --
          two-stage only, matched by position.

        Hard-fails on a missing file, a length/coverage mismatch, or a
        multi-stage ``.npy``. Restores nonants afterwards so the spoke's
        main loop sees clean state.
        """
        path = self.opt.options.get("xhat_from_file", None)
        if not path:
            return
        if not os.path.exists(path):
            raise RuntimeError(
                f"--xhat-from-file={path!r} does not exist."
            )

        if path.endswith(".csv"):
            nonant_cache = self._read_xhat_csv(path)
        else:
            # Lazy import to keep numpy out of the non-feature path.
            from mpisppy.confidence_intervals import ciutils
            if self.opt.multistage:
                raise RuntimeError(
                    "--xhat-from-file with a .npy file is two-stage only; "
                    "use the .csv format (node_name, variable_name, value) "
                    "for multi-stage. See "
                    "doc/designs/multistage_xhat_write_design.md."
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

        n_nonants = sum(len(v) for v in nonant_cache.values())
        if self.cylinder_rank == 0:
            print(f"[xhat-from-file] evaluating {path!r} "
                  f"({n_nonants} nonants)")
        try:
            Eobj = self.opt.evaluate(nonant_cache)
        except Exception as e:
            # Treat an evaluation/solver failure as "no usable objective"
            # (Eobj=None) so the run continues, but surface it so a real
            # failure is not silently swallowed behind a bare Eobj=None.
            Eobj = None
            if self.cylinder_rank == 0:
                print(f"[xhat-from-file] evaluate failed: {e!r}; "
                      f"treating candidate as Eobj=None")
        # Same predicate XhatBase._try_one uses to detect infeasibility
        # in some scenario. When True and feasibility cuts are enabled,
        # pack a no-good cut so the hub extension can install it on
        # every scenario — same path the regular xhatter takes.
        infeasP = 0.0
        try:
            infeasP = self.opt.no_incumbent_prob()
        except Exception:
            pass
        if infeasP != 0.:
            from mpisppy.extensions.xhatbase import pack_no_good_feasibility_cut
            try:
                emitted = pack_no_good_feasibility_cut(self.opt)
            except Exception as e:
                emitted = False
                if self.cylinder_rank == 0:
                    print(f"[xhat-from-file] feasibility-cut emit failed: {e!r}")
            if self.cylinder_rank == 0:
                tag = "emitted" if emitted else "skipped"
                print(f"[xhat-from-file] candidate infeasible "
                      f"(infeasP={infeasP}); feasibility cut {tag}")
            Eobj = None
        # Restore nonants so the main loop starts from clean state.
        self.opt._restore_nonants()
        if Eobj is not None and math.isfinite(Eobj):
            self.update_if_improving(Eobj)
        elif self.cylinder_rank == 0:
            print(f"[xhat-from-file] candidate gave Eobj={Eobj!r}; not "
                  f"updating inner bound")

    def _read_xhat_csv(self, path):
        """Read a canonical by-name xhat CSV into a ``{node: ndarray}``
        cache for this spoke's local scenarios (any number of stages).

        The CSV is keyed by node-local variable name, so the per-node
        order is resolved from the local scenarios' ``nonant_vardata_list``
        (matching the writer's name-localization), then
        ``sputils.read_nonant_tree_csv`` orders the values. The file may
        carry more nodes than this rank needs; only the local nodes are
        read.
        """
        from mpisppy.utils import sputils
        bundling = getattr(self.opt, "bundling", False)
        node_varname_order = dict()
        for sname, s in self.opt.local_scenarios.items():
            for node in s._mpisppy_node_list:
                if node.name in node_varname_order:
                    continue
                node_varname_order[node.name] = [
                    sputils._node_local_nonant_name(var.name, sname, bundling)
                    for var in node.nonant_vardata_list
                ]
        return sputils.read_nonant_tree_csv(path, node_varname_order)
