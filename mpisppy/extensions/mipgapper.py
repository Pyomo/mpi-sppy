###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
""" Adaptive mipgap controller. Reads inner/outer bound gaps from the
    hub/spoke comm at each miditer and tightens the subproblem mipgap
    to ``problem_rel_gap * mipgap_ratio`` (clamped from above by
    ``starting_mipgap``). For a static, iteration-indexed schedule,
    use ``--mipgaps-json`` instead, which feeds solver_options_layers
    directly without requiring this extension.
"""

import warnings

from mpisppy import global_toc
import mpisppy.extensions.extension
import mpisppy.utils.sputils as sputils


class Gapper(mpisppy.extensions.extension.Extension):

    def __init__(self, ph):
        self.ph = ph
        self.cylinder_rank = self.ph.cylinder_rank
        self.gapperoptions = self.ph.options["gapperoptions"]  # required
        self.starting_mipgap = self.gapperoptions.get("starting_mipgap", None)
        self.mipgap_ratio = self.gapperoptions.get("mipgap_ratio", None)
        # Compatibility shim: programmatic callers used to set
        # gapperoptions["mipgapdict"] for a static schedule.
        # That role is now filled by solver_options_layers
        # (cfg_vanilla.add_gapper does this for --mipgaps-json
        # automatically). Translate any legacy mipgapdict into
        # after_iter layers on the host PHBase so existing user
        # scripts keep producing the same per-iteration mipgap, and
        # warn so callers migrate over time.
        mipgapdict = self.gapperoptions.get("mipgapdict")
        self._static_compat = False
        if mipgapdict is not None:
            warnings.warn(
                "Gapper's static mipgapdict is deprecated. The "
                "schedule is being translated to "
                "solver_options_layers automatically; remove the "
                "Gapper extension and instead pass --mipgaps-json "
                "on the CLI, or append solver_options_layer("
                "('after_iter', N), {'mipgap': v}) entries to "
                "options['solver_options_layers'] directly.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Insert just before the reserved dynamic_gapper layer
            # so it remains last (and so wins over these for any
            # adaptive writes).
            for N in sorted(mipgapdict):
                self.ph.solver_options_layers.insert(
                    -1,
                    sputils.solver_options_layer(
                        ("after_iter", int(N)),
                        {"mipgap": mipgapdict[N]},
                    ),
                )
            self._static_compat = True
            return
        if self.starting_mipgap is None:
            raise RuntimeError(
                f"{self.ph._get_cylinder_name()}: Gapper auto-mode "
                "requires starting_mipgap to be set."
            )

    def _print_msg(self, msg):
        global_toc(
            f"{self.ph._get_cylinder_name()} {self.__class__.__name__}: {msg}",
            self.cylinder_rank == 0,
        )

    def set_mipgap(self, mipgap):
        """ Set the runtime-adaptive mipgap.

        Writes into the PHBase dynamic_gapper layer, which sits at
        the top of solver_options_layers and so overrides every
        CLI-configured mipgap when the fold is taken at solve time.
        """
        mipgap = float(mipgap)
        layer_options = self.ph._dynamic_solver_options_layer["options"]
        oldgap = layer_options.get("mipgap")
        if oldgap is None or oldgap != mipgap:
            oldgap_str = "None" if oldgap is None else f"{oldgap*100:.3f}%"
            self._print_msg(
                f"Changing mipgap from {oldgap_str} to {mipgap*100:.3f}%")
        layer_options["mipgap"] = mipgap

    def pre_iter0(self):
        if self._static_compat:
            return
        # spcomm not yet set in `__init__`, so check this here
        if self.ph.spcomm is None:
            raise RuntimeError(
                "Automatic gapper can only be used with cylinders -- "
                "needs both an upper bound and lower bound cylinder")
        self.set_mipgap(self.starting_mipgap)

    def post_iter0(self):
        return

    def _autoset_mipgap(self):
        self.ph.spcomm.receive_innerbounds()
        self.ph.spcomm.receive_outerbounds()
        _, problem_rel_gap = self.ph.spcomm.compute_gaps()
        subproblem_rel_gap = problem_rel_gap * self.mipgap_ratio
        if subproblem_rel_gap < self.starting_mipgap:
            self.set_mipgap(subproblem_rel_gap)
        elif self.ph._PHIter == 1:
            # Reapply the starting cap so the first iterk solve uses
            # it; later iterk solves keep whatever set_mipgap last
            # wrote into the dynamic layer.
            self.set_mipgap(self.starting_mipgap)

    def miditer(self):
        if self._static_compat:
            return
        self._autoset_mipgap()

    def enditer(self):
        return

    def post_everything(self):
        return
