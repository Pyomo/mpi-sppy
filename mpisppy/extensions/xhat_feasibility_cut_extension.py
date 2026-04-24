###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Hub-side extension that receives feasibility cuts emitted by xhat spokes
and installs them into every local scenario so an infeasible xhat is not
revisited.

See doc/xhat_feasibility_cuts_design.md (and the user-facing
doc/src/xhat_feasibility_cuts.rst) for the design and usage.
"""

import pyomo.environ as pyo
from pyomo.core.expr.numeric_expr import LinearExpression

import mpisppy.utils.sputils as sputils
from mpisppy.cylinders.spwindow import Field
from mpisppy.extensions.extension import Extension


class XhatFeasibilityCutExtension(Extension):
    """Install feasibility cuts produced by xhat spokes.

    Wire format (matches Field.XHAT_FEASIBILITY_CUT):
      [ row_0, row_1, ..., row_{K-1}, count ]
    where each row is
      [ rhs_constant, nonant_coef_1, ..., nonant_coef_N ]
    and ``count`` is the number of valid rows this batch (0..K).

    A row encodes the constraint
        rhs_constant + sum_i nonant_coef_i * x_i >= 0
    over the local scenario's nonants (keyed by
    ``s._mpisppy_data.nonant_indices`` in insertion order).

    Binary-only first-stage is a precondition; setup_hub hard-fails
    if any nonant is not binary. See the design doc.
    """

    def __init__(self, spbase_object):
        super().__init__(spbase_object)
        self._cuts_per_iter = int(
            self.opt.options.get("xhat_feasibility_cuts_count", 0)
        )
        if self._cuts_per_iter <= 0:
            raise RuntimeError(
                "XhatFeasibilityCutExtension was attached but "
                "options['xhat_feasibility_cuts_count'] is not a positive "
                "integer. Enable the feature via --xhat-feasibility-cuts-count "
                "or remove the extension from the hub."
            )
        self._nonant_len = None  # filled in at setup_hub
        self._row_len = None
        self._recv_buffer = None  # filled in at register_receive_fields
        self._install_counter = 0  # monotonic key for the ConstraintList

    # ---- binary-only precondition check ----------------------------------

    @staticmethod
    def _is_binary_var(v):
        """Return True if v is a binary pyomo VarData.

        Accepts either ``v.is_binary()`` (if present) or a domain check
        (``Binary``, ``BooleanSet``), and treats a 0/1-bounded integer
        variable as binary so the user isn't punished for a common
        modeling style.
        """
        if v.is_binary():
            return True
        if v.is_integer():
            lb, ub = v.bounds
            if lb is not None and ub is not None and lb >= 0 and ub <= 1:
                return True
        return False

    def _assert_all_nonants_binary(self):
        """Scan every nonant on every local scenario / node; raise on non-binary.

        For a bundle ``s``, ``s._mpisppy_data.nonant_indices`` is the
        canonical (consolidated) nonant set — that is what cuts will be
        installed against, so that is also what we validate.
        """
        for sname, s in self.opt.local_scenarios.items():
            for ndn_i, v in s._mpisppy_data.nonant_indices.items():
                if not self._is_binary_var(v):
                    raise RuntimeError(
                        "--xhat-feasibility-cuts-count > 0 requires every "
                        "first-stage (nonant) variable to be binary; found "
                        f"non-binary nonant {v.name!r} (key {ndn_i}) on "
                        f"local scenario {sname!r} with domain {v.domain}. "
                        "The first-milestone feasibility-cut generator is "
                        "no-good-only. Support for integer and continuous "
                        "first-stage variables is planned as a follow-up "
                        "milestone (pyomo Benders / Farkas extension)."
                    )

    # ---- Extension hooks -------------------------------------------------

    def setup_hub(self):
        self._assert_all_nonants_binary()
        # Nonant count per local scenario; cuts come packed against this.
        # All local scenarios must have the same nonant count (that is
        # already a PH invariant), so read from one.
        any_s = next(iter(self.opt.local_scenarios.values()))
        self._nonant_len = len(any_s._mpisppy_data.nonant_indices)
        self._row_len = 1 + self._nonant_len
        # ConstraintList per local scenario to accumulate incoming cuts.
        for s in self.opt.local_scenarios.values():
            s._mpisppy_model.xhat_feasibility_cuts = pyo.Constraint(pyo.Any)

    def register_send_fields(self):
        # We do not send anything; the spoke is the sender.
        return

    def register_receive_fields(self):
        spcomm = self.opt.spcomm
        ranks = spcomm.fields_to_ranks.get(Field.XHAT_FEASIBILITY_CUT, [])
        if not ranks:
            # No xhatter spoke is emitting cuts this run; extension is a no-op.
            self._recv_buffer = None
            return
        # For now assume a single emitting spoke (mirrors cross-scen).
        assert len(ranks) == 1, (
            "XhatFeasibilityCutExtension expects exactly one spoke to emit "
            f"Field.XHAT_FEASIBILITY_CUT; found {len(ranks)}."
        )
        self._recv_buffer = spcomm.register_recv_field(
            Field.XHAT_FEASIBILITY_CUT, ranks[0]
        )

    def sync_with_spokes(self):
        if self._recv_buffer is None:
            return
        if self._recv_buffer.is_new():
            self._install_cuts(self._recv_buffer.array())

    # ---- Cut installation ------------------------------------------------

    def _install_cuts(self, buf):
        """Unpack the buffer and append each row to every local scenario.

        Buffer layout: K rows of length row_len followed by a scalar
        ``count``. Rows 0..count-1 are valid; the rest are ignored.
        """
        n_cuts = int(buf[-1])
        if n_cuts <= 0:
            return
        row_len = self._row_len
        for k in range(n_cuts):
            row = buf[k * row_len : (k + 1) * row_len]
            rhs_constant = float(row[0])
            coefs = [float(c) for c in row[1:]]
            # A defensive zero-row check — the spoke may have written
            # fewer cuts than the header claims if we ever tighten that.
            if rhs_constant == 0.0 and all(c == 0.0 for c in coefs):
                continue
            self._install_counter += 1
            key = self._install_counter
            for s in self.opt.local_scenarios.values():
                linear_vars = list(s._mpisppy_data.nonant_indices.values())
                # Constraint form: rhs_constant + sum coef_i x_i >= 0
                # LinearExpression has a single `constant=` slot; we
                # pass rhs_constant as that constant and 0 as the RHS.
                expr = LinearExpression(
                    constant=rhs_constant,
                    linear_coefs=coefs,
                    linear_vars=linear_vars,
                )
                s._mpisppy_model.xhat_feasibility_cuts[key] = (0.0, expr, None)
                if sputils.is_persistent(s._solver_plugin):
                    s._solver_plugin.add_constraint(
                        s._mpisppy_model.xhat_feasibility_cuts[key]
                    )
